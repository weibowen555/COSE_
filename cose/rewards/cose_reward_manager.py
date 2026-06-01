"""
COSE Reward Manager — cross-role reward computation.

4-role pipeline: Proposer → Validator → Solver → Judge

Cross-role reward flow (binary format gate):
- Proposer reward  ← format × (val_score + difficulty) / 2.
                     format ∈ {0, 1}; format = 0 collapses the reward to 0.
                     difficulty(p) = 4 · p · (1 − p), the bell-curve "zone of
                     proximal development" — peaks at p = 0.5 (Solver right
                     half the time = learning frontier), 0 at p ∈ {0, 1}.
- Validator reward ← directional calibration between val_score and judge_score
                     (1 if agree, 0 if disagree or bad format).
- Solver reward    ← format × judge_score, format ∈ {0, 1}.
- Judge reward     ← format only, ∈ {0, 1}.

The multiplicative coupling means format failure (no/imbalanced tags) costs
the entire reward — preventing the model from trading format for content.
The bell-curve difficulty replaces the older (1 − p)
"harder-is-better" term, which created a Solver-dominance arms race where p
collapsed toward 0 (or, after Solver self-improvement, toward 1) and the
Proposer's gradient signal evaporated.

A hard quality gate at val_score < QUALITY_GATE drops the difficulty term
so low-quality questions cannot ride a favorable p value into reward.

Confidence-based importance weights are computed separately in the trainer
and stored as `sample_importance` — they do NOT multiply rewards.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Tag extraction helpers
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag>."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_scores(text: str) -> List[float]:
    """
    Extract all numeric scores from text — MAE-style permissive cascade.

    Priority:
    1. <score>...</score> tags. Inside, accept (in order): fraction (e.g. "7/10"),
       float (e.g. "7.5"), integer (e.g. "8"). One score extracted per tag pair.
    2. Fallback if no tags or all tags failed to parse: take the LAST numerical
       value (fraction / float / integer) anywhere in the text.

    Matches MAE's `extract_score_from_tags` so parse-failure rates are comparable
    between the two pipelines.
    """
    from fractions import Fraction

    extracted_scores: List[float] = []

    for match in re.finditer(r"<score>(.*?)</score>", text, re.DOTALL):
        tag_content = match.group(1).strip()
        score_value: Optional[float] = None

        # Priority 1: fraction "n/m"
        frac_match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", tag_content)
        if frac_match:
            try:
                num, denom = int(frac_match.group(1)), int(frac_match.group(2))
                if denom != 0:
                    score_value = float(Fraction(num, denom))
            except (ValueError, ZeroDivisionError):
                pass

        # Priority 2: float
        if score_value is None:
            float_match = re.search(r"\b(\d+\.\d+)\b", tag_content)
            if float_match:
                try:
                    score_value = float(float_match.group(1))
                except ValueError:
                    pass

        # Priority 3: integer
        if score_value is None:
            integer_match = re.search(r"\b(\d+)\b", tag_content)
            if integer_match:
                try:
                    score_value = float(integer_match.group(1))
                except ValueError:
                    pass

        if score_value is not None:
            extracted_scores.append(score_value)

    # Fallback: no parseable <score> tags → grab the last number anywhere in text.
    if not extracted_scores:
        all_numbers = re.findall(r"\b\d+/\d+\b|\b\d+\.\d+\b|\b\d+\b", text)
        if all_numbers:
            last_num = all_numbers[-1]
            if "/" in last_num:
                parts = last_num.split("/")
                try:
                    if int(parts[1]) != 0:
                        extracted_scores.append(float(Fraction(int(parts[0]), int(parts[1]))))
                except (ValueError, ZeroDivisionError):
                    pass
            else:
                try:
                    extracted_scores.append(float(last_num))
                except ValueError:
                    pass

    return extracted_scores


def _has_think(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def _count_tags(text: str, tag: str) -> Tuple[int, int]:
    """Count opening and closing occurrences of <tag> / </tag>."""
    return text.count(f"<{tag}>"), text.count(f"</{tag}>")


def _proposer_format_reward(text: str) -> float:
    """
    COSE Proposer format reward — binary {0, 1} on the single
    `<question></question>` contract.

    1.0 → exactly one balanced <question></question> pair.
    0.0 → anything else (empty, missing, imbalanced, or >1 pairs).
    """
    open_q, close_q = _count_tags(text, "question")
    if open_q == 1 and close_q == 1:
        return 1.0
    return 0.0


def _solver_format_reward(text: str) -> float:
    """
    Solver format reward — binary {0, 1} on the single
    `<answer></answer>` contract.

    1.0 → exactly one balanced <answer></answer> pair.
    0.0 → anything else (empty, missing, imbalanced, or >1 pairs).
    """
    open_count, close_count = _count_tags(text, "answer")
    if open_count == 1 and close_count == 1:
        return 1.0
    return 0.0


def _judge_format_reward(text: str) -> float:
    """
    COSE Judge format reward — binary {0, 1} on the single
    `<score>X</score>` contract (also reused by Validator, same schema).

    1.0 → exactly one balanced <score></score> pair.
    0.0 → anything else (empty, missing, imbalanced, or >1 pairs).
    """
    open_count, close_count = _count_tags(text, "score")
    if open_count == 1 and close_count == 1:
        return 1.0
    return 0.0


# Hard quality gate (MAE-style): if val_score < this threshold, drop the
# difficulty term so low-quality questions can't ride a favorable p value.
# Default kept at 0.7 for backwards compatibility; override via
# `cose.cose.quality_gate=<float>` (e.g. 0.4 for weaker base models that
# can't clear 0.7 early in training, leaving the buffer empty).
QUALITY_GATE = 0.7


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class COSERewardManager:
    """
    Stateless reward manager for COSE 4-role self-play.

    All reward functions accept the role's own data PLUS the downstream
    role's decoded texts, so rewards flow across roles.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int = 0,
        output_path: str = "./",
        prompt_manager=None,
        confidence_signal: str = "self_certainty",
        calibration_weight: float = 0.5,
        quality_gate: float = QUALITY_GATE,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.output_path = output_path
        self.prompt_manager = prompt_manager
        self.confidence_signal = confidence_signal
        self.quality_gate = float(quality_gate)
        # Trainer flips this on/off per step. When True, compute_proposer_reward
        # uses a curriculum-warmup formula that bypasses the hard val_score gate
        # (format reward only) so a base model can learn the <question> schema
        # before being filtered by Validator quality.
        self.in_quality_gate_warmup = False
        # Solver reward = format × [α · judge + (1 − α) · calibration],
        # where calibration = 1 − |c_solver − judge|. α = calibration_weight.
        # α = 1.0 recovers the previous "format × judge" formula (no
        # calibration target). α = 0.0 makes the reward purely about
        # confidence-vs-correctness agreement.
        self.calibration_weight = float(calibration_weight)

    def set_prompt_manager(self, prompt_manager):
        """Called by the trainer after initialization."""
        self.prompt_manager = prompt_manager

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def decode_responses(self, data) -> List[str]:
        """Decode response texts from the batch. Uses 'responses' key."""
        responses = data.batch["responses"]
        texts = []
        for i in range(responses.shape[0]):
            resp_ids = responses[i]
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                mask = resp_ids != self.tokenizer.pad_token_id
                resp_ids = resp_ids[mask]
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    def _make_token_level_reward(self, per_sample_rewards: torch.Tensor, data) -> torch.Tensor:
        """
        Place per-sample scalar rewards at the last non-pad response token.
        Output shape: (bs, response_length).
        """
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
            mask = responses != pad_id
        else:
            mask = torch.ones_like(responses, dtype=torch.bool)

        for i in range(responses.shape[0]):
            nonpad_positions = torch.nonzero(mask[i], as_tuple=False).flatten()
            if len(nonpad_positions) > 0:
                last = nonpad_positions[-1].item()
                reward_tensor[i, last] = per_sample_rewards[i]

        return reward_tensor

    # -------------------------------------------------------------------
    # Cross-role reward computation
    # -------------------------------------------------------------------

    def compute_proposer_reward(
        self, proposer_data, validator_texts: List[str],
        judge_scores: List[float],
        judge_valid: Optional[List[bool]] = None,
        validator_scores: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, dict, List[float]]:
        """
        Proposer reward = format × (val_score + difficulty) / 2,
        with format ∈ {-1, 0.5, 1.0} (format failure → r = -1 hard penalty).

        Components:
        - val_score ∈ [0,1]: Validator's <score> on question i, normalized as
          (X−1)/9. Encodes question quality.
        - difficulty(p) = 1 − p ∈ [0,1]: harder-is-better term. Rewards the
          Proposer for questions the Solver fails (low p).
        - format ∈ {-1, 0.5, 1.0}: well-formedness of the <question> tags;
          gates the entire content reward multiplicatively.

        Hard quality gate: if val_score < QUALITY_GATE (0.7), the difficulty
        term is dropped (content_term = val_score). Prevents low-quality
        questions from riding a favorable p value into reward.

        `judge_valid` is accepted for backward compatibility but no longer
        gates the difficulty term: the trainer sets failed-parse Judge cells
        to 0.0 in the ensemble mean, so a Judge that can't evaluate pulls p
        toward 0 — and the bell curve naturally penalizes p ≈ 0 too.

        If `validator_scores` is provided (length matching proposer batch), it
        is used directly as val_score per sample. This lets the trainer do the
        n_samples Validator-ensemble aggregation upstream and pass a single
        per-question score here. Otherwise val_score is parsed from
        validator_texts (single-rollout legacy path).

        Returns (reward_tensor, metrics, val_scores_per_sample) — the raw
        val_scores are still exposed for Solver importance in Phase 3.

        Bad proposer format (no <question>) → reward = -1.0.
        """
        proposer_texts = self.decode_responses(proposer_data)
        bs = len(proposer_texts)
        rewards = torch.zeros(bs)
        val_scores: List[float] = []
        if judge_valid is None:
            judge_valid = [True] * len(judge_scores)
        metrics = {
            "proposer/reward": [],
            "proposer/has_question": [],
            "proposer/validator_quality_score": [],
            "proposer/difficulty": [],
            "proposer/format_reward": [],
            "proposer/quality_gate_passed": [],
            "proposer/judge_score_p": [],
            "proposer/judge_valid_per_question": [],
        }

        has_question_flags: List[bool] = []
        use_pre_aggregated = validator_scores is not None
        for i in range(bs):
            question = _extract_tag(proposer_texts[i], "question")
            has_question_flags.append(question is not None)

            val_score = 0.0
            if question is not None:
                if use_pre_aggregated and i < len(validator_scores):
                    val_score = float(np.clip(validator_scores[i], 0.0, 1.0))
                elif i < len(validator_texts):
                    scores = _extract_scores(validator_texts[i])
                    if scores:
                        raw_score = max(1.0, min(10.0, scores[-1]))
                        val_score = (raw_score - 1.0) / 9.0
            val_scores.append(val_score)

        p_arr = np.asarray(judge_scores[:bs], dtype=np.float32)
        p_arr = np.clip(p_arr, 0.0, 1.0)
        # Harder-is-better difficulty term: difficulty(p) = 1 − p.
        difficulty_raw = 1.0 - p_arr
        # judge_valid is now a pure diagnostic flag (parse-fail = 0 in the
        # mean upstream). Quality gate on val_score still suppresses reward.
        judge_valid_arr = np.asarray(judge_valid[:bs], dtype=bool)
        if judge_valid_arr.shape[0] < bs:
            judge_valid_arr = np.concatenate([
                judge_valid_arr, np.zeros(bs - judge_valid_arr.shape[0], dtype=bool),
            ])

        for i in range(bs):
            metrics["proposer/validator_quality_score"].append(val_scores[i])
            p = float(p_arr[i]) if i < len(p_arr) else 0.0
            metrics["proposer/judge_score_p"].append(p)
            metrics["proposer/judge_valid_per_question"].append(
                1.0 if (i < len(judge_valid_arr) and bool(judge_valid_arr[i])) else 0.0
            )

            fmt_reward = (
                _proposer_format_reward(proposer_texts[i])
                if has_question_flags[i] else 0.0
            )
            metrics["proposer/format_reward"].append(fmt_reward)

            # No hard quality gate: low val_score becomes low reward via the
            # multiplicative content term, not a -0.3 cliff. The Proposer still
            # has incentive to produce high-quality questions because val_score
            # multiplies into the reward, but a 4/10 question now gets a small
            # positive reward instead of being indistinguishable from a format
            # failure.
            diff_i = float(difficulty_raw[i]) if (
                i < len(difficulty_raw) and has_question_flags[i]
            ) else 0.0
            metrics["proposer/difficulty"].append(diff_i)
            # Backwards-compat metric: 1.0 always when format passes, since
            # there is no quality gate to fail anymore.
            metrics["proposer/quality_gate_passed"].append(
                1.0 if has_question_flags[i] else 0.0
            )

            # Reward formula (binary format gate):
            #   fmt = 0 (any format failure)        → r = 0       (multiplicative collapse)
            #   in format-only warmup, fmt = 1      → r = 1
            #   well-formed, post-warmup            → r = fmt × (val + 4·p·(1-p)) / 2
            metrics["proposer/has_question"].append(1.0 if has_question_flags[i] else 0.0)
            if self.in_quality_gate_warmup:
                # Format-only warmup so a base model can discover the
                # <question> schema before val_score and difficulty kick in.
                rewards[i] = fmt_reward
            else:
                content_term = (val_scores[i] + diff_i) / 2.0
                rewards[i] = fmt_reward * content_term

            metrics["proposer/reward"].append(rewards[i].item())

        reward_tensor = self._make_token_level_reward(rewards, proposer_data)
        return reward_tensor, metrics, val_scores

    def compute_validator_format_only_reward(
        self, validator_data,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Format-only reward for the Validator: scores the <score>X</score>
        contract via `_judge_format_reward` (Validator emits the same schema).

        - Exactly 1 balanced <score> pair → 1.0
        - Anything else                   → 0.0

        Used to train the Validator's format compliance without coupling to
        downstream calibration. Wired in when
        `+cose.cose.format_train_validator_n=<int>` is set.
        """
        validator_texts = self.decode_responses(validator_data)
        bs = len(validator_texts)
        rewards = torch.zeros(bs)
        metrics = {
            "validator_fmt/format_reward": [],
            "validator_fmt/has_score": [],
        }
        for i in range(bs):
            r = _judge_format_reward(validator_texts[i])
            rewards[i] = r
            metrics["validator_fmt/format_reward"].append(r)
            metrics["validator_fmt/has_score"].append(1.0 if r > 0.0 else 0.0)
        reward_tensor = self._make_token_level_reward(rewards, validator_data)
        return reward_tensor, metrics

    def compute_judge_format_only_reward(
        self, judge_data,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Format-only reward for the Judge: scores the <score>X</score>
        contract via `_judge_format_reward`.

        Same rules and rationale as `compute_validator_format_only_reward`.
        Used to train the Judge's format compliance even when
        `cose.train_judge=False`. Wired in when
        `+cose.cose.format_train_judge_n=<int>` is set.
        """
        judge_texts = self.decode_responses(judge_data)
        bs = len(judge_texts)
        rewards = torch.zeros(bs)
        metrics = {
            "judge_fmt/format_reward": [],
            "judge_fmt/has_score": [],
        }
        for i in range(bs):
            r = _judge_format_reward(judge_texts[i])
            rewards[i] = r
            metrics["judge_fmt/format_reward"].append(r)
            metrics["judge_fmt/has_score"].append(1.0 if r > 0.0 else 0.0)
        reward_tensor = self._make_token_level_reward(rewards, judge_data)
        return reward_tensor, metrics

    def compute_validator_reward(
        self, validator_data, validator_texts: List[str],
        judge_scores: List[float],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Validator reward = predictive calibration.

        The validator evaluates question quality. A well-calibrated validator
        should predict downstream solvability: questions it rates highly should
        be answerable (high judge score), questions it rates poorly should be
        hard or unsolvable (low judge score).

        We measure this as agreement in direction:
        - Both high (val >= 0.5 and judge >= 0.5) → reward = 1.0
        - Both low  (val <  0.5 and judge <  0.5) → reward = 1.0
        - Disagree → reward = 0.0
        - Bad format → reward = 0.0

        This trains the validator to be a reliable signal for question quality
        without forcing exact score matching between quality and correctness.
        """
        bs = len(validator_texts)
        rewards = torch.zeros(bs)
        metrics = {
            "validator/reward": [],
            "validator/format_ok": [],
            "validator/quality_score": [],
            "validator/calibration": [],
        }

        for i in range(bs):
            scores = _extract_scores(validator_texts[i])
            format_ok = len(scores) > 0 and _has_think(validator_texts[i])

            if not format_ok:
                rewards[i] = 0.0
                metrics["validator/format_ok"].append(0.0)
                metrics["validator/quality_score"].append(0.0)
                metrics["validator/calibration"].append(0.0)
            else:
                metrics["validator/format_ok"].append(1.0)
                raw_score = max(1.0, min(10.0, scores[-1]))
                val_normalized = (raw_score - 1.0) / 9.0
                metrics["validator/quality_score"].append(val_normalized)

                judge_score = judge_scores[i] if i < len(judge_scores) else 0.5
                # Directional calibration: do validator and judge agree on high/low?
                val_high = (val_normalized >= 0.5)
                judge_high = (judge_score >= 0.5)
                calibrated = (val_high == judge_high)
                rewards[i] = 1.0 if calibrated else 0.0
                metrics["validator/calibration"].append(1.0 if calibrated else 0.0)

            metrics["validator/reward"].append(rewards[i].item())

        reward_tensor = self._make_token_level_reward(rewards, validator_data)
        return reward_tensor, metrics

    def compute_solver_reward(
        self, solver_data,
        judge_scores: List[float],
        judge_valid: Optional[List[bool]] = None,
        c_solver: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solver reward = format × judge, with format ∈ {-1, 0.5, 1.0}.

        - judge ∈ [0,1]: Judge's <score> on this attempt, normalized.
        - format ∈ {-1, 0.5, 1.0}: gates the entire content reward.

        Format failure (fmt < 0) hard-penalizes to -1 regardless of judge.
        c_solver is accepted for backward compatibility but unused.
        """
        solver_texts = self.decode_responses(solver_data)
        bs = len(solver_texts)
        if judge_valid is None:
            judge_valid = [True] * len(judge_scores)

        rewards = torch.zeros(bs)
        metrics = {
            "solver/reward": [],
            "solver/has_answer": [],
            "solver/judge_score": [],
            "solver/format_reward": [],
            "solver/judge_valid_frac": [],
        }

        for i in range(bs):
            js = judge_scores[i] if i < len(judge_scores) else 0.0
            ok = bool(judge_valid[i]) if i < len(judge_valid) else False
            judge_term = float(np.clip(js, 0.0, 1.0)) if ok else 0.0

            fmt_reward = _solver_format_reward(solver_texts[i])

            metrics["solver/judge_score"].append(float(js))
            metrics["solver/format_reward"].append(fmt_reward)
            metrics["solver/judge_valid_frac"].append(1.0 if ok else 0.0)
            metrics["solver/has_answer"].append(1.0 if fmt_reward > 0.0 else 0.0)

            # Multiplicative coupling: binary format gates judge term.
            #   fmt = 0 (any format failure) → r = 0      (multiplicative collapse)
            #   fmt = 1 (compliant)          → r = judge
            rewards[i] = fmt_reward * judge_term
            metrics["solver/reward"].append(rewards[i].item())

        reward_tensor = self._make_token_level_reward(rewards, solver_data)
        return reward_tensor, metrics

    def compute_judge_reward(
        self, judge_data, judge_texts: List[str],
        solver_confidences: np.ndarray,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Judge reward = format only (binary).

        Range:
            r = 1.0    if exactly one balanced <score>X</score>     (compliant)
            r = 0.0    otherwise (empty, missing, imbalanced, or >1 pairs)

        The Judge is trained purely to produce parseable <score>X</score>
        outputs — no calibration objective, no agreement penalty. The
        rationale: in our setup, Solver/Proposer training is the load-bearing
        part of the loss. The Judge only needs to *reliably emit a numeric
        score* so the downstream rewards have signal. Once the Judge's
        format is solid, the parsed score gives Solver/Proposer gradient.

        solver_confidences is accepted for backward compatibility but unused.
        """
        bs = len(judge_texts)
        rewards = torch.zeros(bs)
        metrics = {
            "judge/reward": [],
            "judge/format_ok": [],
            "judge/format_reward": [],
            "judge/score_given": [],
        }

        for i in range(bs):
            text = judge_texts[i]
            scores = _extract_scores(text)
            fmt_reward = _judge_format_reward(text)
            metrics["judge/format_reward"].append(fmt_reward)

            format_ok = len(scores) > 0 and fmt_reward > 0.0
            metrics["judge/format_ok"].append(1.0 if format_ok else 0.0)

            if format_ok:
                raw_score = max(1.0, min(10.0, scores[-1]))
                normalized = (raw_score - 1.0) / 9.0
                metrics["judge/score_given"].append(normalized)
            else:
                metrics["judge/score_given"].append(0.0)

            # Pure binary format reward.
            #   fmt = 0 (any format failure) → r = 0
            #   fmt = 1 (compliant)          → r = 1
            rewards[i] = float(fmt_reward)
            metrics["judge/reward"].append(rewards[i].item())

        reward_tensor = self._make_token_level_reward(rewards, judge_data)
        return reward_tensor, metrics

    # -------------------------------------------------------------------
    # Confidence extraction (used by trainer)
    # -------------------------------------------------------------------

    def compute_confidences(self, data) -> np.ndarray:
        """
        Extract per-sample mean confidence from old_log_probs.
        Higher = more confident.
        """
        batch = data.batch
        if "old_log_probs" not in batch:
            bs = batch["responses"].shape[0]
            return np.zeros(bs)

        log_probs = batch["old_log_probs"]
        responses = batch["responses"]

        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
            mask = (responses != pad_id).float()
        else:
            mask = torch.ones_like(responses, dtype=torch.float32)

        masked_lp = log_probs * mask
        token_counts = mask.sum(dim=-1).clamp(min=1)
        mean_conf = masked_lp.sum(dim=-1) / token_counts
        return mean_conf.detach().cpu().numpy()
