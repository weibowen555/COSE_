"""
COSE Ray PPO Trainer: Confidence-Orchestrated Self-Evolution.

4-role pipeline with cross-role rewards:
  Proposer → Validator → Solver → Judge

Reward flow:
  - Proposer reward  ← Validator's quality <score> (normalized to [0,1])
  - Validator reward  ← Calibration (quality score vs judge outcome)
  - Solver reward     ← Judge's <score> (normalized to [0,1])
  - Judge reward      ← Format-based

Confidence-based importance:
  - Each role's confidence (mean log-prob) is computed per sample
  - Chain importance = f(c_proposer, c_validator, c_solver, c_judge)
  - Stored as `sample_importance` in batch — weights the PPO loss,
    NOT the reward

Pipeline per step:
  Phase 1: Rollout all 4 roles sequentially (no rewards yet)
  Phase 2: Assign cross-role rewards
  Phase 3: Compute advantages, attach importance weights
  Phase 4: Concat all 4 role batches, single actor update
"""

import gc
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
from torch.utils.data import DataLoader, SequentialSampler
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import (
    compute_advantage,
    compute_response_mask,
    apply_kl_penalty,
    reduce_metrics,
    compute_timing_metrics,
    agg_loss,
)
from verl.utils.dataset.rl_dataset import collate_fn

from cose.trainer.ppo.base_ray_trainer import (
    GeneralIORayPPOTrainer,
    _timer,
)
from cose.utils.tracking import ReasonRLTracking
from cose.utils.dataset.rl_dataset import RLHFDataset
from cose.utils.logging_utils.stdout import PrettyPrinter
from cose.data_construction.constructor import (
    get_gen_general_io_data,
    get_pred_general_io_data,
    get_judge_general_io_data,
)
from cose.trainer.ppo.cose_dataset_manager import COSEDatasetManager
from cose.rewards.cose_reward_manager import (
    COSERewardManager,
    QUALITY_GATE,
    _extract_tag,
    _extract_scores,
)


# Minimal set of metrics logged to wandb — each one carries information no other
# listed metric duplicates. Pairs like (per-answer + per-question, confidence_mean
# + importance_mean) are deduped in favor of the one that drives the reward
# directly. All other computed metrics are dropped before logging.
ESSENTIAL_METRICS = {
    # Framework health (sample-level format rates and Judge parse success).
    "gen_general/proposer/has_question",
    "pred_general/solver/has_answer",
    "cose/judge_valid_frac_per_question",

    # Reward signals — the only source of gradient.
    "gen_general/proposer/reward",
    "pred_general/solver/reward",
    "pred_general/solver/format_reward",
    "pred_general/solver/judge_score",

    # MAE-style additive Proposer reward components.
    "gen_general/proposer/validator_quality_score",
    "gen_general/proposer/difficulty",
    "gen_general/proposer/format_reward",
    "gen_general/proposer/quality_gate_passed",
    "gen_general/proposer/judge_score_p",

    # Per-role importance weights (already summarize cross-role confidence).
    "cose/proposer_importance_mean",
    "cose/solver_importance_mean",
    "cose/solver_importance_p10",
    "cose/solver_importance_p90",
    "cose/solver_importance_at_floor",

    # Buffer health — the canonical "is the framework stuck?" panel.
    "cose/buffer_size",
    "cose/buffer_growth_per_step",
    "cose/buffer_empty",
    "cose/buffer_admission_kept",
    "cose/buffer_admission_dropped",
    "cose/format_warmup_active",

    # Validator-collapse diagnostics: distribution percentiles + parse rates.
    "cose/val_score_p10",
    "cose/val_score_p50",
    "cose/val_score_p90",
    "cose/val_score_max",
    "cose/val_score_mean",
    "cose/validator_native_parse_rate",
    "cose/validator_gpt_fallback_attempted",
    "cose/validator_gpt_fallback_recovered",
    "cose/validator_gpt_fallback_recover_rate",
    "cose/validator_valid_frac_per_question",

    # Curriculum health: where is judge_score_p living, and how learnable
    # are the questions we're training on?
    "cose/judge_p_p10",
    "cose/judge_p_p50",
    "cose/judge_p_p90",
    "cose/learnability_mean",
    "cose/judge_native_parse_rate",
    "cose/judge_gpt_fallback_attempted",
    "cose/judge_gpt_fallback_recovered",
    "cose/judge_gpt_fallback_recover_rate",

    # Replay diagnostics — staleness of buffer-sampled questions.
    "cose/solver_replay_size",
    "cose/solver_this_step_sampled",
    "cose/solver_replayed_from_past",
    "cose/replay_fresh_frac",
    "cose/replay_age_mean",
    "cose/replay_age_max",

    # Step cadence (Proposer frequency control).
    "cose/propose_step",
    "cose/propose_every_n_steps",
    "cose/num_new_questions",

    # Format-only side training for Validator/Judge.
    "cose/validator_format_train_n",
    "cose/judge_format_train_n",
    "validate_general/validator_fmt/format_reward",
    "validate_general/validator_fmt/has_score",
    "judge_general/judge_fmt/format_reward",
    "judge_general/judge_fmt/has_score",

    # PPO internals.
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/entropy",
}


def _filter_essential_metrics(metrics: dict) -> dict:
    """Keep only essential metrics + anything benchmark-related."""
    def keep(k: str) -> bool:
        if k in ESSENTIAL_METRICS:
            return True
        # Keep benchmark/validation metrics (prefixes vary by eval)
        if k.startswith("val/") or k.startswith("eval/") or k.startswith("benchmark/"):
            return True
        return False
    return {k: v for k, v in metrics.items() if keep(k)}


class COSERayPPOTrainer(GeneralIORayPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cose_config = self.config.cose.get("cose", {})
        self.confidence_signal = cose_config.get("confidence_signal", "self_certainty")
        self.gating_enabled = cose_config.get("gating_enabled", True)
        self.gate_bottom_percentile = cose_config.get("gate_bottom_percentile", 0.3)
        self.curriculum_enabled = cose_config.get("curriculum_enabled", True)
        self.curriculum_std = cose_config.get("curriculum_std", 0.2)
        self.weighting_enabled = cose_config.get("weighting_enabled", True)
        self.selection_strategy = cose_config.get("selection_strategy", "confidence_curriculum")
        # Per-response confidence aggregation: mean over the bottom-fraction
        # of token confidences (most-uncertain tokens). 0.2 ≈ "average over
        # the worst-confidence 20% of response tokens." Set to 1.0 to recover
        # plain mean over all tokens; small values (~0.01) approximate min.
        self.confidence_bottom_frac = float(cose_config.get("confidence_bottom_frac", 0.2))

        # GPT fallback for Validator/Judge score parsing. When the actor
        # rambles without emitting <score>X</score>, regex parse fails and
        # the score defaults to 0 — collapsing downstream gradient. This
        # falls back to a fresh gpt-4.1-nano (or BENCH_JUDGE_MODEL) call
        # with the same template to recover a meaningful score.
        self.judge_parse_fallback = bool(cose_config.get("judge_parse_fallback", True))
        try:
            from cose.rewards.cose_gpt_fallback import GPTScoreFallback
            api_keys = self._load_api_keys_for_fallback()
            self._gpt_fallback = (
                GPTScoreFallback(api_keys=api_keys)
                if self.judge_parse_fallback and api_keys else None
            )
            if self._gpt_fallback and self._gpt_fallback.enabled:
                PrettyPrinter.status(
                    "COSE",
                    f"GPT score-fallback enabled "
                    f"(model={self._gpt_fallback.model}, keys={len(api_keys)})",
                    "success",
                )
        except Exception as e:
            self._gpt_fallback = None
            PrettyPrinter.status("COSE", f"GPT fallback disabled: {e}", "warn")

        # Confidence-prioritized replay buffer
        self.cose_dataset = COSEDatasetManager.remote()

    @staticmethod
    def _load_api_keys_for_fallback() -> list:
        """Reuse the api.json the rest of the pipeline already loads."""
        import json, os
        for path in ("api.json", os.path.expanduser("~/api.json")):
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    keys = data.get("api_keys", []) or [data.get("api_key")]
                    keys = [k for k in keys if k]
                    if keys:
                        return keys
                except Exception:
                    continue
        return []

        PrettyPrinter.status(
            "COSE",
            f"Pipeline: signal={self.confidence_signal}, "
            f"selection={self.selection_strategy}, "
            f"gating={self.gating_enabled}, curriculum={self.curriculum_enabled}, "
            f"weighting={self.weighting_enabled}",
            "success",
        )

    # -------------------------------------------------------------------
    # No-seed bootstrap
    # -------------------------------------------------------------------

    def _create_train_pred_dataloader(self, problem_type, data_len):
        dataset_size = ray.get(self.dataset_manager.get_dataset_size.remote("general"))
        if dataset_size == 0:
            PrettyPrinter.status("COSE", "general dataset empty — skipping solver", "info")
            return None
        return super()._create_train_pred_dataloader(problem_type, data_len)

    def _create_train_judge_dataloader(self, problem_type, data_len):
        dataset_size = ray.get(self.dataset_manager.get_dataset_size.remote("general_pair"))
        if dataset_size == 0:
            PrettyPrinter.status("COSE", "general_pair dataset empty — skipping judge", "info")
            return None
        return super()._create_train_judge_dataloader(problem_type, data_len)

    def _init_seed_dataset(self):
        cose_config = self.config.cose.get("cose", {})
        if not cose_config.get("no_seed", True):
            return super()._init_seed_dataset()
        PrettyPrinter.section_header("COSE No-Seed Mode")
        ray.get(self.dataset_manager.add_general_batch.remote([], self.global_steps))
        ray.get(self.dataset_manager.add_general_pair_batch.remote([], self.global_steps))
        PrettyPrinter.status("COSE", "Empty datasets initialized", "success")

    # -------------------------------------------------------------------
    # Confidence extraction
    # -------------------------------------------------------------------

    def _compute_confidence(self, batch) -> np.ndarray:
        """Per-sample confidence from self_certaintys, aggregated as the
        mean over the BOTTOM `confidence_bottom_frac` (default 20%) of
        response token confidences — i.e., the most-uncertain tail.

        Rationale: a response is only as confident as the part the model
        was uncertain about. Mean over all tokens lets confident filler
        tokens (`<answer>`, whitespace, common phrasing) drown out the
        few decision tokens that actually carry the answer. Min picks
        a single token and is noisy. Mean over bottom-k strikes the
        balance — focused on the uncertain region, but averaged over
        enough tokens to be stable.

        Pad / non-response tokens are excluded before the bottom-k.
        For each row: k = max(1, round(frac × n_valid_tokens)).
        """
        bs = batch.batch["responses"].shape[0]
        responses = batch.batch["responses"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            mask = (responses != pad_id)
        else:
            mask = torch.ones_like(responses, dtype=torch.bool)

        if "self_certaintys" in batch.batch:
            signal = batch.batch["self_certaintys"]
        elif "old_log_probs" in batch.batch:
            signal = batch.batch["old_log_probs"]
        else:
            return np.zeros(bs)

        signal = signal.float()
        out = torch.zeros(bs, dtype=signal.dtype, device=signal.device)
        frac = max(1e-6, min(1.0, self.confidence_bottom_frac))
        for i in range(bs):
            valid = signal[i][mask[i]]
            n_valid = valid.numel()
            if n_valid == 0:
                continue
            k = max(1, int(round(frac * n_valid)))
            bottom_k = torch.topk(valid, k=k, largest=False).values
            out[i] = bottom_k.mean()

        return out.detach().cpu().float().numpy()

    # -------------------------------------------------------------------
    # Importance computation (percentile-based)
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize_confidence(c: np.ndarray) -> np.ndarray:
        """Confidence is already in [0, 1] from `normalized_peakedness` +
        bottom-frac aggregation. We floor at 0.1 to prevent gradient
        starvation: under aggressive bottom-k aggregation many rollouts
        collapse to c ≈ 0, which would zero out their importance and stall
        Solver/Proposer training. The floor preserves magnitude information
        (a c=1 sample still gets 10× the gradient weight of a c≈0 one) but
        ensures every sample contributes some learning signal."""
        return np.clip(np.asarray(c, dtype=np.float64), 0.1, 1.0)

    def _compute_proposer_importance(
        self, val_scores: np.ndarray, c_validator: np.ndarray,
    ) -> np.ndarray:
        """Proposer importance = clip(val_score × c_validator, 0.1, 1.0).

        The Validator certifies the question (val_score is its <score>) and
        c_validator is how confidently the Validator gave that score. Their
        product is a question-level "trustworthy quality" signal that weights
        the Proposer's gradient: the Proposer learns most from questions
        the Validator certified clearly and confidently."""
        v = np.asarray(val_scores, dtype=np.float64)
        c = np.asarray(c_validator, dtype=np.float64)
        return np.clip(v * c, 0.1, 1.0)

    def _compute_validator_importance(self, c_judge: np.ndarray) -> np.ndarray:
        """Validator importance = clip(c_judge, 0.1, 1.0). (Validator is
        graded by Judge agreement; weight by Judge confidence.)"""
        return self._normalize_confidence(c_judge)

    def _compute_solver_importance(
        self, val_scores: np.ndarray, c_validator: np.ndarray, c_judge: np.ndarray,
    ) -> np.ndarray:
        """Solver importance = clip(val_score × c_validator × c_judge, 0.1, 1.0).

        Three factors. val_score × c_validator is the question-level trustworthy
        quality (same as Proposer importance). c_judge is the per-attempt Judge
        confidence on the Solver's specific answer. Product weights the Solver
        gradient: the Solver learns most from (a) questions the Validator
        certified well, (b) where the Judge was confident in its grade."""
        v = np.asarray(val_scores, dtype=np.float64)
        cv = np.asarray(c_validator, dtype=np.float64)
        cj = np.asarray(c_judge, dtype=np.float64)
        return np.clip(v * cv * cj, 0.1, 1.0)

    def _compute_judge_importance(self, c_solver: np.ndarray) -> np.ndarray:
        """Judge importance = clip(c_solver, 0.1, 1.0)."""
        return self._normalize_confidence(c_solver)

    # -------------------------------------------------------------------
    # Phase 1: Rollout only (no reward, no advantage)
    # -------------------------------------------------------------------

    def _rollout_role(self, batch, metrics, timing_raw, role_name, training: bool = True):
        """
        Run rollout + log_prob for a role. Returns (batch, confidences).

        Log probs are ALWAYS computed (needed for per-role confidence,
        which feeds the importance weights of other roles).

        When training=False (inference-only role): skip ref policy and critic
        values since those are only needed for the PPO gradient update.
        """
        # 1. Rollout
        gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        with _timer(f"gen/{role_name}", timing_raw):
            gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_output)
        batch.batch["response_mask"] = compute_response_mask(batch)
        self._balance_batch(batch, metrics=metrics)
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # 2. Log probs (+ self_certaintys) — always needed for confidence
        with _timer(f"old_log_prob/{role_name}", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropy_coeff = self.config.cose.get(
                f"{role_name.split('_')[0]}_entropy_coeff",
                self.config.actor_rollout_ref.actor.entropy_coeff,
            )
            old_log_prob.meta_info["entropy_coeff"] = entropy_coeff
            batch = batch.union(old_log_prob)

        # 3. Reference policy & critic — only needed for PPO update
        if training:
            if self.use_reference_policy:
                with _timer(f"ref/{role_name}", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.use_critic:
                with _timer(f"values/{role_name}", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

        # 4. Confidence from log-probs — used only for Phase 3 importance weights,
        # not logged to wandb (the per-role importance means already summarize it).
        confidences = self._compute_confidence(batch)

        gc.collect()
        return batch, confidences

    # -------------------------------------------------------------------
    # Phase 3: Assign reward + compute advantage
    # -------------------------------------------------------------------

    def _finalize_role(self, batch, reward_tensor, metrics, timing_raw, role_name,
                       sample_importance=None):
        """
        Assign reward tensor and compute advantage for a role.
        Optionally attach sample_importance as a batch field.
        """
        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor

        # Advantage
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            config=self.config.algorithm,
        )

        # Attach importance weight (used by actor update to weight loss)
        if sample_importance is not None:
            bs = batch.batch["responses"].shape[0]
            imp = sample_importance[:bs]
            batch.batch["sample_importance"] = torch.tensor(
                imp, dtype=torch.float32,
            ).to(batch.batch["responses"].device)
        else:
            bs = batch.batch["responses"].shape[0]
            batch.batch["sample_importance"] = torch.ones(
                bs, dtype=torch.float32,
            ).to(batch.batch["responses"].device)

        return batch

    # -------------------------------------------------------------------
    # Text extraction helpers
    # -------------------------------------------------------------------

    def _extract_questions_from_batch(self, batch):
        """Extract questions from proposer responses."""
        questions = []
        responses = batch.batch["responses"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        for i in range(responses.shape[0]):
            ids = responses[i]
            if pad_id is not None:
                ids = ids[ids != pad_id]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            m = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
            questions.append(m.group(1).strip() if m else "")
        return questions

    def _extract_answers_from_batch(self, batch):
        """Extract answers from solver responses."""
        answers = []
        responses = batch.batch["responses"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        for i in range(responses.shape[0]):
            ids = responses[i]
            if pad_id is not None:
                ids = ids[ids != pad_id]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            answers.append(m.group(1).strip() if m else "")
        return answers

    def _decode_batch_responses(self, batch) -> list[str]:
        """Decode all response texts from a batch."""
        responses = batch.batch["responses"]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        texts = []
        for i in range(responses.shape[0]):
            ids = responses[i]
            if pad_id is not None:
                ids = ids[ids != pad_id]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    # -------------------------------------------------------------------
    # Batch builders for downstream roles
    # -------------------------------------------------------------------

    def _build_role_batch(self, items, role_name):
        """Build a DataProto from a list of prompt dicts, ready for rollout."""
        if not items:
            return None

        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        n = len(items)
        remainder = n % n_gpus
        if remainder != 0:
            items = items[:n - remainder]
        if not items:
            return None

        parquet_path = f"{self.config.trainer.default_local_dir}/temp_{role_name}.parquet"
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        pd.DataFrame(items).to_parquet(parquet_path)

        dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
            extra_source_key=f"{role_name}_train",
        )
        if len(dataset) == 0:
            return None

        use_len = len(dataset) - (len(dataset) % n_gpus)
        if use_len <= 0:
            return None

        loader = DataLoader(
            dataset=dataset,
            batch_size=use_len,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=SequentialSampler(dataset),
        )
        batch_dict = next(iter(loader))
        return DataProto.from_single_dict(batch_dict)

    def _make_validator_items(self, questions, n_samples: int = 1):
        """
        Emit `n_samples` Validator prompt-copies per question, contiguous layout:
        [q0·v0, q0·v1, ..., q0·v{n-1}, q1·v0, ...]
        Consumers reshape to (N, n_samples) and aggregate (typically nanmean).
        """
        validator_template = self.prompt_manager.get_template("validator")
        items = []
        for idx, q in enumerate(questions):
            if not q:
                continue
            try:
                prompt_text = validator_template.format(question=q)
            except (KeyError, ValueError):
                prompt_text = f"{validator_template}\n\nQuestion: {q}"
            for s in range(n_samples):
                items.append({
                    "data_source": "validate_general",
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "question": q, "answer": "", "ability": "general",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {
                        "split": "train",
                        "index": idx * n_samples + s,
                        "q_index": idx,
                        "v_index": s,
                        "metric": "validate_general",
                    },
                })
        return items

    def _make_solver_items(self, questions, n_samples: int = 1):
        """
        Emit `n_samples` Solver prompt-copies per question, laid out contiguously:
        [q0·s0, q0·s1, ..., q0·s{n-1}, q1·s0, ..., qN·s{n-1}]
        Consumers can reshape to (N, n_samples) over the flat axis.
        """
        items = []
        for idx, q in enumerate(questions):
            if not q:
                continue
            prompt_text = self.prompt_manager.get_solver_instruction(q)
            for s in range(n_samples):
                items.append({
                    "data_source": "pred_general",
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "question": q, "answer": "", "ability": "general",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {
                        "split": "train",
                        "index": idx * n_samples + s,
                        "q_index": idx,
                        "s_index": s,
                        "metric": "pred_general",
                    },
                })
        return items

    def _make_judge_items(self, questions, answers, n_samples: int = 1):
        """
        Emit `n_samples` Judge prompt-copies per (question, answer) pair,
        laid out contiguously. `questions` and `answers` must be the same
        length (one per Solver attempt, already flattened over the Solver's
        sample dimension). Resulting order:
        [(q0,a0)·j0, (q0,a0)·j1, ..., (q0,a0)·j{n-1}, (q0,a1)·j0, ...]
        """
        judge_template = self.prompt_manager.get_judge_instruction(prompt_type="answer")
        items = []
        for idx, (q, a) in enumerate(zip(questions, answers)):
            if not q:
                continue
            try:
                prompt_text = judge_template.format(question=q, answer=a if a else "(no answer)")
            except (KeyError, ValueError):
                prompt_text = f"{judge_template}\n\nQuestion: {q}\n\nAnswer: {a}"
            for s in range(n_samples):
                items.append({
                    "data_source": "judge_general",
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "question": q, "answer": a, "ability": "general",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {
                        "split": "train",
                        "index": idx * n_samples + s,
                        "a_index": idx,
                        "j_index": s,
                        "metric": "judge_general",
                    },
                })
        return items

    # -------------------------------------------------------------------
    # Main fit() — 4-role co-training with cross-role rewards
    # -------------------------------------------------------------------

    def fit(self):
        logger = ReasonRLTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            tags=self.config.trainer.wandb_tags,
            resume="must" if self.config.trainer.resume_mode == "auto"
                and self.config.trainer.wandb_run_id is not None else False,
            run_id=self.config.trainer.wandb_run_id
                if self.config.trainer.wandb_run_id is not None else None,
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = (
                "{%- for message in messages -%}{{- '\\n' if not loop.first -}}"
                "{{- message['content'] -}}{%- endfor -%}"
            )

        # Initialize reward manager
        cose_cfg = self.config.cose.get("cose", {})
        reward_mgr = COSERewardManager(
            tokenizer=self.tokenizer,
            output_path=self.config.trainer.default_local_dir,
            confidence_signal=cose_cfg.get("confidence_signal", "self_certainty"),
            calibration_weight=float(cose_cfg.get("calibration_weight", 0.5)),
        )
        # Format-only warmup: for the first N steps, the Proposer reward is
        # `r = fmt` (no val_score, no difficulty). Lets a base model learn
        # the <question> schema before content quality starts mattering.
        # Default 0 = off. Set via `+cose.cose.quality_gate_warmup_steps=<int>`
        # (kept the legacy key name for script-compat — the underlying
        # mechanism no longer involves a quality gate).
        self._quality_gate_warmup_steps = int(cose_cfg.get("quality_gate_warmup_steps", 0))
        # Proposer+Validator cadence: only run question generation every N
        # training steps. Off-steps still sample from the buffer and train
        # Solver+Judge, so per-step compute is much lower (no Proposer/Validator
        # rollout, no buffer add). Default 1 = run every step (legacy).
        # Set via `+cose.cose.propose_every_n_steps=<int>`.
        self._propose_every_n_steps = max(1, int(cose_cfg.get("propose_every_n_steps", 1)))
        # Format-only side-training for Validator and Judge: even when
        # `train_validate`/`train_judge` are False, route a small subset of
        # their rollouts through PPO using only the <score> format reward.
        # Keeps the format schema alive on roles that otherwise get no
        # gradient. Default 0 = off.
        # Set via `+cose.cose.format_train_validator_n=<int>` and
        # `+cose.cose.format_train_judge_n=<int>`.
        self._format_train_validator_n = int(cose_cfg.get("format_train_validator_n", 0))
        self._format_train_judge_n = int(cose_cfg.get("format_train_judge_n", 0))
        # Reward manager needs the same flag to soften its reward formula
        # during warmup (else the Proposer still sees -0.3 on quality-fail).
        reward_mgr.quality_gate_warmup_steps = self._quality_gate_warmup_steps
        self._reward_mgr = reward_mgr

        # Initial validation
        if self.config.trainer.get("val_before_train", True) and (
            self.global_steps == 0 or self.config.trainer.get("val_only", False)
        ):
            benchmark_metrics = self._run_benchmark_evaluation()
            if benchmark_metrics:
                logger.log(data=benchmark_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.trainer.val_only:
            return

        # Seed dataset
        if self.loaded_datasets:
            PrettyPrinter.section_header("Resuming from checkpoint")
        else:
            self._init_seed_dataset()

        self.global_steps += 1
        self.pretrain_pred = (
            self.config.cose.pretrain_pred_steps > 0
            and self.global_steps <= self.config.cose.pretrain_pred_steps
        )

        data_len = self.config.data.train_batch_size

        while self.global_steps < self.total_training_steps:
            PrettyPrinter.section_header(f"COSE Step {self.global_steps}")
            PrettyPrinter.progress_bar(self.global_steps, self.total_training_steps, "Training")

            metrics = {}
            timing_raw = {}
            role_batches = {}

            with _timer("step", timing_raw):

                # Cleanup
                if self.global_steps - self._last_cleanup_step >= self._cleanup_frequency:
                    with _timer("cleanup", timing_raw):
                        self.cleanup()
                    self._last_cleanup_step = self.global_steps

                # ==========================================================
                # PHASE 1: Sequential rollout for all 4 roles
                # ==========================================================

                # Step-cadence gate: only run Proposer+Validator+buffer-add
                # every N steps. Step 1 (the first step after init) is always
                # a propose step so the buffer has content. Off-steps still
                # sample from the buffer and run Solver+Judge.
                do_propose = ((self.global_steps - 1) % self._propose_every_n_steps) == 0
                metrics["cose/propose_step"] = 1.0 if do_propose else 0.0
                metrics["cose/propose_every_n_steps"] = float(self._propose_every_n_steps)

                # Per-role ensemble size used across Validator/Solver/Judge rollouts.
                # Defined unconditionally — needed downstream regardless of do_propose.
                n_samples = int(self.config.cose.get("reward", {}).get("n_samples", 1) or 1)
                n_samples = max(1, n_samples)

                # Defaults for the no-propose path. Initialized here so all
                # downstream phases have well-defined empty arrays to work
                # against.
                proposer_batch = None
                c_proposer = np.zeros(0)
                questions: list[str] = []
                valid_mask: list[bool] = []
                valid_questions: list[str] = []
                N_valid = 0
                validator_batch = None
                c_validator = np.zeros(0)
                validator_texts: list[str] = []
                val_scores_full: list[float] = []
                val_valid_per_question = np.zeros(0, dtype=bool)
                validator_texts_per_attempt: list[str] = []
                c_validator_per_attempt = np.zeros(0, dtype=np.float32)

                if do_propose:
                    # --- Role 1: Proposer rollout ---
                    PrettyPrinter.section_header("COSE Phase 1: Proposer Rollout")
                    gen_dataloader = self._create_train_gen_dataloader(
                        problem_type="general", data_len=data_len, dataset_key="general",
                    )
                    try:
                        batch_dict = next(gen_dataloader)
                    except StopIteration:
                        gen_dataloader = self._create_train_gen_dataloader(
                            problem_type="general", data_len=data_len, dataset_key="general",
                        )
                        batch_dict = next(gen_dataloader)

                    proposer_batch = DataProto.from_single_dict(batch_dict)
                    proposer_batch, c_proposer = self._rollout_role(
                        proposer_batch, metrics, timing_raw, "gen_general",
                        training=self.config.cose.get("train_propose", True),
                    )

                    # Extract questions from proposer output
                    questions = self._extract_questions_from_batch(proposer_batch)
                    valid_mask = [bool(q) for q in questions]
                    valid_questions = [q for q in questions if q]
                    N_valid = len(valid_questions)
                    val_scores_full = [0.0] * len(questions)
                    validator_texts = [""] * len(questions)
                    c_validator = np.zeros(len(questions))
                    val_valid_per_question = np.zeros(N_valid, dtype=bool)

                metrics["cose/num_new_questions"] = float(len(valid_questions))

                # --- Role 2: Validator rollout (n_samples evaluations per question) ---
                # Only runs on propose-steps. On off-steps the per-question
                # arrays initialized above are already empty and Phase 1b/2/3
                # skip Proposer+Validator paths.
                if do_propose and valid_questions:
                    PrettyPrinter.section_header("COSE Phase 1: Validator Rollout")
                    val_items = self._make_validator_items(valid_questions, n_samples=n_samples)
                    val_proto = self._build_role_batch(val_items, "validate_general")
                    if val_proto is not None:
                        validator_batch, c_validator_valid = self._rollout_role(
                            val_proto, metrics, timing_raw, "validate_general",
                            training=self.config.cose.get("train_validate", False),
                        )
                        validator_texts_all = self._decode_batch_responses(validator_batch)
                        bs_v = len(validator_texts_all)
                        expected_v = N_valid * n_samples
                        if bs_v < expected_v:
                            pad = expected_v - bs_v
                            validator_texts_all = list(validator_texts_all) + [""] * pad
                            c_validator_valid = np.concatenate([c_validator_valid, np.zeros(pad)])

                        # Parse each Validator text → normalized score in [0,1].
                        # On parse failure, fall back to a fresh GPT call with
                        # the same Validator template so the score isn't lost
                        # to actor-side parse failures.
                        raw_val = np.zeros(expected_v, dtype=np.float32)
                        parse_ok_v = np.zeros(expected_v, dtype=bool)
                        for idx in range(expected_v):
                            s = _extract_scores(validator_texts_all[idx])
                            if s:
                                raw_val[idx] = (max(1.0, min(10.0, s[-1])) - 1.0) / 9.0
                                parse_ok_v[idx] = True

                        # GPT fallback: rerun the same Validator template
                        # through gpt-4.1-nano for any items the actor
                        # failed to parse. Recovers signal that would
                        # otherwise be 0 → 0 grad collapse.
                        n_gpt_recovered_v = 0
                        n_gpt_attempted_v = 0
                        if (self._gpt_fallback is not None
                                and self._gpt_fallback.enabled):
                            failed_v = [i for i in range(expected_v) if not parse_ok_v[i]]
                            n_gpt_attempted_v = len(failed_v)
                            if failed_v:
                                # Mirror _make_validator_items: get_template("validator")
                                # + format(question=...) with the same fallback for
                                # templates that don't use {question}.
                                validator_template = self.prompt_manager.get_template("validator")
                                fb_prompts = []
                                for i in failed_v:
                                    q_text = valid_questions[i // n_samples]
                                    try:
                                        fb_prompts.append(
                                            validator_template.format(question=q_text)
                                        )
                                    except (KeyError, ValueError):
                                        fb_prompts.append(
                                            f"{validator_template}\n\nQuestion: {q_text}"
                                        )
                                fb_scores = self._gpt_fallback.score_many(
                                    fb_prompts, max_workers=15,
                                )
                                for i, sc in zip(failed_v, fb_scores):
                                    if sc is not None:
                                        raw_val[i] = float(sc)
                                        parse_ok_v[i] = True
                                        n_gpt_recovered_v += 1
                        metrics["cose/validator_gpt_fallback_attempted"] = float(n_gpt_attempted_v)
                        metrics["cose/validator_gpt_fallback_recovered"] = float(n_gpt_recovered_v)
                        metrics["cose/validator_gpt_fallback_recover_rate"] = (
                            float(n_gpt_recovered_v / n_gpt_attempted_v) if n_gpt_attempted_v > 0 else 0.0
                        )
                        metrics["cose/validator_native_parse_rate"] = (
                            float((expected_v - n_gpt_attempted_v) / expected_v) if expected_v > 0 else 0.0
                        )

                        # Plain mean over the ensemble axis (no nanmean — failures = 0).
                        per_q_2d = raw_val.reshape(N_valid, n_samples)
                        per_q_valid_2d = parse_ok_v.reshape(N_valid, n_samples)
                        per_q_mean = per_q_2d.mean(axis=1)
                        val_valid_per_question = per_q_valid_2d.any(axis=1)  # diagnostic only
                        per_q_filled = per_q_mean

                        # Confidence: per-attempt + per-question mean (axis=1).
                        c_validator_per_attempt = np.asarray(
                            c_validator_valid[:expected_v], dtype=np.float32,
                        )
                        c_validator_2d = c_validator_per_attempt.reshape(N_valid, n_samples)
                        c_validator_per_question = c_validator_2d.mean(axis=1)

                        # Save per-attempt texts (used only when train_validate=True).
                        validator_texts_per_attempt = list(validator_texts_all[:expected_v])

                        # Map back to full-length-N arrays for downstream consumers.
                        vi = 0
                        for i, is_valid in enumerate(valid_mask):
                            if is_valid and vi < N_valid:
                                c_validator[i] = float(c_validator_per_question[vi])
                                validator_texts[i] = validator_texts_all[vi * n_samples]  # first-sample trace
                                val_scores_full[i] = float(per_q_filled[vi])
                                vi += 1

                        # Release validator GPU memory if not training it AND
                        # format-only side training is also off. Otherwise
                        # we need the batch alive for Phase 3.
                        if (not self.config.cose.get("train_validate", False)
                                and self._format_train_validator_n <= 0):
                            del validator_batch
                            validator_batch = None
                            torch.cuda.empty_cache()
                            gc.collect()

                # ==========================================================
                # PHASE 1b: Quality gate + buffer add + Solver replay sampling
                # ==========================================================
                # Apply MAE-style hard quality gate (val_score >= QUALITY_GATE)
                # to this-step questions, append the survivors to the cose_dataset
                # buffer, then priority-sample bs_sol_target questions for the
                # Solver pass. Sampled questions can come from THIS step or any
                # prior step — that's the replay buffer benefit.
                PrettyPrinter.section_header("COSE Phase 1: Quality Gate & Buffer Sampling")

                # No quality gate: any well-formed question enters the buffer.
                # val_score still influences reward (multiplicative term) and
                # buffer-sample priority (importance = val × c_validator), so
                # low-quality questions are downweighted, not discarded.
                in_warmup = (
                    self._quality_gate_warmup_steps > 0
                    and self.global_steps <= self._quality_gate_warmup_steps
                )
                keep_mask = [bool(is_valid) for is_valid in valid_mask]
                metrics["cose/buffer_admission_kept"] = float(sum(keep_mask))
                metrics["cose/buffer_admission_dropped"] = float(
                    sum(1 for v in valid_mask if not v)
                )
                metrics["cose/format_warmup_active"] = 1.0 if in_warmup else 0.0

                # ---- val_score distribution diagnostics. Even without a hard
                # gate, percentiles tell us whether Validator scores are
                # collapsing or spreading — that drives Proposer/Solver
                # importance weights downstream.
                val_arr = np.asarray(val_scores_full, dtype=np.float32)
                if val_arr.size:
                    metrics["cose/val_score_p10"] = float(np.percentile(val_arr, 10))
                    metrics["cose/val_score_p50"] = float(np.percentile(val_arr, 50))
                    metrics["cose/val_score_p90"] = float(np.percentile(val_arr, 90))
                    metrics["cose/val_score_max"] = float(val_arr.max())
                    metrics["cose/val_score_mean"] = float(val_arr.mean())
                # (judge_score_p distribution metrics are emitted later, after
                # the Judge phase has populated judge_scores.)

                new_entries = []
                i_to_offset = {}  # original questions[i] → offset in new_entries
                for i, keep in enumerate(keep_mask):
                    if keep:
                        i_to_offset[i] = len(new_entries)
                        new_entries.append({
                            "question": questions[i],
                            "generation": questions[i],
                            "thought": questions[i],
                            "c_proposer": float(c_proposer[i]),
                            "c_validator": float(c_validator[i]),
                            "val_score": float(val_scores_full[i]),
                        })

                buffer_size_before = ray.get(self.cose_dataset.get_size.remote())
                if new_entries:
                    ray.get(self.cose_dataset.add_questions.remote(new_entries, self.global_steps))
                    parent_data = [
                        {"question": questions[i], "generation": questions[i], "thought": questions[i]}
                        for i, keep in enumerate(keep_mask) if keep
                    ]
                    ray.get(self.dataset_manager.add_general_batch.remote(parent_data, self.global_steps))
                buffer_size_after = buffer_size_before + len(new_entries)
                metrics["cose/buffer_size"] = float(buffer_size_after)
                metrics["cose/buffer_growth_per_step"] = float(len(new_entries))
                # Empty-buffer flag — a 1.0 here means the Solver phase is
                # about to be skipped/empty; it's the canonical "stuck" signal.
                metrics["cose/buffer_empty"] = 1.0 if buffer_size_after == 0 else 0.0

                # Map: buffer index → original questions[i] index (this step only).
                buffer_idx_to_i = {
                    buffer_size_before + offset: i for i, offset in i_to_offset.items()
                }
                # Map: original questions[i] index → v in [0, N_valid).
                i_to_v = {}
                v_idx = 0
                for i, is_valid in enumerate(valid_mask):
                    if is_valid:
                        i_to_v[i] = v_idx
                        v_idx += 1

                # Priority-sample bs_sol_target questions from the buffer.
                # On propose-steps target N_valid so per-step Solver+Judge
                # compute matches Option B (no extra rollout budget). On
                # off-steps (no fresh Proposer batch), fall back to the
                # configured train_batch_size so the Solver still trains.
                if do_propose:
                    bs_sol_target = N_valid
                else:
                    bs_sol_target = int(self.config.data.train_batch_size)
                if buffer_size_after > 0 and bs_sol_target > 0:
                    sampled_items, sampled_indices = ray.get(
                        self.cose_dataset.select_for_solver.remote(
                            bs_sol_target,
                            strategy=self.selection_strategy,
                            global_step=self.global_steps,
                        )
                    )
                    sampled_questions = [item["question"] for item in sampled_items]
                else:
                    sampled_items, sampled_indices, sampled_questions = [], [], []

                N_sol = len(sampled_questions)
                # Per-sampled-question val_score and c_validator (frozen at
                # the question's insertion-step). Used for Solver importance
                # = val_score × c_validator × c_judge below.
                sampled_val_scores = np.array(
                    [float(item.get("val_score", 1.0)) for item in sampled_items],
                    dtype=np.float64,
                )
                sampled_c_validator = np.array(
                    [float(item.get("c_validator", 0.0)) for item in sampled_items],
                    dtype=np.float64,
                )
                # Per-sampled-position → original questions[i] index, or -1 if
                # the sampled question was replayed from a prior step.
                sampled_to_i = [buffer_idx_to_i.get(idx, -1) for idx in sampled_indices]
                n_this_step_sampled = sum(1 for i in sampled_to_i if i >= 0)
                metrics["cose/solver_replay_size"] = float(N_sol)
                metrics["cose/solver_this_step_sampled"] = float(n_this_step_sampled)
                metrics["cose/solver_replayed_from_past"] = float(N_sol - n_this_step_sampled)
                metrics["cose/replay_fresh_frac"] = (
                    float(n_this_step_sampled / N_sol) if N_sol > 0 else 0.0
                )
                # Age of replayed items — high age means we keep grinding old
                # questions because nothing new is passing the gate.
                origin_steps = [
                    int(item.get("created_step", self.global_steps))
                    for item in sampled_items
                ]
                if origin_steps:
                    ages = np.asarray(
                        [self.global_steps - s for s in origin_steps], dtype=np.float32
                    )
                    metrics["cose/replay_age_mean"] = float(ages.mean())
                    metrics["cose/replay_age_max"] = float(ages.max())

                # --- Role 3: Solver rollout on buffer-sampled questions ---
                # The same rollouts feed (a) the Solver gradient and (b) the
                # Proposer reward's difficulty term — but only for sampled
                # questions that came from THIS step (mapping via sampled_to_i).
                # This-step questions that weren't sampled get judge_valid=False
                # in the Proposer reward, which falls back to (val_score+format)/3.
                PrettyPrinter.section_header("COSE Phase 1: Solver Rollout (replay)")
                solver_batch = None
                # Per-this-step-question arrays (0 / "" for unsampled).
                c_solver = np.zeros(len(questions))
                solver_texts = [""] * len(questions)
                answers = [""] * len(questions)
                # Per-sampled-attempt arrays (length N_sol * n_samples).
                answers_flat: list[str] = []
                c_solver_per_attempt = np.zeros(0, dtype=np.float32)
                # Per-sampled-question aggregates (length N_sol).
                c_solver_per_sampled = np.zeros(N_sol, dtype=np.float32)
                answers_per_sampled = [""] * N_sol

                if sampled_questions:
                    sol_items = self._make_solver_items(sampled_questions, n_samples=n_samples)
                    sol_proto = self._build_role_batch(sol_items, "pred_general")
                    if sol_proto is not None:
                        solver_batch, c_solver_valid = self._rollout_role(
                            sol_proto, metrics, timing_raw, "pred_general",
                            training=self.config.cose.get("train_solve", True),
                        )
                        solver_texts_valid = self._decode_batch_responses(solver_batch)
                        answers_valid = self._extract_answers_from_batch(solver_batch)
                        # solver_batch is laid out [s0·a0,...,s0·a{n-1}, s1·a0,...]
                        bs_sol = len(answers_valid)
                        expected = N_sol * n_samples
                        if bs_sol < expected:
                            # _build_role_batch may truncate to a multiple of n_gpus;
                            # pad with empty strings to keep the (N_sol, n_samples) reshape clean.
                            pad = expected - bs_sol
                            answers_valid = list(answers_valid) + [""] * pad
                            solver_texts_valid = list(solver_texts_valid) + [""] * pad
                            c_solver_valid = np.concatenate([c_solver_valid, np.zeros(pad)])

                        answers_flat = list(answers_valid[:expected])
                        c_solver_per_attempt = np.asarray(c_solver_valid[:expected], dtype=np.float32)

                        c_solver_per_sampled = c_solver_per_attempt.reshape(N_sol, n_samples).mean(axis=1)
                        answers_per_sampled = [answers_flat[s * n_samples] for s in range(N_sol)]
                        solver_texts_per_sampled = [solver_texts_valid[s * n_samples] for s in range(N_sol)]

                        # Map back to ORIGINAL question positions for this-step
                        # questions that were sampled (downstream traces and
                        # Proposer-reward-side per-question arrays).
                        for s, i in enumerate(sampled_to_i):
                            if i >= 0:
                                c_solver[i] = float(c_solver_per_sampled[s])
                                solver_texts[i] = solver_texts_per_sampled[s]
                                answers[i] = answers_per_sampled[s]

                # --- Role 4: Judge rollout (n_samples evaluations per sampled answer) ---
                PrettyPrinter.section_header("COSE Phase 1: Judge Rollout")
                judge_batch = None
                # Per-this-step-question arrays (0 / "" / False for unsampled).
                c_judge = np.zeros(len(questions))
                judge_texts = [""] * len(questions)
                judge_score_per_question = np.zeros(N_valid, dtype=np.float32)
                judge_valid_per_question = np.zeros(N_valid, dtype=bool)
                # Per-sampled-attempt arrays (length N_sol * n_samples).
                judge_score_per_answer = np.zeros(N_sol * n_samples, dtype=np.float32)
                judge_valid_per_answer = np.zeros(N_sol * n_samples, dtype=bool)
                c_judge_per_attempt = np.zeros(N_sol * n_samples, dtype=np.float32)
                # Per-sampled-question aggregates (length N_sol).
                c_judge_per_sampled = np.zeros(N_sol, dtype=np.float32)
                judge_score_per_sampled = np.zeros(N_sol, dtype=np.float32)

                if sampled_questions and answers_flat:
                    # MAE-style: ensemble the Judge n_samples times PER (Q, A) pair.
                    # Layout = [(q0,a0)·j0, (q0,a0)·j1, ..., (q0,a0)·j{n-1},
                    #           (q0,a1)·j0, ..., (qN,aS-1)·j{n-1}] (q indexes sampled).
                    questions_flat = [q for q in sampled_questions for _ in range(n_samples)]
                    judge_items = self._make_judge_items(
                        questions_flat, answers_flat, n_samples=n_samples,
                    )
                    judge_proto = self._build_role_batch(judge_items, "judge_general")
                    if judge_proto is not None:
                        judge_batch, c_judge_valid = self._rollout_role(
                            judge_proto, metrics, timing_raw, "judge_general",
                            training=self.config.cose.get("train_judge", False),
                        )
                        judge_texts_valid = self._decode_batch_responses(judge_batch)
                        bs_j = len(judge_texts_valid)
                        expected_j = N_sol * n_samples * n_samples
                        if bs_j < expected_j:
                            pad = expected_j - bs_j
                            judge_texts_valid = list(judge_texts_valid) + [""] * pad
                            c_judge_valid = np.concatenate([c_judge_valid, np.zeros(pad)])

                        # Parse each judge text → normalized score in [0,1].
                        raw_judge = np.zeros(expected_j, dtype=np.float32)
                        parse_ok_j = np.zeros(expected_j, dtype=bool)
                        for idx in range(expected_j):
                            s = _extract_scores(judge_texts_valid[idx])
                            if s:
                                raw_judge[idx] = (max(1.0, min(10.0, s[-1])) - 1.0) / 9.0
                                parse_ok_j[idx] = True

                        # GPT fallback: rerun the same Judge template through
                        # gpt-4.1-nano for any items the actor failed to
                        # parse. Layout: idx → (q_pair, judge_sample) where
                        # q_pair = idx // n_samples; the Q+A pair is
                        # (questions_flat[q_pair], answers_flat[q_pair]).
                        n_gpt_recovered_j = 0
                        n_gpt_attempted_j = 0
                        if (self._gpt_fallback is not None
                                and self._gpt_fallback.enabled):
                            failed_j = [i for i in range(expected_j) if not parse_ok_j[i]]
                            n_gpt_attempted_j = len(failed_j)
                            if failed_j:
                                judge_template = self.prompt_manager.get_judge_instruction(
                                    prompt_type="answer",
                                )
                                fb_prompts = []
                                for i in failed_j:
                                    q_pair = i // n_samples
                                    q_text = (questions_flat[q_pair]
                                              if q_pair < len(questions_flat) else "")
                                    a_text = (answers_flat[q_pair]
                                              if q_pair < len(answers_flat) else "")
                                    fb_prompts.append(
                                        judge_template.format(question=q_text, answer=a_text)
                                    )
                                fb_scores = self._gpt_fallback.score_many(
                                    fb_prompts, max_workers=15,
                                )
                                for i, sc in zip(failed_j, fb_scores):
                                    if sc is not None:
                                        raw_judge[i] = float(sc)
                                        parse_ok_j[i] = True
                                        n_gpt_recovered_j += 1
                        metrics["cose/judge_gpt_fallback_attempted"] = float(n_gpt_attempted_j)
                        metrics["cose/judge_gpt_fallback_recovered"] = float(n_gpt_recovered_j)
                        metrics["cose/judge_gpt_fallback_recover_rate"] = (
                            float(n_gpt_recovered_j / n_gpt_attempted_j) if n_gpt_attempted_j > 0 else 0.0
                        )
                        metrics["cose/judge_native_parse_rate"] = (
                            float((expected_j - n_gpt_attempted_j) / expected_j) if expected_j > 0 else 0.0
                        )

                        # Reshape to (N_sol, n_samples_solver, n_samples_judge).
                        raw_judge_3d = raw_judge.reshape(N_sol, n_samples, n_samples)
                        parse_ok_3d = parse_ok_j.reshape(N_sol, n_samples, n_samples)

                        # Plain mean over the Judge ensemble axis → per-(Q, A) scores.
                        per_answer = raw_judge_3d.mean(axis=2)                  # (N_sol, n_samples)
                        per_answer_valid = parse_ok_3d.any(axis=2)

                        judge_score_per_answer = per_answer.reshape(N_sol * n_samples)
                        judge_valid_per_answer = per_answer_valid.reshape(N_sol * n_samples)

                        # Per-sampled-question aggregates.
                        judge_score_per_sampled = per_answer.mean(axis=1).astype(np.float32)
                        judge_valid_per_sampled = per_answer_valid.any(axis=1)

                        c_judge_3d = np.asarray(c_judge_valid[:expected_j], dtype=np.float32).reshape(
                            N_sol, n_samples, n_samples,
                        )
                        c_judge_per_attempt = c_judge_3d.mean(axis=2).reshape(N_sol * n_samples)
                        c_judge_per_sampled = c_judge_3d.mean(axis=(1, 2))
                        judge_texts_per_sampled = [
                            judge_texts_valid[s * n_samples * n_samples] for s in range(N_sol)
                        ]

                        # Map sampled-position values → original questions[i] /
                        # valid-only v positions for downstream Proposer reward.
                        # Unsampled this-step questions remain 0/False, which makes
                        # the Proposer reward fall back to (val_score+format)/3.
                        for s, i in enumerate(sampled_to_i):
                            if i >= 0 and i in i_to_v:
                                v = i_to_v[i]
                                judge_score_per_question[v] = float(judge_score_per_sampled[s])
                                judge_valid_per_question[v] = bool(judge_valid_per_sampled[s])
                                c_judge[i] = float(c_judge_per_sampled[s])
                                judge_texts[i] = judge_texts_per_sampled[s]

                        # Release judge GPU memory if not training it.
                        if not self.config.cose.get("train_judge", False):
                            del judge_batch
                            judge_batch = None
                            torch.cuda.empty_cache()
                            gc.collect()

                # ==========================================================
                # PHASE 2: Cross-role reward assignment
                # ==========================================================
                PrettyPrinter.section_header("COSE Phase 2: Cross-Role Rewards")

                # Per-question `p_success` (length == len(questions)).
                # Failed-parse cells contributed 0.0 to the ensemble mean upstream,
                # so partial parse failures already pull p downward. judge_valid_q
                # is kept as a diagnostic flag (logged via cose/judge_valid_frac_*).
                judge_scores = [0.0] * len(questions)
                judge_valid_q = [False] * len(questions)
                vi = 0
                for i, is_valid in enumerate(valid_mask):
                    if is_valid and vi < N_valid:
                        judge_scores[i] = float(judge_score_per_question[vi])
                        judge_valid_q[i] = bool(judge_valid_per_question[vi])
                        vi += 1

                metrics["cose/judge_valid_frac_per_question"] = (
                    float(np.mean(judge_valid_per_question)) if judge_valid_per_question.size else 0.0
                )
                metrics["cose/validator_valid_frac_per_question"] = (
                    float(np.mean(val_valid_per_question)) if val_valid_per_question.size else 0.0
                )

                # judge_score_p distribution → curriculum health. Bell-curve
                # learnability peaks at p=0.5; mean(4p(1-p)) shows when
                # questions are too-easy (p≈1) or too-hard (p≈0).
                p_arr_metric = np.asarray(judge_scores, dtype=np.float32)
                if p_arr_metric.size:
                    metrics["cose/judge_p_p10"] = float(np.percentile(p_arr_metric, 10))
                    metrics["cose/judge_p_p50"] = float(np.percentile(p_arr_metric, 50))
                    metrics["cose/judge_p_p90"] = float(np.percentile(p_arr_metric, 90))
                    metrics["cose/learnability_mean"] = float(
                        np.mean(4.0 * p_arr_metric * (1.0 - p_arr_metric))
                    )

                # Proposer reward ← (val_score + (1 − p) + format) / 3, MAE-style.
                # Hard quality gate (val_score ≥ 0.7) keeps the difficulty term;
                # below the gate (or when judge_valid_q=False) the reward falls
                # back to (val_score + format) / 3. val_scores_full is the
                # n_samples-ensembled per-question Validator score.
                # Curriculum warmup: during the first N steps the reward
                # manager skips the val_score+difficulty content term so the
                # Proposer learns the <question> schema from format reward alone.
                reward_mgr.in_quality_gate_warmup = (
                    self._quality_gate_warmup_steps > 0
                    and self.global_steps <= self._quality_gate_warmup_steps
                )
                proposer_reward = None
                val_scores_all = list(val_scores_full)
                if do_propose and proposer_batch is not None:
                    proposer_reward, proposer_metrics, val_scores_all = reward_mgr.compute_proposer_reward(
                        proposer_batch, validator_texts, judge_scores,
                        judge_valid=judge_valid_q, validator_scores=val_scores_full,
                    )
                    metrics.update({f"gen_general/{k}": np.mean(v) for k, v in proposer_metrics.items()})
                metrics["cose/format_warmup_active"] = float(
                    1.0 if reward_mgr.in_quality_gate_warmup else 0.0
                )

                # Validator reward ← Calibration vs judge outcome.
                # validator_batch length == N_valid * n_samples (one row per ensemble
                # call). Broadcast per-question judge_scores via np.repeat so each
                # validator row sees its question's judge score.
                if validator_batch is not None:
                    judge_scores_v = [judge_scores[i] for i, m in enumerate(valid_mask) if m]
                    judge_scores_for_validator = list(np.repeat(judge_scores_v, n_samples))
                    validator_reward, validator_metrics = reward_mgr.compute_validator_reward(
                        validator_batch, validator_texts_per_attempt, judge_scores_for_validator,
                    )
                    metrics.update({f"validate_general/{k}": np.mean(v) for k, v in validator_metrics.items()})

                # Judge reward ← Agreement with solver confidence (Phase 1 internal).
                # judge_batch is from the reward-internal stream; if Judge is being
                # trained, this is its gradient source.
                if judge_batch is not None:
                    judge_texts_valid_for_judge = self._decode_batch_responses(judge_batch)
                    c_sol_for_judge = np.repeat(c_solver_per_attempt, n_samples)
                    judge_reward, judge_metrics = reward_mgr.compute_judge_reward(
                        judge_batch, judge_texts_valid_for_judge, c_sol_for_judge,
                    )
                    metrics.update({f"judge_general/{k}": np.mean(v) for k, v in judge_metrics.items()})

                # ==========================================================
                # Solver reward (single-stream Option B).
                # ==========================================================
                # The Phase 1 Solver+Judge rollouts run on this-step's Proposer
                # questions. Same rollouts feed both the Proposer reward (via
                # `judge_score_per_question`) and the Solver gradient (via
                # `judge_score_per_answer`). No separate buffer-sampled training
                # pass — every Solver/Judge invocation produces gradient.
                solver_reward = None
                if (solver_batch is not None
                        and self.config.cose.get("train_solve", True)
                        and N_sol > 0):
                    solver_reward, solver_metrics = reward_mgr.compute_solver_reward(
                        solver_batch,
                        judge_scores=judge_score_per_answer.tolist(),
                        judge_valid=judge_valid_per_answer.tolist(),
                        c_solver=c_solver_per_attempt.tolist(),
                    )
                    metrics.update({
                        f"pred_general/{k}": np.mean(v) for k, v in solver_metrics.items()
                    })


                # ==========================================================
                # PHASE 3: Per-role importance + advantages
                # ==========================================================
                PrettyPrinter.section_header("COSE Phase 3: Per-Role Importance & Advantages")

                # Gather valid-only confidence arrays
                c_val_v = np.array([c_validator[i] for i, m in enumerate(valid_mask) if m])
                c_sol_v = np.array([c_solver[i] for i, m in enumerate(valid_mask) if m])
                c_jud_v = np.array([c_judge[i] for i, m in enumerate(valid_mask) if m])
                val_scores_v = np.array([val_scores_all[i] for i, m in enumerate(valid_mask) if m])

                # --- Proposer importance = clip(val_score × c_validator, 0.1, 1.0) ---
                # Map valid-only importance back to full proposer batch
                proposer_imp_valid = (
                    self._compute_proposer_importance(val_scores_v, c_val_v)
                    if len(c_val_v) > 0 else np.ones(0)
                )
                proposer_importance = np.ones(len(questions))  # default for invalid questions
                vi = 0
                for i, is_valid in enumerate(valid_mask):
                    if is_valid and vi < len(proposer_imp_valid):
                        proposer_importance[i] = proposer_imp_valid[vi]
                        vi += 1
                metrics["cose/proposer_importance_mean"] = float(np.mean(proposer_importance))

                if do_propose and proposer_batch is not None:
                    proposer_batch = self._finalize_role(
                        proposer_batch, proposer_reward, metrics, timing_raw,
                        "gen_general", sample_importance=proposer_importance,
                    )
                    if self.config.cose.get("train_propose", True):
                        role_batches["gen_general"] = proposer_batch

                # --- Validator importance = f(c_judge) ---
                # validator_batch has length N_valid * n_samples after ensembling, so
                # the per-question importance is broadcast n_samples times to align.
                # Only add to role_batches (i.e., train) if train_validate=True.
                if validator_batch is not None and self.config.cose.get("train_validate", False):
                    if len(c_jud_v) > 0:
                        imp_per_question = self._compute_validator_importance(c_jud_v)
                        validator_importance = np.repeat(imp_per_question, n_samples)
                    else:
                        validator_importance = np.ones(N_valid * n_samples)
                    metrics["cose/validator_importance_mean"] = float(np.mean(validator_importance))
                    validator_batch = self._finalize_role(
                        validator_batch, validator_reward, metrics, timing_raw,
                        "validate_general", sample_importance=validator_importance,
                    )
                    role_batches["validate_general"] = validator_batch
                elif (validator_batch is not None
                        and self._format_train_validator_n > 0
                        and "validate_general" not in role_batches):
                    # Format-only side training: take the first
                    # `format_train_validator_n` rollouts from the Validator
                    # batch and route them to PPO with format-only reward
                    # (only the <score> schema matters — no calibration).
                    n_v = min(self._format_train_validator_n, len(validator_batch))
                    if n_v > 0:
                        v_sub = validator_batch.slice(0, n_v)
                        v_reward, v_fmt_metrics = reward_mgr.compute_validator_format_only_reward(v_sub)
                        metrics.update({
                            f"validate_general/{k}": np.mean(v) for k, v in v_fmt_metrics.items()
                        })
                        v_imp = np.ones(n_v)
                        metrics["cose/validator_format_train_n"] = float(n_v)
                        v_sub = self._finalize_role(
                            v_sub, v_reward, metrics, timing_raw,
                            "validate_general", sample_importance=v_imp,
                        )
                        role_batches["validate_general"] = v_sub

                # --- Solver importance: PER-ATTEMPT, full three-factor product ---
                # importance(q, s) = clip(val_score(q) × c_validator(q) × c_judge(q, s), 0.1, 1.0)
                # val_score × c_validator is the question-level "trustworthy
                # quality" signal (frozen at insertion). c_judge is the
                # per-attempt Judge confidence on the Solver's specific
                # answer. The product weights the Solver gradient: learn most
                # from well-certified questions where the Judge graded
                # confidently.
                if (solver_batch is not None
                        and self.config.cose.get("train_solve", True)
                        and solver_reward is not None):
                    bs_sol = N_sol * n_samples
                    if bs_sol > 0 and c_judge_per_attempt.size >= bs_sol:
                        # Broadcast per-sampled-question val_score and c_validator
                        # to per-attempt by repeating n_samples times.
                        val_per_attempt = np.repeat(sampled_val_scores, n_samples)
                        cval_per_attempt = np.repeat(sampled_c_validator, n_samples)
                        c_judge_pa = np.asarray(
                            c_judge_per_attempt[:bs_sol], dtype=np.float64,
                        )
                        solver_importance = self._compute_solver_importance(
                            val_per_attempt, cval_per_attempt, c_judge_pa,
                        )
                    else:
                        solver_importance = np.ones(bs_sol)
                    metrics["cose/solver_importance_mean"] = float(np.mean(solver_importance))
                    metrics["cose/solver_importance_std"] = float(np.std(solver_importance))
                    metrics["cose/solver_importance_p10"] = float(np.percentile(solver_importance, 10))
                    metrics["cose/solver_importance_p90"] = float(np.percentile(solver_importance, 90))
                    # Floor-saturation rate: fraction of samples pinned at the
                    # 0.1 clip floor. High value = importance signal collapsed.
                    metrics["cose/solver_importance_at_floor"] = float(
                        (np.asarray(solver_importance) <= 0.1 + 1e-6).mean()
                    )
                    solver_batch = self._finalize_role(
                        solver_batch, solver_reward, metrics, timing_raw,
                        "pred_general", sample_importance=solver_importance,
                    )
                    role_batches["pred_general"] = solver_batch

                # --- Judge importance = f(mean_s c_solver) ---
                # One importance value per question (per-question mean of Solver
                # confidences over the n_samples attempts), broadcast to the
                # N_valid * n_samples * n_samples Judge rollouts (one weight per
                # judge call, all calls on the same question share the same weight).
                if judge_batch is not None and self.config.cose.get("train_judge", False):
                    # Judge runs on sampled questions, so the gradient batch size
                    # is N_sol * n_samples * n_samples (not N_valid).
                    if len(c_sol_v) > 0:
                        imp_per_question = self._compute_judge_importance(c_sol_v)
                        judge_importance = np.repeat(imp_per_question, n_samples * n_samples)
                    else:
                        judge_importance = np.ones(N_sol * n_samples * n_samples)
                    metrics["cose/judge_importance_mean"] = float(np.mean(judge_importance))
                    judge_batch = self._finalize_role(
                        judge_batch, judge_reward, metrics, timing_raw,
                        "judge_general", sample_importance=judge_importance,
                    )
                    role_batches["judge_general"] = judge_batch
                elif (judge_batch is not None
                        and self._format_train_judge_n > 0
                        and "judge_general" not in role_batches):
                    # Format-only side training for the Judge: same idea as
                    # the Validator block above. First N rollouts get format
                    # reward only (1.0/0.5/-1.0 on the <score> contract).
                    n_j = min(self._format_train_judge_n, len(judge_batch))
                    if n_j > 0:
                        j_sub = judge_batch.slice(0, n_j)
                        j_reward, j_fmt_metrics = reward_mgr.compute_judge_format_only_reward(j_sub)
                        metrics.update({
                            f"judge_general/{k}": np.mean(v) for k, v in j_fmt_metrics.items()
                        })
                        j_imp = np.ones(n_j)
                        metrics["cose/judge_format_train_n"] = float(n_j)
                        j_sub = self._finalize_role(
                            j_sub, j_reward, metrics, timing_raw,
                            "judge_general", sample_importance=j_imp,
                        )
                        role_batches["judge_general"] = j_sub

                # ==========================================================
                # Post-Phase-3 bookkeeping: confidence updates on the SAMPLED
                # buffer indices (so the per-question c_solver/c_judge stored
                # in the buffer reflects the latest policy's behavior on
                # replayed questions), pair_data, and text dumps.
                # Buffer add + parent_data add already happened in Phase 1b.
                # ==========================================================
                if N_sol > 0 or valid_questions:
                    # Refresh c_solver / c_judge / last_p on the sampled
                    # buffer entries. last_p (Judge's <score> on the most
                    # recent answer for this question) drives the PER-style
                    # calibration-error priority in select_for_solver.
                    # Runs on every step regardless of do_propose so the
                    # buffer's per-question stats stay current with the latest
                    # policy.
                    if N_sol > 0 and sampled_indices:
                        ray.get(self.cose_dataset.update_solver_confidence.remote(
                            list(sampled_indices),
                            [float(c) for c in c_solver_per_sampled.tolist()],
                            list(answers_per_sampled),
                            self.global_steps,
                        ))
                        if c_judge_per_sampled.size:
                            ray.get(self.cose_dataset.update_judge_confidence.remote(
                                list(sampled_indices),
                                [float(c) for c in c_judge_per_sampled.tolist()],
                            ))
                        if judge_score_per_sampled.size >= N_sol:
                            ray.get(self.cose_dataset.update_last_p.remote(
                                list(sampled_indices),
                                [float(p) for p in judge_score_per_sampled[:N_sol].tolist()],
                            ))

                    # Store pairs for the parent dataset_manager (only this-step
                    # questions that were quality-gated AND sampled have answers).
                    pair_data = [
                        {"question": questions[i], "answer": answers[i],
                         "generation": answers[i], "thought": answers[i]}
                        for i, keep in enumerate(keep_mask)
                        if keep and questions[i] and answers[i]
                    ]
                    if pair_data:
                        ray.get(self.dataset_manager.add_general_pair_batch.remote(
                            pair_data, self.global_steps,
                        ))

                    # MAE-style text dumps: question_<RUN>.txt + pair_<RUN>.txt for
                    # gated-pass questions; low_quality_question.txt for gated-fail.
                    try:
                        agent_dir = self.config.get("agent_output_dir") or os.path.join(
                            self.config.trainer.default_local_dir, "agent_output",
                        )
                        os.makedirs(agent_dir, exist_ok=True)
                        run_name = self.config.trainer.experiment_name

                        if new_entries:
                            question_path = os.path.join(agent_dir, f"question_{run_name}.txt")
                            with open(question_path, "a") as f:
                                f.write(f"\n=== Global Training Step {self.global_steps} ===\n")
                                for entry in new_entries:
                                    f.write(f"Question: {entry['question']}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Thought: {entry['thought']}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Generation: {entry['generation']}\n")
                                    f.write("==============================================\n")
                                    f.write("\n")
                                f.write("\n")

                        if pair_data:
                            pair_path = os.path.join(agent_dir, f"pair_{run_name}.txt")
                            with open(pair_path, "a") as f:
                                f.write(f"\n=== Global Training Step {self.global_steps} ===\n")
                                for item in pair_data:
                                    f.write(f"Question: {item['question']}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Answer: {item['answer']}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Thought: {item['thought']}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Generation: {item['generation']}\n")
                                    f.write("==============================================\n")
                                    f.write("\n")
                                f.write("\n")

                        # Dump below-gate questions for inspection (matches MAE).
                        dropped_indices = [
                            i for i, is_valid in enumerate(valid_mask)
                            if is_valid and not keep_mask[i]
                        ]
                        if dropped_indices:
                            low_q_path = os.path.join(agent_dir, "low_quality_question.txt")
                            with open(low_q_path, "a") as f:
                                f.write(f"\n=== Global Training Step {self.global_steps} ===\n")
                                for i in dropped_indices:
                                    f.write(f"Question: {questions[i]}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Generation: {questions[i]}\n")
                                    f.write("==============================================\n")
                                    f.write(f"Validator score: {val_scores_full[i]:.3f}\n")
                                    f.write("==============================================\n")
                                    f.write("\n")
                                f.write("\n")
                    except Exception as e:
                        PrettyPrinter.status("DUMP", f"Failed to dump Q/A text: {e}", "warn")

                # ==========================================================
                # PHASE 4: Actor update on all role batches
                # ==========================================================
                if not role_batches:
                    PrettyPrinter.status("ERROR", "No batches — skipping step", "error")
                    self.global_steps += 1
                    continue

                batch = DataProto.concat(list(role_batches.values()))

                # Free per-role references now that we've concat'd — the concatenated
                # batch is what goes into update_actor. Keeping the individual role
                # batch references alive doubles memory pressure during the update.
                role_batches.clear()
                proposer_batch = None
                validator_batch = None
                solver_batch = None
                judge_batch = None
                torch.cuda.empty_cache()
                gc.collect()

                PrettyPrinter.section_header("COSE Phase 4: Parameter Updates")
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                # Validation
                if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with _timer("testing", timing_raw):
                        val_metrics = self._run_benchmark_evaluation()
                        if val_metrics:
                            PrettyPrinter.table(
                                ["Metric", "Value"],
                                [[k, v] for k, v in val_metrics.items()],
                                title="Benchmark Results",
                            )
                    metrics.update(val_metrics)

                # COSE dataset stats
                ds_stats = ray.get(self.cose_dataset.get_stats.remote())
                for k, v in ds_stats.items():
                    metrics[f"cose/dataset/{k}"] = float(v) if isinstance(v, (int, float)) else v

                # Checkpoint
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Timing + logging (filter to essential metrics only)
            metrics.update({f"timing/{k}": v for k, v in timing_raw.items()})
            logger.log(data=_filter_essential_metrics(metrics), step=self.global_steps)

            PrettyPrinter.status(
                "COSE",
                f"Step {self.global_steps} done. Roles trained: {list(role_batches.keys())}",
                "success",
            )

            if self.global_steps >= self.config.cose.pretrain_pred_steps:
                self.pretrain_pred = False
            self.global_steps += 1
            gc.collect()

            if self.global_steps >= self.total_training_steps:
                if self.config.trainer.save_freq > 0:
                    self._save_checkpoint()
                break

        PrettyPrinter.section_header("COSE Training Complete")
