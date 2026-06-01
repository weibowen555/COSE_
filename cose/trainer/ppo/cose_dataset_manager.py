"""
COSE Dataset Manager: confidence-prioritized replay buffer.

Wraps the parent DatasetManager (Ray actor) with per-question confidence
tracking and confidence-based selection strategies.

Each question entry in the dataset gets augmented with:
  - c_proposer: confidence when proposer generated this question
  - c_validator: confidence when validator validated this question
  - c_solver: confidence when solver last answered this question (updated over time)
  - c_judge: confidence when judge last evaluated this question (updated over time)
  - chain_agreement: min(c_validator, c_solver, c_judge) — updated when any role adds a score
  - times_selected: how many times this question was selected for solver training
  - last_selected_step: the step when this question was last selected
"""

import os
import threading
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray


@ray.remote
class COSEDatasetManager:
    """
    Ray actor that stores questions with per-role confidence scores
    and supports confidence-prioritized sampling.
    """

    def __init__(self):
        # Core storage: list of question dicts
        self.questions = []  # Each: {question, generation, thought, c_proposer, c_validator, c_solver, c_judge, ...}
        self.pairs = []      # Q+A pairs: {question, answer, c_solver, c_judge, ...}
        self.lock = threading.Lock()

    # -------------------------------------------------------------------
    # Add / update
    # -------------------------------------------------------------------

    def add_questions(self, entries: List[Dict], global_step: int):
        """Add new questions from proposer (with c_proposer, c_validator, val_score)."""
        with self.lock:
            for entry in entries:
                item = {
                    "question": entry.get("question", ""),
                    "generation": entry.get("generation", ""),
                    "thought": entry.get("thought", ""),
                    "c_proposer": entry.get("c_proposer", 0.0),
                    "c_validator": entry.get("c_validator", 0.0),
                    # Validator <score> for this question (normalized [0,1]).
                    # Used as the val_score factor in the unified importance formula.
                    "val_score": entry.get("val_score", 1.0),
                    "c_solver": None,       # Not yet solved
                    "c_judge": None,        # Not yet judged
                    "last_p": None,         # Last Judge <score> on this Q's last answer
                    "chain_agreement": None, # Computed after solver+judge
                    "answer": "",           # Filled when solved
                    "times_selected": 0,
                    "last_selected_step": -1,
                    "created_step": global_step,
                }
                self.questions.append(item)
            return len(self.questions)

    def update_solver_confidence(self, indices: List[int], confidences: List[float],
                                  answers: List[str], global_step: int):
        """Update c_solver and answer for selected questions after solver run."""
        with self.lock:
            for idx, conf, ans in zip(indices, confidences, answers):
                if 0 <= idx < len(self.questions):
                    self.questions[idx]["c_solver"] = conf
                    self.questions[idx]["answer"] = ans
                    self.questions[idx]["last_selected_step"] = global_step
                    self._recompute_chain(idx)

    def update_judge_confidence(self, indices: List[int], confidences: List[float]):
        """Update c_judge for selected questions after judge run."""
        with self.lock:
            for idx, conf in zip(indices, confidences):
                if 0 <= idx < len(self.questions):
                    self.questions[idx]["c_judge"] = conf
                    self._recompute_chain(idx)

    def update_last_p(self, indices: List[int], p_values: List[float]):
        """Update last_p (Judge's normalized <score> on the most recent
        Solver answer for this question). Used by the PER-style priority
        in select_for_solver(strategy="importance")."""
        with self.lock:
            for idx, p in zip(indices, p_values):
                if 0 <= idx < len(self.questions):
                    self.questions[idx]["last_p"] = float(p)

    def _recompute_chain(self, idx: int):
        """Recompute chain_agreement for a single question."""
        q = self.questions[idx]
        scores = []
        if q["c_validator"] is not None:
            scores.append(q["c_validator"])
        if q["c_solver"] is not None:
            scores.append(q["c_solver"])
        if q["c_judge"] is not None:
            scores.append(q["c_judge"])
        q["chain_agreement"] = min(scores) if scores else None

    def add_pairs(self, entries: List[Dict], global_step: int):
        """Add Q+A pairs for compatibility with parent DatasetManager interface."""
        with self.lock:
            for entry in entries:
                self.pairs.append({
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "generation": entry.get("generation", ""),
                    "thought": entry.get("thought", ""),
                    "created_step": global_step,
                })
            return len(self.pairs)

    # -------------------------------------------------------------------
    # Selection strategies
    # -------------------------------------------------------------------

    def select_for_solver(self, batch_size: int, strategy: str = "confidence_curriculum",
                          global_step: int = 0) -> Tuple[List[Dict], List[int]]:
        """
        Select questions for solver training from the full dataset.

        Returns (selected_items, selected_indices) where indices reference
        self.questions positions (for later confidence update).

        Strategies:
          - importance: priority = val_score × percentile_rank(c_validator).
                        Question-level importance from Validator-side signals only:
                        the Validator certifies the question, so question sampling
                        is governed by Validator confidence. Judge confidence is
                        an answer-level concept and doesn't enter buffer sampling.
          - confidence_curriculum: prioritize medium c_solver (learning frontier)
          - uniform: random sample (baseline)
          - new_first: prioritize never-solved questions
        """
        with self.lock:
            n = len(self.questions)
            if n == 0:
                return [], []

            actual_batch = min(batch_size, n)

            if strategy == "importance":
                # Question-level importance with three factors:
                #   priority(q) = val_score(q) × c_validator(q) × learnability(q)
                #
                # val_score, c_validator: Validator-side quality signals
                # frozen at insertion. The Validator certifies the question;
                # its <score> and its confidence in giving that score govern
                # the question's intrinsic worth.
                #
                # learnability(q) = 4 · last_p · (1 − last_p)         (if solved)
                #                 = 1.0                                 (if never solved)
                # Bell-curve "zone of proximal development" — peaks at
                # last_p = 0.5 (Solver right half the time = learning
                # frontier), zero at both extremes (trivial OR unsolvable).
                # Floor at 0.1 prevents starvation when Solver perfectly
                # aces or perfectly fails a question.
                # Never-solved questions get learnability=1.0 so they're
                # tried at least once before drifting toward frontier
                # selection.
                val_scores = np.array(
                    [float(q.get("val_score", 1.0)) for q in self.questions],
                    dtype=np.float64,
                )
                c_vals = np.array(
                    [float(q.get("c_validator", 0.0)) for q in self.questions],
                    dtype=np.float64,
                )

                learnability = np.empty(n, dtype=np.float64)
                # Allow swap of bell-shaped 4p(1-p) for the older 1-p reward via env var (ablation)
                learnability_form = os.environ.get("COSE_LEARNABILITY", "bell")
                for i, q in enumerate(self.questions):
                    p = q.get("last_p")
                    if p is None:
                        learnability[i] = 1.0
                    else:
                        p = float(p)
                        if learnability_form == "linear":
                            learnability[i] = max(1.0 - p, 0.1)
                        else:
                            learnability[i] = max(4.0 * p * (1.0 - p), 0.1)

                priorities = (
                    np.clip(val_scores, 0.0, 1.0)
                    * np.clip(c_vals, 0.0, 1.0)
                    * learnability
                    + 1e-6
                )

                total = priorities.sum()
                probs = priorities / total if total > 0 else np.ones(n) / n
                indices = np.random.choice(n, size=actual_batch, replace=False, p=probs).tolist()

            elif strategy == "uniform":
                indices = np.random.choice(n, size=actual_batch, replace=False).tolist()

            elif strategy == "new_first":
                # Prioritize never-solved, then random from rest
                never_solved = [i for i in range(n) if self.questions[i]["c_solver"] is None]
                solved = [i for i in range(n) if self.questions[i]["c_solver"] is not None]
                if len(never_solved) >= actual_batch:
                    indices = np.random.choice(never_solved, size=actual_batch, replace=False).tolist()
                else:
                    remaining = actual_batch - len(never_solved)
                    if solved:
                        extra = np.random.choice(solved, size=min(remaining, len(solved)), replace=False).tolist()
                    else:
                        extra = []
                    indices = never_solved + extra

            elif strategy == "confidence_curriculum":
                # Priority based on solver confidence:
                #   - Never-solved questions → high priority (must try once)
                #   - Medium c_solver → highest priority (learning frontier)
                #   - High c_solver → low priority (mastered)
                #   - Low c_solver → medium priority (hard but worth revisiting)
                priorities = np.zeros(n)
                for i in range(n):
                    q = self.questions[i]
                    if q["c_solver"] is None:
                        # Never solved — must try. Give high but not maximum priority
                        # so we still include some frontier questions
                        priorities[i] = 0.8
                    else:
                        # Gaussian peaking at median confidence across dataset
                        # We use raw c_solver value here (not percentile rank) to be
                        # comparable across steps. The peak targets "medium" confidence.
                        c = q["c_solver"]
                        # Find median of all solved questions
                        solved_confs = [self.questions[j]["c_solver"] for j in range(n)
                                        if self.questions[j]["c_solver"] is not None]
                        if solved_confs:
                            median_c = np.median(solved_confs)
                            std_c = max(np.std(solved_confs), 1e-6)
                            # Gaussian centered at median
                            priorities[i] = np.exp(-0.5 * ((c - median_c) / std_c) ** 2)
                        else:
                            priorities[i] = 0.5

                    # Freshness bonus: questions not selected recently get a small boost
                    steps_since = global_step - q["last_selected_step"] if q["last_selected_step"] >= 0 else 100
                    freshness = min(steps_since / 10.0, 1.0)  # caps at 1.0 after 10 steps
                    priorities[i] *= (0.5 + 0.5 * freshness)  # blend priority with freshness

                # Normalize to probability distribution
                total = priorities.sum()
                if total > 0:
                    probs = priorities / total
                else:
                    probs = np.ones(n) / n

                indices = np.random.choice(n, size=actual_batch, replace=False, p=probs).tolist()

            else:
                raise ValueError(f"Unknown selection strategy: {strategy}")

            # Mark as selected
            for idx in indices:
                self.questions[idx]["times_selected"] += 1
                self.questions[idx]["last_selected_step"] = global_step

            selected = [deepcopy(self.questions[idx]) for idx in indices]
            return selected, indices

    def select_for_judge(self, batch_size: int, global_step: int = 0) -> Tuple[List[Dict], List[int]]:
        """Select Q+A pairs for judge — only questions that have answers."""
        with self.lock:
            # Only select questions that have been solved (have an answer)
            solved_indices = [i for i in range(len(self.questions))
                             if self.questions[i]["answer"] and self.questions[i]["c_solver"] is not None]
            if not solved_indices:
                return [], []

            actual_batch = min(batch_size, len(solved_indices))
            indices = np.random.choice(solved_indices, size=actual_batch, replace=False).tolist()
            selected = [deepcopy(self.questions[idx]) for idx in indices]
            return selected, indices

    # -------------------------------------------------------------------
    # Getters (compatibility + stats)
    # -------------------------------------------------------------------

    def get_size(self) -> int:
        return len(self.questions)

    def get_pair_size(self) -> int:
        return len(self.pairs)

    def get_stats(self) -> Dict:
        """Return summary statistics about the dataset."""
        with self.lock:
            n = len(self.questions)
            if n == 0:
                return {"total": 0}

            n_solved = sum(1 for q in self.questions if q["c_solver"] is not None)
            n_judged = sum(1 for q in self.questions if q["c_judge"] is not None)
            n_with_chain = sum(1 for q in self.questions if q["chain_agreement"] is not None)

            solver_confs = [q["c_solver"] for q in self.questions if q["c_solver"] is not None]
            chain_confs = [q["chain_agreement"] for q in self.questions if q["chain_agreement"] is not None]

            return {
                "total": n,
                "solved": n_solved,
                "judged": n_judged,
                "with_chain": n_with_chain,
                "c_solver_mean": float(np.mean(solver_confs)) if solver_confs else 0.0,
                "c_solver_median": float(np.median(solver_confs)) if solver_confs else 0.0,
                "chain_mean": float(np.mean(chain_confs)) if chain_confs else 0.0,
                "avg_times_selected": float(np.mean([q["times_selected"] for q in self.questions])),
            }

    def get_questions_as_general_format(self) -> List[Dict]:
        """Return questions in parent DatasetManager format (for compatibility)."""
        with self.lock:
            return [
                {"question": q["question"], "generation": q["generation"], "thought": q["thought"]}
                for q in self.questions
            ]

    def get_pairs_as_general_format(self) -> List[Dict]:
        """Return pairs in parent DatasetManager format."""
        with self.lock:
            return deepcopy(self.pairs)
