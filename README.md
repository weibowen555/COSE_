# COSE: Confidence-Orchestrated Self-Evolution

COSE is a four-role self-play RL framework for large language models. A
single LLM plays four roles — **Proposer**, **Validator**, **Solver**, and
**Judge** — and an *intrinsic confidence signal* (computed directly from
next-token logits) drives quality gating, curriculum scheduling, replay
prioritization, and per-sample gradient weighting. No external reward
model and no human-labeled data are needed; all rewards are produced by
the model itself, with confidence-derived weighting to keep the signal
robust.


---

## Table of Contents

1. [The four roles](#the-four-roles)
2. [Confidence signals](#confidence-signals)
3. [Mechanisms](#mechanisms)
4. [Repository layout](#repository-layout)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Ablations & hyperparameter studies](#ablations--hyperparameter-studies)
10. [Acknowledgements](#acknowledgements)

---

## The four roles

In every training step the same LLM (with frozen / shared weights) plays
four distinct roles via different prompt templates:

| Role | Input | Output | Loss-affecting role |
|---|---|---|---|
| **Proposer** | _(none, no-seed mode)_ or one reference task | `<question>…</question>` | Trained on `R_prop` |
| **Validator** | the proposed question | `<score>1–10</score>` (and `<think>…</think>` justification) | Produces $v$, $c_V$ used downstream |
| **Solver** | the validated question | `<answer>…</answer>` (with optional CoT) | Trained on `R_solv` |
| **Judge** | (question, answer) | one or two `<score>1–10</score>` tags | Produces judge confidence $c_J$ and answer score $p$ |

Question-side and answer-side scores from the same Judge head can be
emitted either as a *combined* `judge` template (two scores in one
generation) or as *separate* `judge_question` / `judge_answer` templates.
The default in the released config is `judge_answer`-only (`train_judge=false`),
because the question side is delegated to the Validator.

---

## Confidence signals

Per-token confidence is computed from next-token logits $\boldsymbol{\ell}\in\mathbb{R}^V$
in [`verl/utils/intrinsic_signals.py`](../verl-confidence-signal/verl/utils/intrinsic_signals.py).
Available signals (registered via `@register_intrinsic_signal`):

| Name | Formula | Range |
|---|---|---|
| `max_logit` | $\max_v \ell_v$ | $(-\infty,\infty)$ |
| `max_prob` | $\max_v \mathrm{softmax}(\boldsymbol{\ell})_v$ | $[1/V,1]$ |
| `log_max_prob` | $\max_v \log\mathrm{softmax}(\boldsymbol{\ell})_v$ | $(-\infty,0]$ |
| `top_k_margin` | $\ell_{(1)} - \ell_{(2)}$ | $[0,\infty)$ |
| `entropy` | $H(p)$ | $[0,\log V]$ |
| `neg_entropy` | $-H(p)$ | $[-\log V,0]$ |
| `logit_variance` | $\mathrm{Var}_v(\ell_v)$ | $[0,\infty)$ |
| `self_certainty` | $\log V - H(p)$ (KL to uniform) | $[0,\log V]$ |
| `normalized_peakedness` (default) | $\mathrm{clip}\bigl(2\cdot(1-H(p)/\log V)-1,\,0,\,1\bigr)$ | $[0,1]$ |

`normalized_peakedness` is the only signal that lands in $[0,1]$ and is
therefore directly usable as a probability-like weight; that is why COSE
adopts it as the default.

**Sequence-level aggregation.** The per-token score $c_t$ is reduced to a
single scalar per generation by taking the **bottom-α quantile** over the
output tokens (default $\alpha=0.30$):

$$c_{\text{seq}} \;=\; \mathrm{quantile}_{\alpha}\bigl(\{c_t : t \in \text{output}\}\bigr).$$

---

## Mechanisms

Four mechanisms use the confidence signal:

1. **Quality gate.** Drop the lowest-confidence X% of Proposer outputs
   before they enter the replay buffer (`cose.gating_enabled`,
   `gate_bottom_percentile`).
2. **Curriculum.** Apply a Gaussian sampling weight centred at the
   median-confidence percentile (`curriculum_enabled`, `curriculum_std`).
3. **Replay priority.** When sampling from the buffer, weight each
   question by $v\cdot c_V\cdot 4p(1-p)$ — Validator score times
   Validator confidence times bell-shaped learnability
   (`selection_strategy=importance`).
4. **Per-sample loss weighting.** Solver and Judge gradients are scaled
   by $w_S = \mathrm{clip}(v\cdot c_V\cdot c_J,\;0.1,\;1)$, so the
   gradients downstream of a low-confidence assessment are softly damped
   (`weighting_enabled`, `weighting_roles`).

All four mechanisms can be ablated independently (see
[ablations](#ablations--hyperparameter-studies) below).

---

## Repository layout

```
COSE/
├── absolute_zero_reasoner/
│   ├── configs/
│   │   ├── azr_ppo_trainer_general.yaml   # base PPO trainer config
│   │   └── cose_trainer.yaml              # COSE-specific overlay
│   ├── data_construction/
│   │   └── initial_prompt_templates/
│   │       ├── default.json               # original MAE templates
│   │       └── seir_4role.json            # COSE 4-role templates
│   ├── rewards/
│   │   └── reward_managers.py             # Proposer/Solver/Judge reward + GPT eval
│   ├── trainer/
│   │   └── ppo/
│   │       ├── cose_ray_trainer.py        # 4-role training loop
│   │       ├── cose_dataset_manager.py    # replay buffer + priority sampling
│   │       └── reason_rl_ray_trainer.py   # base reasoning trainer
│   └── utils/
│       └── benchmark_config.py            # 19-benchmark suite registry
├── data/
│   └── code_reason/test_answer.parquet    # shared train+val parquet
├── validation_datasets/                   # 19 per-benchmark *_test.parquet files
├── scripts/
│   ├── selfplay/
│   │   ├── cose_0.6b.sh                   # Qwen3-0.6B canonical launcher
│   │   ├── cose_3b.sh                     # Qwen2.5-3B-Instruct
│   │   ├── cose_4b.sh                     # Qwen3-4B-Base
│   │   ├── cose_llama3.2_3b.sh            # Llama-3.2-3B-Instruct
│   │   └── cose_val_only.sh               # standalone benchmark eval
│   └── evaluation/
│       └── eval_baselines.sh              # sweep launcher: AZR / R-Zero / COSE
├── outputs/                               # Hydra working-dir captures
└── README.md
```

---

## Installation

COSE depends on a customised fork of verl (with intrinsic-signals
support). Set up a clean venv:

```bash
# 1. Create venv at $SEIR_ROOT/seir_env
python3.10 -m venv ${SEIR_ROOT}/seir_env
source ${SEIR_ROOT}/seir_env/bin/activate
pip install --upgrade pip

# 2. Install the verl fork (intrinsic-signals branch) and the AZR submodule
cd ${SEIR_ROOT}/verl-confidence-signal && pip install -e .
cd ${SEIR_ROOT}/COSE                       && pip install -e .

# 3. Pin evalplus to AZR's compatible commit
pip install --no-deps git+https://github.com/evalplus/evalplus.git@d362e933
pip install "setuptools<81" stopit

# 4. flash-attn (needed for vLLM rollout)
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Environment variables (set once in `~/.bashrc` or pass via `sbatch --export`):

```bash
export SEIR_ROOT=/scratch/$USER/SEIR
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
# OpenAI key for the gpt-4.1-nano benchmark judge
echo 'sk-…' > ~/.openai_token && chmod 600 ~/.openai_token
export OPENAI_API_KEY=$(cat ~/.openai_token)
# HuggingFace token (Llama is gated)
echo 'hf_…' > ~/.hf_token && chmod 600 ~/.hf_token
```

---

## Quick start

```bash
# Launch the canonical COSE run on Qwen3-0.6B (4 × A100 80GB, ~24 h)
sbatch ${COSE_DIR}/scripts/selfplay/cose_0.6b.sh

# Evaluate a checkpoint on the 19-benchmark reasoning + math suite
MODEL=${COSE_DIR}/checkpoints/COSE_0.6B_…/global_step_100/actor/huggingface \
  sbatch ${COSE_DIR}/scripts/selfplay/cose_val_only.sh
```

---

## Training

Each backbone has a dedicated launcher under `scripts/selfplay/`. All
share the same logic — they only differ in batch size, response length,
GPU-memory fraction, and save frequency. Defaults:

| Backbone | bs | mini / micro | maxP / maxR | save_freq | wall |
|---|---|---|---|---|---|
| Qwen3-0.6B | 128 | 32 / 2 | 8192 / 16384 | every 20 steps | 2 d |
| Qwen3-4B-Base | 128 | 32 / 1 | 6144 / 8096 | every 20 steps | 2 d |
| Qwen2.5-3B-Instruct | 128 | 32 / 1 | 6144 / 8096 | every 20 steps | 2 d |
| Llama-3.2-3B-Instruct | 128 | 32 / 1 | 6144 / 8096 | every 20 steps | 2 d |

Shared hyperparameters: `lr=1e-6`, `reinforce_plus_plus` advantage,
`temperature=1.0`, `n_gpus_per_node=4`, `tensor_model_parallel_size=2`,
`ulysses_sequence_parallel_size=2`, `total_epochs=30`,
`test_freq=20` (benchmark eval every 20 steps), `kl_ctrl.kl_coef=0.0`,
`use_kl_loss=False`, `include_ref=False`.

To override the model at submission time:

```bash
MODEL=Qwen/Qwen3-8B  sbatch ${COSE_DIR}/scripts/selfplay/cose_4b.sh
```

To swap the confidence signal:

```bash
CONFIDENCE_SIGNAL=self_certainty  sbatch ${COSE_DIR}/scripts/selfplay/cose_llama3.2_3b.sh
```

---

## Evaluation

### Reasoning + math suite (19 benchmarks, gpt-4.1-nano judge)

A single checkpoint goes through `cose_val_only.sh`, which runs the same
benchmark pipeline as the in-training `val/benchmark_accuracy/*` metrics
but with `benchmark_max_samples=500` instead of the in-training default
(100). Output: a single `step:0 - val/benchmark_accuracy/...` line that
contains per-benchmark accuracies.

```bash
# Evaluate a trained COSE checkpoint
MODEL=${COSE_DIR}/checkpoints/COSE_4B_…/global_step_200/actor/huggingface \
  sbatch ${COSE_DIR}/scripts/selfplay/cose_val_only.sh
```

### Code suite (HumanEval+, MBPP+)

```bash
# evalplus + LCB on one checkpoint
MODEL=${COSE_DIR}/checkpoints/COSE_4B_…/global_step_200/actor/huggingface \
  TAG=cose_qwen3_4b_step200 \
  sbatch ${AZR_DIR}/scripts/evaluation/eval_code_one.sh
```

Output JSONs land under
`${AZR_DIR}/evalplus_results/{humaneval,mbpp}/<TAG>_eval_results.json`.

### Cross-framework sweep

`eval_baselines.sh` dispatches both code and reasoning evals on a curated
list of AZR / R-Zero / COSE checkpoints:

```bash
bash ${COSE_DIR}/scripts/evaluation/eval_baselines.sh             # all
ONLY=reason bash ${COSE_DIR}/scripts/evaluation/eval_baselines.sh # only reasoning
ONLY=code   bash ${COSE_DIR}/scripts/evaluation/eval_baselines.sh # only code
DRY_RUN=1   bash ${COSE_DIR}/scripts/evaluation/eval_baselines.sh # print, don't submit
```

For AZR / MAE / R-Zero checkpoints stored as FSDP shards (`model_world_size_N_rank_i.pt`),
first run `merge_fsdp_for_eval.sh` to materialise HuggingFace safetensors.

---

## Ablations & hyperparameter studies

### Mechanism ablations (RQ5)

| Variant | Switch |
|---|---|
| `(A)` w/o confidence weighting | `azr.cose.weighting_enabled=False` |
| `(B)` w/o confidence priority | `azr.cose.selection_strategy=uniform` |
| `(C)` linear $1-p$ learnability | env var `COSE_LEARNABILITY=linear` |

Convenience launchers:

```bash
# Llama-3.2-3B, ablation (A)
sbatch -J cose_llama3.2_3b_abA --export=ALL,WEIGHTING_ENABLED=False,ABLATION_TAG=abA \
  ${COSE_DIR}/scripts/selfplay/cose_llama3.2_3b_ablate.sh
```

### Hyperparameter studies (RQ6)

**Batch size.** Submit `cose_<bb>_bs.sh` with `BATCH_SIZE`:

```bash
BATCH_SIZE=32  sbatch ${COSE_DIR}/scripts/selfplay/cose_llama3.2_3b_bs.sh
```

**Confidence signal.** Submit `cose_llama3.2_3b_signal.sh` with `CONFIDENCE_SIGNAL`:

```bash
CONFIDENCE_SIGNAL=self_certainty  sbatch ${COSE_DIR}/scripts/selfplay/cose_llama3.2_3b_signal.sh
```

Both helper scripts adjust `ppo_mini_batch_size` proportionally to keep
inner PPO updates constant.

---

## Acknowledgements

COSE builds directly on:

- **Absolute Zero Reasoner** (LeapLabTHU) — the four-role self-play scaffold
- **verl** (volcengine) — the underlying RL trainer; we extend it with a
  registry of intrinsic confidence signals
- **R-Zero**, **Multi-Agent Evolve (MAE)** — the closest related
  self-evolving baselines we compare against

The 19-benchmark evaluation suite and the gpt-4.1-nano-based judge
infrastructure are inherited from AZR; we use them unchanged so that
COSE numbers are directly comparable to published AZR / R-Zero / MAE
numbers.
