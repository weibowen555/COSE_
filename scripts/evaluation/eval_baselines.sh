#!/bin/bash
# =============================================================================
# Sweep: evaluate AZR + R-Zero + COSE baseline checkpoints on:
#   (a) COSE benchmark suite — 19 reasoning parquets, gpt-4.1-nano judge
#       (math, gsm8k, mmlu, mmlu_pro, gpqa, arc_challenge, bbh, truthfulqa,
#        livebench_reasoning, amc, minerva, olympiad, hellaswag, winogrande,
#        commonsenseqa, openbookqa, naturalquestions, triviaqa, squad)
#       → submitted via cose_val_only.sh (project "COSE-eval" on wandb)
#   (b) AZR code benchmarks — HumanEvalPlus + MbppPlus via evalplus,
#       greedy decoding, results land in
#       ${AZR_DIR}/evalplus_results/
#       → submitted via Absolute-Zero-Reasoner/scripts/evaluation/eval_code_one.sh
#
# Usage:
#     bash scripts/evaluation/eval_baselines.sh             # submit all
#     ONLY=reason bash scripts/evaluation/eval_baselines.sh # only (a)
#     ONLY=code   bash scripts/evaluation/eval_baselines.sh # only (b)
#     DRY_RUN=1   bash scripts/evaluation/eval_baselines.sh # print only
# =============================================================================


# ---- Paths (override at submit time if your layout differs) ----
SEIR_ROOT=${SEIR_ROOT:-<PATH_TO_SEIR_ROOT>}     # e.g. /scratch/$USER/SEIR
COSE_DIR=${COSE_DIR:-${SEIR_ROOT}/COSE}
AZR_DIR=${AZR_DIR:-${SEIR_ROOT}/Absolute-Zero-Reasoner}
RZERO_DIR=${RZERO_DIR:-${SEIR_ROOT}/R-Zero}
MAE_DIR=${MAE_DIR:-${SEIR_ROOT}/Multi-agent-evolve}
VENV=${VENV:-${SEIR_ROOT}/seir_env/bin/activate}
HF_CACHE=${HF_CACHE:-${SEIR_ROOT}/.cache}

set -e
cd ${COSE_DIR}

# Tag, checkpoint path
CHECKPOINTS=(
    # ---------- R-Zero (latest *merged* step; model_merger.py is hardcoded
    #     to run on global_step_5 — step_6 exists as raw FSDP only) ------
    "rzero_qwen3_4b      ${RZERO_DIR}/storage/models/qwen3-4b_questioner_v1/global_step_5/actor/huggingface"
    "rzero_qwen3_0p6b    ${RZERO_DIR}/storage/models/qwen3-0p6b_questioner_v1/global_step_5/actor/huggingface"
    "rzero_llama3p2_3b   ${RZERO_DIR}/storage/models/llama3p2-3b_questioner_v1/global_step_5/actor/huggingface"
    "rzero_qwen2p5_3b    ${RZERO_DIR}/storage/models/qwen2p5-3b_questioner_v1/global_step_5/actor/huggingface"
    # ---------- COSE — user-selected step per backbone (cross-framework comparison) ---
    "cose_llama3p2_3b_step100  ${COSE_DIR}/checkpoints/COSE_LLAMA3.2_3B_20260505_011212/global_step_100/actor/huggingface"
    "cose_qwen3_4b_step200     ${COSE_DIR}/checkpoints/COSE_4B_20260503_231653/global_step_200/actor/huggingface"
    "cose_qwen3_0p6b_step260   ${COSE_DIR}/checkpoints/COSE_0.6B_20260502_002809/global_step_260/actor/huggingface"
    "cose_qwen2p5_3b_step150   ${COSE_DIR}/checkpoints/COSE_3B_20260504_195252/global_step_150/actor/huggingface"
    # ---------- AZR (latest global_step per run) ---------------------------
    "azr_qwen3_4b        ${AZR_DIR}/checkpoints/azr_qwen3_4b_20260511_003751/test_answer/Qwen3-4B-Base/answer_conditional/global_step_90/actor/huggingface"
    "azr_qwen3_0p6b      ${AZR_DIR}/checkpoints/azr_qwen3_0p6b_20260511_052510/test_answer/Qwen3-0.6B/answer_conditional/global_step_180/actor/huggingface"
    "azr_llama3p2_3b     ${AZR_DIR}/checkpoints/azr_llama3p2_3b_20260511_010146/test_answer/Llama-3.2-3B-Instruct/answer_conditional/global_step_150/actor/huggingface"
    "azr_qwen2p5_3b      ${AZR_DIR}/checkpoints/azr_qwen2p5_3b_20260511_164534/test_answer/Qwen2.5-3B-Instruct/answer_conditional/global_step_50/actor/huggingface"
    # ---------- Optional: base models (sanity baseline, no training) ------
    # "base_qwen3_4b      Qwen/Qwen3-4B-Base"
    # "base_qwen2p5_3b    Qwen/Qwen2.5-3B-Instruct"
)

ONLY=${ONLY:-both}    # both | reason | code

submit() {
    local desc="$1"; shift
    if [ -n "$DRY_RUN" ]; then
        echo "DRY_RUN [$desc]: $*"
    else
        echo "submitting [$desc]: $*"
        "$@"
    fi
}

for entry in "${CHECKPOINTS[@]}"; do
    tag=$(echo "$entry" | awk '{print $1}')
    path=$(echo "$entry" | awk '{print $2}')

    if [ ! -d "$path" ] && [[ "$path" != */* ]]; then
        : # treat as HF repo id, allow
    elif [ ! -d "$path" ]; then
        echo "[skip] $tag: path does not exist: $path" >&2
        continue
    fi

    # (a) COSE reasoning benchmark suite — gpt-4.1-nano judge, ~30-60 min on 2× A100.
    if [ "$ONLY" = "both" ] || [ "$ONLY" = "reason" ]; then
        submit "reason ${tag}" \
            sbatch -J "eval_reason_${tag}" \
                   --export=ALL,MODEL="$path" \
                   ${COSE_DIR}/scripts/selfplay/cose_val_only.sh
    fi

    # (b) AZR code benchmarks — HumanEvalPlus + MbppPlus, greedy, ~30 min on 1× A100
    if [ "$ONLY" = "both" ] || [ "$ONLY" = "code" ]; then
        submit "code   ${tag}" \
            sbatch -J "eval_code_${tag}" --export=ALL,MODEL="$path",TAG="$tag" \
                   ${AZR_DIR}/scripts/evaluation/eval_code_one.sh
    fi
done