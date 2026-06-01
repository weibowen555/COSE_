#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=cose_4b
#SBATCH --output=/scratch/%u/SEIR/logs/%x-%N-%j.out
#SBATCH --error=/scratch/%u/SEIR/logs/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --mem=160gb
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00

# ---- Paths (override at submit time if your layout differs) ----
SEIR_ROOT=${SEIR_ROOT:-<PATH_TO_SEIR_ROOT>}     # e.g. /scratch/$USER/SEIR
COSE_DIR=${COSE_DIR:-${SEIR_ROOT}/COSE}
AZR_DIR=${AZR_DIR:-${SEIR_ROOT}/Absolute-Zero-Reasoner}
RZERO_DIR=${RZERO_DIR:-${SEIR_ROOT}/R-Zero}
MAE_DIR=${MAE_DIR:-${SEIR_ROOT}/Multi-agent-evolve}
VENV=${VENV:-${SEIR_ROOT}/seir_env/bin/activate}
HF_CACHE=${HF_CACHE:-${SEIR_ROOT}/.cache}


# =============================================================================
# COSE canonical training launcher — cose_4b (Qwen/Qwen3-4B-Base)
# Reconstructed from training-log set-x trace for run "COSE_4B".
# Produces checkpoints under ${COSE_DIR}/checkpoints/${RUN_NAME}/.
# =============================================================================

set -x
umask 0027
unset ROCR_VISIBLE_DEVICES
mkdir -p /scratch/${USER}/SEIR/logs

module load gnu10
cd ${COSE_DIR}
source ${VENV}

export NCCL_DEBUG=INFO
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=WARN
export RAY_DASHBOARD_AGENT_ENABLED=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export ACCELERATE_LOG_LEVEL=info
export HF_HOME=${HF_CACHE}
export HF_DATASETS_CACHE=${HF_CACHE}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_CACHE}/hub
if [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN=$(cat "${HOME}/.hf_token")
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
export WANDB_API_KEY=${WANDB_API_KEY:-<YOUR_WANDB_API_KEY>}
export TMPDIR=/tmp/${USER}_cose_$$
mkdir -p "${TMPDIR}"

export BENCH_JUDGE_BACKEND=openai
export BENCH_JUDGE_MODEL=gpt-4.1-nano
export NIM_PER_KEY_RPM=500
export BENCH_USE_EXACT_MATCH=0

MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
CONFIDENCE_SIGNAL=${CONFIDENCE_SIGNAL:-normalized_peakedness}

RUN_NAME=COSE_4B_$(date +%Y%m%d_%H%M%S)
DEFAULT_LOCAL_DIR=${COSE_DIR}/checkpoints/${RUN_NAME}

PYTHONUNBUFFERED=1 python -m absolute_zero_reasoner.main_cose \
    --config-name=cose_trainer \
    +benchmark_max_samples=100 \
    azr.benchmark_names=[math,gsm8k,aime24,mmlu_pro,gpqa,arc_challenge,bbh,livebench_reasoning,truthfulqa,ifeval] \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=1312 \
    data.max_prompt_length=6144 \
    data.max_validation_prompt_length=6144 \
    data.max_response_length=8096 \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.intrinsic_signal=${CONFIDENCE_SIGNAL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='COSE' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.remove_previous_ckpt_in_save=False \
    trainer.del_local_ckpt_after_load=True \
    trainer.test_freq=20 \
    trainer.val_before_train=true \
    reward_fn.extraction_type=boxed \
    reward_fn.math_metric=deepscaler \
    reward_fn.llm_model_name="meta/llama-3.1-8b-instruct" \
    +reward_fn.benchmark_max_workers=15 \
    reward_fn.temperature=1.0 \
    reward_fn.max_tokens=1000 \
    reward_fn.top_p=0.95 \
    reward_fn.stream=true \
    azr.task_type=general \
    azr.data_selection_strategy.update_iteration=1 \
    azr.pretrain_pred_steps=-1 \
    azr.problem_types=['general'] \
    azr.pred_data_mix_strategy=uniform_total \
    azr.judge_data_mix_strategy=uniform_total \
    azr.train_judge=false \
    azr.train_solve=true \
    azr.train_validate=false \
    azr.with_answer_generation=False \
    azr.train_propose=True \
    azr.cose.confidence_signal=${CONFIDENCE_SIGNAL} \
    azr.cose.gating_enabled=True \
    azr.cose.curriculum_enabled=True \
    azr.cose.weighting_enabled=True \
    azr.cose.selection_strategy=importance \
    azr.cose.no_seed=True \
    +azr.cose.confidence_bottom_frac=1.0 \
    azr.reward.n_samples=1 \
    azr.reward.generation_reward_config.format_reward=false \
    azr.reward.generation_reward_config.include_references=0.5 \
    azr.reward.generation_reward_config.generation_accuracy_convertion=inverse \
    azr.reward.generation_reward_config.answer_diversity_reward.hierarchical=false \
    azr.data_selection_strategy.content_max_length=6144 \
    azr.data_selection_strategy.valid_question_filter=all \
    azr.data_selection_strategy.batched_estimate=false \
    azr.data_selection_strategy.io_n=1 \
    ++trainer.npu_profile.options=null \
    ++actor_rollout_ref.actor.profiler._target_=verl.utils.profiler.ProfilerConfig \
    ++actor_rollout_ref.actor.profiler.ranks=[] \
    ++actor_rollout_ref.ref.profiler._target_=verl.utils.profiler.ProfilerConfig \
    ++actor_rollout_ref.ref.profiler.ranks=[] \
    ++actor_rollout_ref.rollout.profiler._target_=verl.utils.profiler.ProfilerConfig \
    ++actor_rollout_ref.rollout.profiler.ranks=[] \
    ++critic.profiler._target_=verl.utils.profiler.ProfilerConfig \
    ++critic.profiler.ranks=[] \
    trainer.resume_mode=disable \
    trainer.total_epochs=30 \
    prompt_manager.template_file=absolute_zero_reasoner/data_construction/initial_prompt_templates/seir_4role.json $@ \
    2>&1 | tee /scratch/${USER}/SEIR/logs/cose_${RUN_NAME}.log