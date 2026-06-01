"""
COSE v1 entry point.

Reuses MAE's infrastructure unchanged; swaps in COSERayPPOTrainer which
adds per-role confidence logging on top of GeneralIORayPPOTrainer.
"""
import ray
import hydra
import json
import os
from typing import List

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from cose.trainer.ppo.cose_ray_trainer import COSERayPPOTrainer
from cose.rewards.reward_managers import (
    GeneralIORewardManager,
    BenchmarkEvaluationRewardManager,
)


def load_api_keys(api_file_path: str = "api.json") -> List[str]:
    possible_paths = [
        api_file_path,
        os.path.join(os.path.dirname(__file__), api_file_path),
        os.path.join(os.path.dirname(__file__), "..", api_file_path),
        os.path.join(os.path.dirname(__file__), "..", "..", api_file_path),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading API keys from: {path}")
            with open(path, "r") as f:
                data = json.load(f)
                api_keys = data.get("api_keys", [])
                if api_keys:
                    return api_keys
    print("Warning: No API keys found.")
    return []


@hydra.main(config_path="configs", config_name="cose_trainer", version_base=None)
def main(config):
    print("=" * 60)
    print("COSE v1: Intrinsic Confidence Logging for MAE")
    print("=" * 60)

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    OmegaConf.resolve(config)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.experiment_timestamp = timestamp
    if config.trainer.default_local_dir is None:
        config.trainer.default_local_dir = f"./checkpoints/cose_{timestamp}"
    os.makedirs(config.trainer.default_local_dir, exist_ok=True)
    config.agent_output_dir = config.trainer.default_local_dir

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    processor = None

    api_keys = load_api_keys()

    # Worker setup
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
        )
        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
        )
        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # Training reward manager: reuse MAE's GeneralIORewardManager (works correctly)
    reward_fn = GeneralIORewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        reward_fn_extraction_type=config.reward_fn.extraction_type,
        splitter=config.reward_fn.splitter,
        output_path=config.trainer.default_local_dir,
        generation_reward_config=config.cose.reward.generation_reward_config,
        eval_reward_config=getattr(config.cose.reward, "eval_reward_config", {}),
        model_name=getattr(
            config.reward_fn, "llm_model_name", "meta/llama-3.1-405b-instruct"
        ),
        max_prompt_length=config.data.max_prompt_length,
        temperature=getattr(config.reward_fn, "temperature", 0.7),
        max_tokens=getattr(config.reward_fn, "max_tokens", 1000),
        top_p=getattr(config.reward_fn, "top_p", 0.95),
        stream=getattr(config.reward_fn, "stream", True),
        boxed_retry=config.reward_fn.boxed_retry,
        judge_with_actor=config.reward_fn.judge_with_actor,
        use_format_reward=getattr(config.cose, "use_format_reward", True),
        agent_output_dir=config.agent_output_dir,
        api_keys=api_keys,
    )

    val_reward_fn = BenchmarkEvaluationRewardManager(
        tokenizer=tokenizer,
        model_name=getattr(
            config.cose, "benchmark_eval_model", "meta/llama-3.1-70b-instruct"
        ),
        temperature=getattr(config.reward_fn, "temperature", 0.0),
        max_tokens=getattr(config.reward_fn, "max_tokens", 500),
        top_p=getattr(config.reward_fn, "top_p", 0.95),
        stream=getattr(config.reward_fn, "stream", True),
        boxed_retry=config.reward_fn.boxed_retry,
        api_keys=api_keys,
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    wandb_tags = ["cose_v1"]
    wandb_tags.extend(list(config.cose.problem_types))
    if config.trainer.wandb_tags is not None:
        existing = (
            config.trainer.wandb_tags.split(",")
            if isinstance(config.trainer.wandb_tags, str)
            else list(config.trainer.wandb_tags)
        )
        config.trainer.wandb_tags = wandb_tags + existing
    else:
        config.trainer.wandb_tags = wandb_tags

    trainer = COSERayPPOTrainer(
        past_epoch_window=config.cose.past_epoch_window,
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=None,
        benchmark_reward_fn=val_reward_fn,
    )

    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
