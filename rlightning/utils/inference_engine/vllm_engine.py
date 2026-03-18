import os

from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class vLLMEngine:
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_USE_V1"] = "1"
        import vllm

        self.llm = vllm.LLM(*args, **kwargs)

    def get_tokenizer(self):
        return self.llm.llm_engine.tokenizer.tokenizer

    def get_hidden_size(self):
        return (
            self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.language_model.config.hidden_size
        )

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                use_ray,
            ),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def generate(self, queries, sampling_params):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """
        responses = self.llm.generate(queries, sampling_params, use_tqdm=False)
        return responses


class VLLMEngineAsync:
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_USE_V1"] = "1"

        import vllm

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    def get_tokenizer(self):
        return self.llm.tokenizer.tokenizer

    def get_hidden_size(self):
        return 2048

    async def generate_async(self, queries, sampling_params, request_id=None):
        if request_id is None:
            from vllm.utils import random_uuid

            request_id = random_uuid()

        results_generator = self.llm.generate(queries, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output


def create_vllm_engine(model_config, rollout_mode="sync"):
    import vllm

    logger.debug(vllm.__version__)
    # assert vllm.__version__ > "0.8.2", "OpenRLHF only supports vllm > 0.8.2"

    if rollout_mode == "sync":
        vllm_engine_cls = vLLMEngine
    elif rollout_mode == "async":
        vllm_engine_cls = VLLMEngineAsync
    else:
        raise ValueError(f"Invalid rollout mode: {rollout_mode}")

    pretrain_path = model_config.pretrain_path
    # enforce_eager: Disable CUDA graph in vLLM(default=False)
    enforce_eager = model_config.enforce_eager
    tensor_parallel_size = model_config.tensor_parallel_size
    seed = model_config.seed
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"

    max_model_len = (
        model_config.max_len
        if model_config.max_len
        else model_config.prompt_max_len + model_config.generate_max_len
    )
    max_model_len = model_config.prompt_max_len + model_config.generate_max_len
    enable_prefix_caching = model_config.enable_prefix_caching
    # full_determinism: Enable reproducible behavior during distributed training
    full_determinism = model_config.full_determinism
    gpu_memory_utilization = model_config.gpu_memory_utilization
    num_gpus = int(tensor_parallel_size == 1)
    enable_lora = model_config.enable_lora

    # use_hybrid_engine = shared_pg is not None
    # if use_hybrid_engine and tensor_parallel_size == 1:
    #     # every worker will use 0.2 GPU, so that we can schedule
    #     # 2 instances on the same GPUs.
    #     num_gpus = 0.2

    # if not use_hybrid_engine:
    #     # Create a big placement group to ensure that all engines are packed
    #     bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
    #     shared_pg = placement_group(bundles, strategy="PACK")
    #     ray.get(shared_pg.ready())

    # bundle_indices = None
    # if tensor_parallel_size > 1:
    #     bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

    vllm_engine = vllm_engine_cls(
        model=pretrain_path,
        enforce_eager=enforce_eager,
        # worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        distributed_executor_backend=distributed_executor_backend,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        dtype="bfloat16",
        trust_remote_code=True,
        # full_determinism=full_determinism,
        gpu_memory_utilization=gpu_memory_utilization,
        # bundle_indices=bundle_indices,
        # num_gpus=num_gpus,
        # enable_sleep_mode=vllm_enable_sleep,
        # agent_func_path=agent_func_path,
        enable_lora=enable_lora,
    )

    # if vllm_enable_sleep:
    #     batch_vllm_engine_call(vllm_engines, "sleep")
    return vllm_engine
