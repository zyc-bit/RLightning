"""Configuration models and utilities for RLightning using Pydantic and OmegaConf."""

import os
import warnings
from typing import Any, List, Literal, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Config(BaseModel):
    """
    A Config class based on Pydantic.
    """

    model_config = ConfigDict(frozen=False, extra="allow")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    @model_validator(mode="after")
    def _convert_extra_fields(self) -> "Config":
        """
        It converts all extra fields to Config recursively.
        """
        if self.model_extra:
            for key, value in self.model_extra.items():
                if isinstance(value, dict):
                    value = Config.from_dict(value)
                setattr(self, key, value)

        return self

    @classmethod
    def load_yaml(cls, file_path: str = None) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file. If None, returns an empty config.
        Returns:
            Config: An instance of the Config class populated with data from the YAML file.
        """
        if file_path is None:
            return cls()

        with open(file_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        return cls.from_dict(yaml_content)

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get the value of a field by name, with an optional default if the field does not exist.
        """
        return getattr(self, name, default)

    def __getitem__(self, key: str) -> Any:
        """
        Get the value of a field by name using dictionary-like access.
        """
        return getattr(self, key)

    @classmethod
    def from_omegaconf(cls, om_cfg: DictConfig) -> "Config":
        """
        Create and validate a config instance from an OmegaConf DictConfig.
        """
        plain_dict = OmegaConf.to_container(om_cfg, resolve=True)
        return cls.from_dict(plain_dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """
        Create and validate a config instance from a standard Python dictionary.
        """
        try:
            # catch Pydantic validation errors to provide clearer messages
            return cls.model_validate(data)
        except ValidationError as e:
            error_messages = []
            for error in e.errors():
                field_path = ".".join(map(str, error["loc"]))
                message = error["msg"]
                error_messages.append(f"  - field '{field_path}': {message}")

            final_message = (
                "Error to load config, found the following problems:\n"
                + "\n".join(error_messages)
                + "\nPlease check your configuration."
            )
            raise ValueError(final_message)

    def to_dict(self) -> dict:
        """
        Recursively convert the model instance to a standard Python dictionary.
        """
        return self.model_dump()

    def to_yaml(self) -> str:
        """
        Recursively convert the model instance to a YAML-formatted string.
        """
        config_dict = self.to_dict()
        return yaml.safe_dump(config_dict, sort_keys=False)


class EnvConfig(Config):
    """
    Environment configuration.
    """

    name: str
    """ User defined name of the environment for identification purposes """
    backend: str
    """ The simulator backend type of the environment """
    task: str
    """ The task name of the environment """
    num_workers: int = 1
    """ Number of environment workers with the same configuration"""
    num_envs: int = 1
    """ Number of parallel vectorized environments in one environment instance"""
    max_episode_steps: int | None = None
    """ Maximum number of steps per episode """
    num_cpus: int = 1
    """ Number of CPUs to allocate for one environment instance, only valid when env is remote """
    num_gpus: float = 0.0
    """ Number of GPUs to allocate for one environment instance, only valid when env is remote"""
    env_kwargs: Config = Config()
    """ Configuration used to initialize"""
    init_params: Config | None = None
    """ The initialization parameters for the environment """
    policy_setup: str = "widowx"
    """ The policy setup for the environment """
    evaluate_cfg: Config | None = None
    """ The evaluation configuration for the environment """

    @model_validator(mode="after")
    def sanity_check(self) -> "EnvConfig":
        """
        Sanity check for the environment configuration.
        """

        internal_supported_backends = [
            "ale",
            "maniskill",
            "mujoco",
            "piper",
            "env_server",
            "isaac_marl",
            "isaac_manager_based",
        ]

        vector_env_supported_backends = [
            "maniskill",
            "isaac_manager_based",
            "isaac_marl",
        ]

        if self.backend not in internal_supported_backends:
            warnings.warn(
                f"Backend '{self.backend}' is not officially supported by RLightning. You may use "
                "a custom env implementation. If not, please refer to the supported backends list."
                f"Supported backends are: {internal_supported_backends}."
            )

        # Check vectorized environment support
        if (
            self.num_envs > 1
            and self.backend in internal_supported_backends
            and self.backend not in vector_env_supported_backends
        ):
            raise ValueError(
                f"Vectorized environments are only supported for backends: "
                f"{vector_env_supported_backends}, but got backend '{self.backend}'."
            )

        return self

    @model_validator(mode="after")
    def setup_robot_control_mode_for_maniskill(self) -> "EnvConfig":
        """
        setup robot control mode for maniskill environments
        """
        if "maniskill" in self.backend and self.init_params is not None:

            def get_robot_control_mode(robot: str) -> str:
                if "google_robot_static" in robot:
                    return (
                        "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_pos_interpolate_by_planner"
                    )
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                else:
                    raise NotImplementedError(f"Robot {robot} not supported")

            self.init_params.control_mode = get_robot_control_mode(self.policy_setup)

        return self


class SamplerConfig(Config):
    """
    Configuration for the sampler.
    """

    type: Literal["all", "batch", "uniform"]
    """Support three types of sampling strategies: all, batch, uniform

        - all: sample all data from the buffer
        - batch: sample a batch of data sequentially
        - uniform: sample a batch of data uniformly from the buffer
    """


class StorageConfig(Config):
    """
    Configuration for the storage backend of the buffer.
    """

    mode: Literal["circular", "fixed"] = "circular"
    """ Storage mode for buffer behavior when capacity is reached """
    type: Literal["unified", "sharded"] = "unified"
    """ The type of the storage"""
    unit: Literal["transition", "episode"] = "transition"
    """ The storage unit """
    device: Literal["cpu", "cuda"] = "cpu"
    """ The device on which the storage is located"""

    @model_validator(mode="after")
    def validate_strategy(self) -> "StorageConfig":
        """Setup the number of shards for sharded storage based on the cluster resources."""
        if self.type == "sharded":
            assert self.device == "cpu", "Sharded storage only supports CPU device"
        return self


class BufferConfig(Config):
    """
    Configuration for the data buffer.
    """

    type: Literal["ReplayBuffer", "RolloutBuffer"]
    """ The type of the buffer"""
    capacity: int
    """ Maximum number of transitions the buffer can store """
    sampler: SamplerConfig | None = None
    """ The sampler configuration"""
    auto_truncate_episode: bool = False
    """ Whether to automatically truncate episodes when they are done"""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    """ The storage backend configuration"""
    node_affinity_env: bool = False
    """ Whether to enable node affinity for environment workers"""
    node_affinity_train: bool = False
    """ Whether to enable node affinity for training workers"""

    @model_validator(mode="after")
    def setup_default_sampler(self) -> "BufferConfig":
        """Set up default sampler based on buffer type if not provided sampler config."""
        default_buffer_samplers = {
            "ReplayBuffer": SamplerConfig(type="uniform"),
            "RolloutBuffer": SamplerConfig(type="all"),
        }

        if self.sampler is None:
            if self.type in default_buffer_samplers:
                self.sampler = default_buffer_samplers[self.type]
            else:
                raise ValueError(f"Sampler config must be provided for buffer type '{self.type}'.")
        return self


class WeightBufferConfig(Config):
    """
    Configuration for the weight buffer.
    """

    type: Literal["WeightBuffer", "CPUWeightBuffer", "ShardedWeightBuffer"] = "WeightBuffer"
    """ The type of the weight buffer """
    buffer_strategy: Literal["None", "Double", "Shared", "Sharded"] = "Double"
    """ The buffer strategy to use """

    @model_validator(mode="after")
    def validate_strategy(self) -> "WeightBufferConfig":
        """
        Validate the buffer strategy based on the buffer type.
        """
        if self.buffer_strategy == "Shared" and self.type != "CPUWeightBuffer":
            raise ValueError("Shared buffer strategy is only supported with CPUWeightBuffer.")
        if self.buffer_strategy == "Sharded" and self.type != "ShardedWeightBuffer":
            raise ValueError("Sharded buffer strategy is only supported with ShardedWeightBuffer.")
        return self


class PolicyConfig(Config):
    """
    Policy configuration.
    """

    type: str
    """ The type of the policy, used it to find the policy cls"""
    backend: Config | None = None
    """ The inference backend of eval policy """

    train_num_gpus: float = 1.0
    """ Number of GPUs to allocate for one training policy instance, only valid when train policy is remote """

    eval_num_gpus: float = 1.0
    """ Number of GPUs to allocate for one evaluation policy instance, only valid when eval policy is remote """

    model_cfg: Config | None = None
    """ The model configuration """

    optim_cfg: Config | None = None
    """ The optimizer configuration """

    rollout_mode: Literal["sync", "async"] = "sync"
    """ The policy inference concurrency mode, sync or async """

    router_type: Literal["simple", "node_affinity"] = "simple"
    """ Router type for async rollout mode. "simple" uses load balancing,
    "node_affinity" routes env requests to policies on the same node. """

    weight_buffer: WeightBufferConfig = Field(default_factory=WeightBufferConfig)
    """ Configuration for the weight buffer """

    policy_kwargs: Config | None = None
    """ Customized keyword arguments for policy initialization """


class TrainConfig(Config):
    """
    Training configuration.
    """

    max_epochs: int
    """ The maximum number of training epochs """

    batch_size: int = 64
    """ The training batch size """

    max_rollout_steps: int = -1
    """ The maximum number of rollout steps during once rollout stage """
    max_eval_rollout_steps: int = -1
    """ The maximum number of rollout steps during once evaluation stage """

    lr: float = 0.0003
    """ The learning rate """
    parallel: Literal["ddp", None] = None
    """ The parallel mode for training, default is None, i.e., no parallel """
    eval_interval: int = -1
    """ The evaluation interval (in epochs) during training """
    save_interval: int = -1
    """ The model saving interval (in epochs) during training """
    save_dir: str = None
    """ The directory to save checkpoints """

    @model_validator(mode="after")
    def setup_default_ckpt_save_dir(self):
        """Setup default checkpoint save directory under the log dir if not provided."""
        if self.save_dir is None:
            log_file = os.environ.get("RLIGHTNING_LOG_FILE", None)
            if log_file is not None:
                log_dir = os.path.dirname(log_file)
                self.save_dir = os.path.join(log_dir, "checkpoints")

        return self


class LogConfig(Config):
    """
    Logging configuration. Including experiment manager and logging level.
    """

    # logging configurations
    level: Literal["DEBUG", "INFO", "WARINING", "ERROR", "CRITICAL"] = "DEBUG"
    """ The logging level """

    # experiment manager configurations
    backend: Literal["tensorboard", "wandb", "swanlab"] = "tensorboard"
    """ The experiment manager backend """
    project: str = "default_project"
    """ The project name"""
    name: str = "default_exp"
    """ The experiment name """
    log_dir: str = "./runs"
    """ The directory to save experiment logs """
    mode: Literal["online", "offline", "shared", "disabled", "cloud", "local"] | None = None
    """ The mode for wandb, online or offline, not work for other backends """

    @model_validator(mode="after")
    def mode_sanity_check_and_setup_local_by_default(self):
        """Sanity check for mode configuration and setup default mode if not provided."""
        # for wandb
        if self.backend == "wandb":
            if self.mode is None:
                self.mode = "offline"
            else:
                if self.mode not in ["online", "offline", "disabled", "shared"]:
                    raise ValueError(
                        f"Invalid mode '{self.mode}' for wandb backend. "
                        f"Supported modes are 'online', 'offline', 'shared', 'disabled'."
                    )
        elif self.backend == "swanlab":
            if self.mode is None:
                self.mode = "local"
            else:
                if self.mode not in ["cloud", "local", "disabled"]:
                    raise ValueError(
                        f"Invalid mode '{self.mode}' for swanlab backend. "
                        f"Supported modes are 'cloud', 'local', 'disabled'."
                    )
        elif self.backend == "tensorboard":
            if self.mode is not None:
                warnings.warn(f"Mode is not applicable for tensorboard backend, " f"but got mode '{self.mode}'.")

        return self


class ResourcePoolConfig(Config):
    """
    Manual resource pool definition, typically from cluster/manual.yaml.

    Notes:
    - num_gpus is **per-node** GPU count (int), or a per-node list for explicit node_ids binding.
    - component keys like train/eval/env/buffer are allowed as extra fields (strings).
    """

    name: str
    num_node: int
    num_gpus: Union[int, List[int]]
    node_ids: Optional[List[str]] = None


class ClusterConfig(Config):
    """
    Cluster configuration.
    """

    # ray
    ray_address: str = "auto"
    """ The address of the Ray cluster, auto means connecting to an existing cluster"""

    class PlacementConfig(Config):
        """
        Placement configuration for cluster resources.

        This is the single source of truth for placement behavior:
        - mode: auto/manual
        - strategy: default/disaggregate/colocate
        - env_strategy: default/device-colocate
        """

        mode: Literal["auto", "manual"] = "auto"
        strategy: Literal["default", "disaggregate", "colocate"] = "default"
        env_strategy: Literal["default", "device-colocate"] = "default"

    # worker numbers
    train_worker_num: int = 1
    """ Number of training workers for train policy"""
    eval_worker_num: int = 1
    """ Number of evaluation workers for eval policy """
    train_each_gpu_num: float = 1.0
    """ Number of GPU per policy worker for training """
    eval_each_gpu_num: float = 1.0
    """ Number of GPU per policy worker for evaluation """
    buffer_worker_num: Union[int, Literal["auto"]] = 1
    """ Number of storage workers for data buffer, auto means automatically determined"""

    # remote settings commonly designed for debugging
    remote_train: bool = True
    """ Whether to run the train policy as remote actor """
    remote_eval: bool = True
    """ Whether to run the eval policy as remote actor """
    remote_storage: bool = True
    """ Whether to run the data buffer as remote actor """
    remote_env: bool = True
    """ Whether to run the environment as remote actor """

    # colocated mode related
    is_colocated: bool = False
    """ Whether the train and eval policy is colocated """
    enable_offload: bool = False
    """ Whether to enable offload for the policy and env """
    rollout_env_interaction: Literal["batched", "streaming", None] = None
    """ The rollout environment interaction mode, batched or streaming """

    # resources scheduling optimization
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    """ Placement configuration for cluster resources """

    # manual mode resource pools (sibling to placement in YAML)
    resource_pool: Optional[List[ResourcePoolConfig]] = None
    """ Manual resource pool configuration list """



class MainConfig(Config):
    """
    Entry configuration class.
    """

    engine: Literal["asyncrl", "async_rsl", "rsl", "syncrl", "eval"] | None = None
    """Registered engine"""

    env: List[EnvConfig] | EnvConfig
    """ List of environment configurations """

    buffer: BufferConfig
    """ Configuration for the data buffer """

    policy: PolicyConfig
    """ Configuration for the policy """

    train: TrainConfig
    """ Configuration for the training process """

    # framework related
    cluster: ClusterConfig | None = None
    """ Configuration for the cluster """

    log: LogConfig = Field(default_factory=LogConfig)
    """ Configuration for logging """

    debug: bool = True
    """ debug mode """
    verbose: bool = True
    """ verbose mode """

    @model_validator(mode="after")
    def convert_single_env_cfg_to_list(self) -> "MainConfig":
        """Convert single environment configuration to a list."""
        if isinstance(self.env, EnvConfig):
            self.env = [self.env]
        return self


def validate_config_for_placement(config: MainConfig) -> MainConfig:
    """Validate the configuration for placement strategy."""
    from rlightning.utils.placement import get_global_resource_manager

    grm = get_global_resource_manager()
    if grm is not None:
        placement_strategy = grm.get_placement_strategy()
    else:
        placement_strategy = "default"

    if placement_strategy != "default":
        # The actual resource requirements are determined by the placement strategy; here we simply set it to 0.1
        config.cluster.train_each_gpu_num = 0.1
        config.cluster.eval_each_gpu_num = 0.1
        for env_cfg in config.env:
            env_cfg.num_gpus = 0.1
            env_cfg.num_cpus = 1

        if placement_strategy == "colocate":
            config.cluster.is_colocated = True
        elif placement_strategy == "disaggregate":
            config.cluster.is_colocated = False

    if config.cluster.is_colocated:
        config.policy.weight_buffer.buffer_strategy = "None"

    # Persist a snapshot of the overridden config.
    # This file makes post-hoc debugging easier when validate_config() mutates fields.
    try:
        from pathlib import Path

        from hydra.core.hydra_config import HydraConfig

        hydra_out_dir = Path(HydraConfig.get().runtime.output_dir)
        hydra_cfg_dir = hydra_out_dir / ".hydra"
        hydra_cfg_dir.mkdir(parents=True, exist_ok=True)
        (hydra_cfg_dir / "config.validated.yaml").write_text(config.to_yaml(), encoding="utf-8")
    except Exception:
        # Non-Hydra runs (e.g. unit tests) should not fail due to config saving.
        pass

    return config
