import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Type

import torch
import torch.nn as nn
from tensordict import TensorDict

from rlightning.env import EnvMeta
from rlightning.policy.base_policy import BasePolicy, PolicyRole
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.config import Config, PolicyConfig, TrainConfig
from rlightning.utils.logger import get_logger
from rlightning.utils.registry import POLICIES

from .ppo_vtrace import PPOVtrace

logger = get_logger(__name__)


@dataclass
class RSLRLVecEnvMeta(EnvMeta):
    num_actions: int = None
    get_observations: Callable = None


@POLICIES.register("RSLRLPolicy")
class RSLRLPolicy(BasePolicy):

    _algorithm_keys = ["DDPG", "D4PG", "DPPO", "DSAC", "PPO", "SAC", "TD3", "PPOVtrace"]

    def __init__(self, config: PolicyConfig, role_type: PolicyRole):
        """Initialize policy and resolve the requested rsl_rl algorithm."""
        super().__init__(config, role_type)

        try:
            from rsl_rl import algorithms

            # register PPOVtrace here
            algorithms.PPOVtrace = PPOVtrace
        except ImportError:
            logger.error("cannot import algorithm from rsl_rl, please ensure you've installed this third-party package")

        policy_kwargs = config.policy_kwargs.to_dict()
        policy_kwargs["device"] = self.device
        algorithm = policy_kwargs.pop("algorithm")

        if "custom" in policy_kwargs:
            self.custom_policy_kwargs = policy_kwargs.pop("custom")
        else:
            self.custom_policy_kwargs = dict()

        assert algorithm in self._algorithm_keys, (
            algorithm,
            self._algorithm_keys,
        )

        self.algorithm = algorithm
        self.algo_cls: Type[algorithms.Agent] = getattr(algorithms, algorithm)
        self.policy_kwargs = policy_kwargs
        self.algo: algorithms.Agent = None
        self.dataset_buffer = deque()

    def construct_network(self, model_config: Config = None, env_meta: EnvMeta = None, *args, **kwargs):
        """Create RSLRL algorithm instance, and register rsl_rl modules to the policy.

        Args:
            model_config (_type_, optional): Model config given for initialization. Defaults to None.
            env_meta (RSLRLVecEnvMeta, optional): Environment configuration. Defaults to None.
            *args: Additional positional arguments kept for interface compatibility.
            **kwargs: Additional keyword arguments kept for interface compatibility.
        """
        self.algo = self.algo_cls(env_meta, **self.policy_kwargs)
        self.algo.to(self.device)

        for k, v in self.algo.__dict__.items():
            name = f"rsl_rl/{k}"
            if name in self._modules or name in self._parameters:
                continue

            if isinstance(v, nn.Module):
                logger.info(f"Detected nn.Module: {name}, registering to policy.")
                self.model_list.append((f"algo.{k}", v))
                self.register_module(name, v)
            elif isinstance(v, nn.Parameter):
                logger.info(f"Detected nn.Parameter: {name}, registering to policy.")
                self.register_parameter(name, v)

    def setup_optimizer(self, optim_cfg):
        """reconstruct optimizer for rsl_rl modules."""

        if optim_cfg is not None:
            Warning.warn(
                "RSLRLPolicy's optimizer is determined by rsl_rl algorithm, the given optim_cfg will be ignored."
            )

        optimizer_cls = type(self.algo.optimizer)
        self.algo.optimizer = optimizer_cls(self.algo.parameters(), **self.algo.optimizer.defaults)

    def get_trainable_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a dict of module state dicts."""

        state_dict = {}

        for name, module in self._modules.items():
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Dict[str, torch.Tensor]]):
        """Load module state dicts by module name with warnings on mismatches."""

        updated_modules = set()
        for module_name, module_params in state_dict.items():
            if module_name not in self._modules:
                logger.warning(f"Module {module_name} not found in policy modules.")
                continue
            updated_modules.add(module_name)
            module: nn.Module = self._modules[module_name]

            module.load_state_dict(module_params, strict=True)

        missed_modules = set(self._modules.keys()) - updated_modules
        if len(missed_modules) > 0:
            logger.warning(
                f"{len(missed_modules)} modules were not found in the provided state_dict. Details: {missed_modules}"
            )

    def init_eval(self, eval_config=None, env_meta: EnvMeta = None):
        """Initialize evaluation mode, threading/async state, and weight updater."""

        self.eval_config = eval_config

        if self.config.model_cfg is not None:
            logger.warning(
                "RSLRLPolicy does not support model configuration with `model_cfg`, please use `policy_kwargs` instead."
            )

        self.construct_network(None, env_meta=env_meta)
        self._find_model()

        self.algo.eval_mode()

        if self.rollout_mode == "sync":
            self._idle_as_infer = threading.Event()
            self._idle_as_infer.set()
        elif self.rollout_mode == "async":
            self._loop = asyncio.get_event_loop()
            # admission control gate for new requests
            self._accept_new_requests = asyncio.Event()
            self._accept_new_requests.set()
            # track in-flight requests and provide a condition to wait for drain
            self.num_requests = 0
            self._num_requests_lock = threading.Lock()
            self._inflight_zero_cv = threading.Condition(self._num_requests_lock)
        else:
            raise ValueError(f"Invalid rollout mode: {self.rollout_mode}")

        if self.weight_buffer_strategy != "None":
            self._update_weights_signal = threading.Event()
            self._update_weights_signal.clear()
            self._weight_update_done = threading.Event()
            self._weight_update_done.set()  # initially "done" — no update in progress
            # always try to update weights for eval policy
            updater_thread = threading.Thread(target=self.update_weights, daemon=True)
            updater_thread.start()
        self.is_init = True

    def init_train(self, train_config: TrainConfig, env_meta=None):
        """Initialize training mode with rsl_rl algorithm."""

        super().init_train(train_config, env_meta=env_meta)
        if self.config.model_cfg is not None:
            logger.warning(
                "RSLRLPolicy does not support model configuration with `model_cfg`, please use `policy_kwargs` instead."
            )

        self.algo.train_mode()

    def update_dataset(self, data: TensorDict | List[TensorDict]):
        """Assign the given dataset to policy for training."""

        if isinstance(data, TensorDict):
            dataset = data.to(self.device).permute(dims=[1, 0])
        else:
            dataset = TensorDict.from_dict(data, auto_batch_size=True, device=self.device).permute(dims=[1, 0])
            # dataset = TensorDict.stack(data, dim=0).to(self.device).permute(dims=[1, 0])

        dataset["normalized_advantages"] = torch.zeros_like(dataset["rewards"], device=self.device)
        dataset["advantages"] = torch.zeros_like(dataset["rewards"], device=self.device)
        self.dataset_buffer.append(dataset)
        return True

    def postprocess(self, env_ret: EnvRet, policy_resp: PolicyResponse) -> PolicyResponse:
        """Convert env transition to rsl_rl format and attach value targets."""
        env_ret = env_ret.cuda()

        data = self.algo.process_transition(
            env_ret._extra["last_observation"],
            env_ret._extra["last_info"],
            policy_resp.action,
            env_ret.last_reward,
            env_ret.observation,
            env_ret.info,
            env_ret.last_terminated | env_ret.last_truncated,
            {k: v for k, v in policy_resp.to_dict().items() if k != "action"},
        )

        if "values" in data:
            data["next_values"] = self.algo.critic.forward(data["next_critic_observations"]).detach()
        self.algo.register_terminations(data["dones"].nonzero().reshape(-1))

        # actually not just policy's response, but for keeping compatible interface we still use it
        return PolicyResponse(env_id=env_ret.env_id, **data)

    def rollout_step(self, env_ret: EnvRet) -> PolicyResponse:
        """Sample action from policy or random policy before initialization."""

        obs = env_ret.observation

        if self.algo.initialized:
            actions, data = self.algo.draw_actions(obs, env_ret.info)
        else:
            actions, data = self.algo.draw_random_actions(obs, env_ret.info)

        return PolicyResponse(env_id=env_ret.env_id, action=actions, **data)

    def train(self):
        """Pop a dataset from the buffer and run a single rsl_rl update."""

        if len(self.dataset_buffer) > 0:
            dataset = self.dataset_buffer.popleft()
        else:
            raise RuntimeError("No dataset available for RSL_RL policy training.")

        # update RSL_RL inner dataset with given dataset
        dataset = [a.to_dict() for a in dataset]

        loss = self.algo.update(dataset)

        # add loss prefix
        loss = {f"training/{k}": v for k, v in loss.items()}
        return loss
