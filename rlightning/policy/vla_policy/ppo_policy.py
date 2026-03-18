from collections import defaultdict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.optim as optim
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from rlightning.policy.base_policy import BasePolicy, PolicyRole
from rlightning.policy.utils.losses import compute_ppo_actor_critic_loss
from rlightning.policy.utils.utils import (
    append_to_dict,
    postprocess_loss_metric,
    preprocess_loss_inputs,
)
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.distributed.collective import all_reduce_dict
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import POLICIES
from rlightning.utils.utils import to_device

logger = get_logger(__name__)


@POLICIES.register("VLAPPOPolicy")
class VLAPPOPolicy(BasePolicy):
    def __init__(self, config, role_type, *args, **kwargs):
        super().__init__(config, role_type)
        # ppo parameters
        self.entropy_bonus = self.config.ppo_cfg.entropy_bonus
        self.clip_ratio = self.config.ppo_cfg.clip_ratio
        self.value_clip_ratio = self.config.ppo_cfg.value_clip_ratio
        self.huber_delta = self.config.ppo_cfg.huber_delta

        # optimizer parameters
        self.optimizer = None
        self.grad_clip_norm = self.config.optim_cfg.grad_clip_norm
        self.optimizer_steps = 0
        self.critic_warmup_steps = 0
        if self.config.optim_cfg.get("critic_warmup_steps", None) and self.config.model_cfg.get(
            "add_value_head", False
        ):
            critic_warmup_steps = getattr(self.config.optim_cfg, "critic_warmup_steps", 0)
            self.critic_warmup_steps = int(critic_warmup_steps)

        self._setup_sampling_params()

        try:
            torch.set_num_threads(int(getattr(self.config, "num_cpus", 1)) or 1)
        except Exception:
            pass

        self._is_ready = True

    def _setup_sampling_params(self):
        # length parameters for rollout
        self._length_params = self.config.length_params
        # sampling parameters for rollout
        self._sampling_params = self.config.sampling_params
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": True if self._sampling_params.get("temperature_eval", -1) > 0 else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def is_ready(self):
        return self._is_ready

    def construct_network(self, env_meta, *args, **kwargs):
        backend = kwargs.get("backend", self.config.backend.backend_name)
        assert (self.role_type != PolicyRole.TRAIN) or (
            backend == "transformers"
        ), "Only support transformers in train mode"
        if backend == "transformers":
            if self.config.model_cfg.model_name == "openpi":
                from rlightning.models.openpi.openpi_utils import get_openpi_model

                self.model = get_openpi_model(self.config.model_cfg, device=self.device)
            elif self.config.model_cfg.model_name == "openvla":
                from rlightning.models.openvla.openvla_model import OpenVLAModel

                self.model = OpenVLAModel(self.config.model_cfg, device=self.device)
            else:
                raise ValueError(f"Unsupported model: {self.config.model_cfg.model_name}")
        elif backend == "vllm":
            # TODO: support vllm
            raise ValueError(f"Unsupported backend: {backend}")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def setup_optimizer(self, optim_cfg):
        betas = (optim_cfg.adam_beta1, optim_cfg.adam_beta2)
        adam_eps = optim_cfg.get("adam_eps", 1.0e-08)
        weight_decay = optim_cfg.get("weight_decay", 1e-2)

        params_actor = []
        params_critic = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "value_head" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)

        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {
                    "params": params_actor,
                    "lr": optim_cfg.lr,
                    "betas": betas,
                }
            )
        if len(params_critic) > 0:
            param_groups.append(
                {
                    "params": params_critic,
                    "lr": optim_cfg.value_lr,
                    "betas": betas,
                }
            )
        self.optimizer = optim.AdamW(param_groups, eps=adam_eps, weight_decay=weight_decay)

    @property
    def _model(self) -> nn.Module:
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def optimizer_step(self):
        grad_norm = nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"] + self.optimizer.param_groups[1]["params"],
            self.grad_clip_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.optimizer_steps += 1
        if self.critic_warmup_steps > 0:
            if self.optimizer_steps >= self.critic_warmup_steps:
                # self.setup_optimizer(enable_warmup=False)
                self.critic_warmup_steps = -1
        return {"grad_norm": grad_norm.item()}

    @torch.inference_mode()
    @profiler.timer_wrap(level="debug")
    def rollout_step(self, env_ret: EnvRet, mode="train") -> PolicyResponse:
        kwargs = self._train_sampling_params if mode == "train" else self._eval_sampling_params

        if self.config.model_cfg.model_name in ["openpi"]:
            kwargs = {"mode": mode}

        env_obs = env_ret.observation
        actions, ppo_result = self.model.get_action(
            env_obs=env_obs,
            **kwargs,
        )

        bootstrap_values = torch.zeros_like(ppo_result["prev_values"], device=self.device)  # [bsz, ]
        #  Handle auto_reset: add bootstrap value ONLY for truncated episodes (not terminated)
        if env_ret.last_truncated.any() and env_ret.info.get("final_observation") is not None:
            if hasattr(self.model, "value_head"):
                final_obs = env_ret.info["final_observation"]
                _, final_results = self.model.get_action(
                    env_obs=final_obs,
                    **kwargs,
                )
                last_step_truncated = env_ret.last_truncated  # [bsz, ]
                # Add bootstrap value to the last step of truncated episodes
                bootstrap_values[last_step_truncated] = final_results["prev_values"][last_step_truncated]

        return PolicyResponse(
            env_id=env_ret.env_id,
            actions=actions,
            values=ppo_result["prev_values"],
            logprobs=ppo_result["prev_logprobs"],
            forward_inputs=ppo_result["forward_inputs"],
            bootstrap_values=bootstrap_values,
        )

    def update_dataset(self, data):
        """
        Update the dataset in the policy by getting a batch from the buffer.
        """
        data = to_device(data, "cpu")
        self.dataset = TensorDict.from_dict(data, auto_batch_size=True, device="cpu")

    @profiler.timer_wrap(level="info")
    def train(self):
        assert self.role_type == PolicyRole.TRAIN, "Only train role can call fit."
        batch_size = len(self.dataset)
        logger.info(f"Received {batch_size} data from Replay Buffer")

        mini_batch_size_per_rank = self.config.train_config.mini_batch_size // self.world_size
        if batch_size % mini_batch_size_per_rank != 0:
            logger.warning(
                f"received batch_size {batch_size} is not divisible by mini_batch_size_per_rank {mini_batch_size_per_rank}. "
                "The remaining data will be discarded."
            )
        num_mini_batch = batch_size // mini_batch_size_per_rank
        assert num_mini_batch > 0, "data size is too small to sample a batch of size {mini_batch_size_per_rank}"

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size_per_rank : (i + 1) * mini_batch_size_per_rank] for i in range(num_mini_batch)
        ]

        metrics = defaultdict(list)
        update_epoch = self.config.train_config.get("update_epoch", 1)
        for _ in range(update_epoch):
            for indicies in tqdm(sampler, desc="Training batch"):
                batch = self.dataset[indicies]
                batch_metrics = self.optimize(batch)
                for k, v in batch_metrics.items():
                    metrics[k].extend(v)
        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
        mean_metrics = all_reduce_dict(
            mean_metrics,
            comm_mode=ParallelMode.TRAIN_DATA_PARALLEL,
            op=torch.distributed.ReduceOp.AVG,
        )
        return mean_metrics

    def optimize(self, mini_batch):
        batch_metrics = defaultdict(lambda: [])
        micro_batch_size = self.config.train_config.micro_batch_size
        assert len(mini_batch) % micro_batch_size == 0, "micro_batch_size must be divisible by mini_batch_size"
        gradient_accum_steps = len(mini_batch) // micro_batch_size
        micro_batches = [
            mini_batch[i * micro_batch_size : (i + 1) * micro_batch_size] for i in range(gradient_accum_steps)
        ]
        for _, micro_batch in enumerate(micro_batches):
            metrics_data = self.compute_ppo_loss(micro_batch, gradient_accum_steps)
            append_to_dict(batch_metrics, metrics_data)

        optimizer_info = self.optimizer_step()

        append_to_dict(batch_metrics, optimizer_info)

        return batch_metrics

    def compute_ppo_loss(self, batch, gradient_accum_steps=1):
        batch = batch.to(self.device)

        forward_inputs = batch["forward_inputs"]

        kwargs = {}
        if self.config.model_cfg.model_name == "openvla":
            kwargs["temperature"] = self._sampling_params["temperature_train"]
            kwargs["top_k"] = self._sampling_params["top_k"]

        compute_values = True if self.config.ppo_cfg.adv_type == "gae" else False

        # Policy loss
        output_dict = self.model(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=self.entropy_bonus > 0,
            compute_values=compute_values,
            **kwargs,
        )

        kwargs = {
            "loss_type": self.config.ppo_cfg.loss_type,
            "logprob_type": self.config.ppo_cfg.logprob_type,
            "reward_type": self.config.ppo_cfg.reward_type,
            "single_action_dim": self.config.model_cfg.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": batch["logprobs"],
            "advantages": batch["advantages"],
            "returns": batch.get("returns", None),
            "prev_values": batch.get("values", None),
            "clip_ratio_low": self.clip_ratio,
            "clip_ratio_high": self.clip_ratio,
            "value_clip": self.value_clip_ratio,
            "huber_delta": self.huber_delta,
            "loss_mask": batch.get("loss_mask", None),
            "loss_mask_sum": batch.get("loss_mask_sum", None),
            "max_episode_steps": batch.get("max_episode_steps", 80),
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }

        kwargs = preprocess_loss_inputs(**kwargs)

        loss, metrics_data = compute_ppo_actor_critic_loss(**kwargs)

        metrics_data = postprocess_loss_metric(metrics_data)

        # entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        # metrics_data["entropy_loss"] = entropy_loss.detach().item()
        loss /= gradient_accum_steps
        loss.backward()
        metrics_data["loss"] = loss.detach().item()
        return metrics_data

    def get_trainable_parameters(self):
        state_dict = {}
        for k, v in self._model.named_parameters():
            # Only include parameters that require gradients
            if v.requires_grad:
                state_dict[k] = v.detach()
        return {"model": state_dict}

    def load_state_dict(self, state_dict, *, trainable_only: bool = True):
        """Load parameters into the underlying model.

        Args:
            state_dict (dict): A dict with key "model" that maps to the parameters
                to be loaded.
            trainable_only (bool, optional): If ``True``, assumes ``state_dict``
                only contains the trainable parameters and will load them with
                ``strict=False`` so that non-specified parameters remain
                unchanged. Defaults to ``True``.
        """
        strict = not trainable_only
        self.model.load_state_dict(state_dict["model"], strict=strict)

    def save_weights(self, save_dir: str, epoch: int):
        path = Path(save_dir) / f"epoch_{epoch}"
        self._model.save(path)

    @torch.inference_mode()
    def postprocess(
        self, data: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Post processing given data"""
        raise NotImplementedError
