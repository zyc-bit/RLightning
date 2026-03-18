import functools
import operator
import os
from pathlib import Path
from typing import Dict

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces as gym_spaces
from tensordict import TensorDict
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from rlightning.models.vision import NatureCNN
from rlightning.policy.base_policy import BasePolicy
from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.logger import get_logger
from rlightning.utils.profiler import profiler
from rlightning.utils.registry import POLICIES

logger = get_logger(__name__)


@POLICIES.register("SimplePPOPolicy")
class SimplePPOPolicy(BasePolicy):
    """A simple PPO Policy with a CNN encoder for testing purposes. it takes
    the actor-critic architecture.
    """

    def construct_network(self, env_meta, *args, **kwargs):
        action_space = env_meta.action_space

        self.is_discrete_action = False

        if isinstance(action_space, ray.ObjectRef):
            action_space = ray.get(action_space)
        if isinstance(action_space, gym_spaces.Discrete):
            action_dim = action_space.n
            self.is_discrete_action = True
        elif isinstance(action_space, gym_spaces.Box):
            # use flatten strategy

            action_dim = functools.reduce(operator.mul, action_space.shape)
        else:
            raise RuntimeError(f"unsupported action space! {type(action_space)}")

        self.encoder = NatureCNN(image_format="HWC")
        self.actor_mean = nn.Linear(self.encoder.out_feature_dim, action_dim)
        self.critic = nn.Linear(self.encoder.out_feature_dim, 1)
        if not self.is_discrete_action:
            self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        else:
            self.actor_logstd = None

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.actor_mean.cuda()
            self.critic.cuda()

            if self.actor_logstd is not None:
                self.actor_logstd.cuda()

    def setup_optimizer(self, optim_cfg):
        parameters = (
            list(self.encoder.parameters()) + list(self.actor_mean.parameters()) + list(self.critic.parameters())
        )
        if self.actor_logstd is not None:
            parameters.append(self.actor_logstd)
        self.optimizer = AdamW(parameters, lr=optim_cfg.lr)

    def get_action(self, obs: torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        x = self.encoder(obs)

        if self.is_discrete_action:
            logits = self.actor_mean(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = self.actor_mean(x)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, logprob, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value

        Args:
            obs (torch.Tensor): Batched observation array

        Returns:
            torch.Tensor: Batched state value
        """

        x = self.encoder(obs)
        return self.critic(x).squeeze(-1)

    def evaluate(self, obs, action):
        obs, action = obs.cuda(), action.cuda()
        x = self.encoder(obs)

        if self.is_discrete_action:
            # For discrete actions
            logits = self.actor_mean(x)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.critic(x).squeeze(-1)
        else:
            # For continuous actions
            mean = self.actor_mean(x)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            value = self.critic(x).squeeze(-1)

        return log_prob, entropy, value

    def get_action_value(self, obs):
        action, logp, ent = self.get_action(obs)
        value = self.get_value(obs)

        return action, logp, ent, value

    def postprocess(self, data):
        raise NotImplementedError

    def update_dataset(self, data):
        """
        Update the dataset in the policy by getting a batch from the buffer.
        """
        self.data = TensorDict.from_dict(data, auto_batch_size=True, device="cuda")

    @profiler.timer_wrap(level="info")
    def train(self):
        data = self.data

        batch_obs = data["observation"]
        batch_actions = data["action"]
        batch_logprobs = data["log_prob"]
        values = data["value"]
        batch_advantages = data["advantages"]
        batch_returns = data["returns"]

        ppo_epochs = self.config.train_config.ppo_epochs
        # batch_size = self.config.train_config.batch_size
        batch_size = len(data)
        minibatch_size = self.config.train_config.minibatch_size
        clip_coef = self.config.train_config.clip_ratio
        entropy_coef = self.config.train_config.entropy_coef
        for _ in range(ppo_epochs):
            idxs = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_idx = idxs[start:end]
                mb_obs = batch_obs[minibatch_idx]
                mb_actions = batch_actions[minibatch_idx]
                mb_old_logprobs = batch_logprobs[minibatch_idx]
                mb_advantages = batch_advantages[minibatch_idx]
                mb_returns = batch_returns[minibatch_idx]

                logprobs, entropy, values = self.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(logprobs - mb_old_logprobs)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optimizer.step()

    @profiler.timer_wrap(level="debug")
    def rollout_step(self, env_ret: EnvRet) -> PolicyResponse:
        observation = env_ret.observation

        obs_tensor = observation.float().unsqueeze(0).cuda()
        action, log_prob, entropy, value = self.get_action_value(obs_tensor)

        return PolicyResponse(
            env_id=env_ret.env_id,
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    def get_trainable_parameters(self):
        state_dict = {}
        for name, model in self.model_list:
            if isinstance(model, DDP):
                module = model.module
            else:
                module = model
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name, model in self.model_list:
            model.load_state_dict(state_dict[name], strict=strict)

    def save_weights(self, save_dir: str, epoch: int):
        save_path = Path(save_dir) / f"epoch_{epoch}"
        os.makedirs(save_path, exist_ok=True)

        state: Dict[str, Dict] = {}
        for name, model in self.model_list:
            if isinstance(model, DDP):
                module = model.module
            else:
                module = model
            state[name] = module.state_dict()

        # Save actor_logstd if present (it is an nn.Parameter, not a module)
        if getattr(self, "actor_logstd", None) is not None:
            state["actor_logstd"] = self.actor_logstd.detach().cpu()

        # Save optimizer state if available
        if getattr(self, "optimizer", None) is not None:
            state["optimizer"] = self.optimizer.state_dict()

        ckpt_path = os.path.join(save_path, f"simple_ppo.pt")
        torch.save(state, ckpt_path)
