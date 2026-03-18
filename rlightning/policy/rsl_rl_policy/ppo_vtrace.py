"""PPO policy variant using V-trace value estimation.

Overrides rsl_rl's PPO to compute value targets and advantages with V-trace corrections.
"""

from typing import Any, Dict

import torch

from rlightning.policy.utils import vtrace

# Try to import rsl_rl components; if unavailable, define placeholders.
try:
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage.storage import Dataset
    from rsl_rl.utils.benchmarkable import Benchmarkable
    from rsl_rl.utils.recurrency import (
        trajectories_to_transitions,
        transitions_to_trajectories,
    )

    HAS_RSL_RL = True
except ImportError:
    HAS_RSL_RL = False
    PPO = object
    Dataset = Any

    class Benchmarkable:
        @staticmethod
        def register(func):
            return func


class PPOVtrace(PPO):

    def __init__(
        self,
        env,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        **kwargs,
    ):
        if not HAS_RSL_RL:
            raise ImportError("PPOVtrace requires 'rsl_rl' to be installed. " "Please install it to use this policy.")
        super().__init__(env, **kwargs)
        self._rho_bar = rho_bar
        self._c_bar = c_bar

        self._register_serializable("_rho_bar", "_c_bar")

    @Benchmarkable.register
    def _process_dataset(self, dataset: Any) -> Any:
        """Override PPO dataset processing to apply V-trace value estimation."""

        rewards = torch.stack([entry["rewards"] for entry in dataset])
        dones = torch.stack([entry["dones"] for entry in dataset])
        values = torch.stack([entry["values"] for entry in dataset])

        if hasattr(dataset[0], "next_values"):
            next_values = torch.stack([entry["values"] for entry in dataset])
        else:
            critic_kwargs = (
                {"hidden_state": (dataset[-1]["critic_state_h"], dataset[-1]["critic_state_c"])}
                if self.recurrent
                else {}
            )
            final_values = self.critic.forward(dataset[-1]["next_critic_observations"], **critic_kwargs)
            next_values = torch.cat((values[1:], final_values.unsqueeze(0)), dim=0)

        behavior_logp = torch.stack([entry["actions_logp"] for entry in dataset])
        actions = torch.stack([entry["actions"] for entry in dataset])
        actor_obs = torch.stack([entry["actor_observations"] for entry in dataset])

        if self.recurrent:
            transition_obs = actor_obs.reshape(*actor_obs.shape[:2], -1)
            actor_state_h = torch.stack([entry["actor_state_h"] for entry in dataset])
            actor_state_c = torch.stack([entry["actor_state_c"] for entry in dataset])
            observations, data = transitions_to_trajectories(transition_obs, dones)
            hidden_state_h, _ = transitions_to_trajectories(actor_state_h, dones)
            hidden_state_c, _ = transitions_to_trajectories(actor_state_c, dones)
            hidden_state = (hidden_state_h[0].transpose(0, 1), hidden_state_c[0].transpose(0, 1))
            action_mean, action_std = self.actor.forward(observations, hidden_state=hidden_state, compute_std=True)

            action_mean = action_mean.reshape(*observations.shape[:-1], self._action_size)
            action_std = action_std.reshape(*observations.shape[:-1], self._action_size)

            action_mean = trajectories_to_transitions(action_mean, data)
            action_std = trajectories_to_transitions(action_std, data)
        else:
            action_mean, action_std = self.actor.forward(actor_obs.flatten(0, 1), compute_std=True)

        dist = torch.distributions.Normal(action_mean, action_std)
        current_logp = dist.log_prob(actions.flatten(0, 1)).sum(-1).reshape(actor_obs.shape[:2])
        log_rhos = current_logp - behavior_logp

        if "timeouts" in dataset[0]:
            timeouts = torch.stack([entry["timeouts"] for entry in dataset])
            rewards += self.gamma * timeouts * values

        vs, advantages = vtrace.vtrace_correction(
            rewards, values, next_values, dones, log_rhos, self.gamma, self._rho_bar, self._c_bar
        )

        amean, astd = advantages.mean(), torch.nan_to_num(advantages.std())

        for step in range(len(dataset)):
            dataset[step]["target_value"] = vs[step]
            dataset[step]["advantages"] = advantages[step]
            dataset[step]["normalized_advantages"] = (advantages[step] - amean) / (astd + 1e-8)
        return dataset

    @Benchmarkable.register
    def _compute_value_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.recurrent:
            observations, data = transitions_to_trajectories(batch["critic_observations"], batch["dones"])
            hidden_state_h, _ = transitions_to_trajectories(batch["critic_state_h"], batch["dones"])
            hidden_state_c, _ = transitions_to_trajectories(batch["critic_state_c"], batch["dones"])
            hidden_states = (hidden_state_h[0].transpose(0, 1), hidden_state_c[0].transpose(0, 1))

            trajectory_evaluations = self.critic.forward(observations, hidden_state=hidden_states)
            trajectory_evaluations = trajectory_evaluations.reshape(*observations.shape[:-1])

            evaluation = trajectories_to_transitions(trajectory_evaluations, data)
        else:
            evaluation = self.critic.forward(batch["critic_observations"])

        value_clipped = batch["values"] + (evaluation - batch["values"]).clamp(-self._clip_ratio, self._clip_ratio)
        returns = batch["target_value"]
        value_losses = (evaluation - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)

        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        return value_loss
