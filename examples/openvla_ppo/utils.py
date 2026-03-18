"""Utilities for OpenVLA PPO preprocessing and postprocessing."""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from rlightning.types import EnvRet, PolicyResponse
from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


def env_preprocess_fn(policy_resp: PolicyResponse) -> np.ndarray:
    """Preprocess PolicyResponse for Env."""
    raw_action_np = policy_resp.actions
    return raw_action_np


def env_ret_preprocess_fn(transition_dict: defaultdict[str, List], env_ret: EnvRet, *args, **kwargs) -> Dict[str, List]:
    """Aggregate env_ret fields for transitions."""
    # episode_buffer["obs_image"].append(env_ret.observation["images"])
    transition_dict["last_rewards"] = env_ret.last_reward
    transition_dict["last_truncated"] = env_ret.last_truncated
    transition_dict["last_terminated"] = env_ret.last_terminated
    return transition_dict


def policy_resp_preprocess_fn(transition_dict: defaultdict[str, List], policy_resp: PolicyResponse):
    """Aggregate policy response fields into the transition dict."""
    if not isinstance(policy_resp, PolicyResponse):
        raise TypeError(f"policy_resp must be an instance of PolicyResponse, got {type(policy_resp)}")

    policy_resp_dict = policy_resp.to_dict()

    for key, value in policy_resp_dict.items():
        if key == "actions":
            continue
        elif key == "forward_inputs":
            for sub_k, sub_v in value.items():
                transition_dict[f"forward_inputs/{sub_k}"] = sub_v
            continue
        transition_dict[key] = value

    return transition_dict


def episode_postprocess_fn(policy_cfg: Any, raw_episode: Dict[str, List]) -> Dict[str, Any]:
    """postprocess raw episode and compute PPO returns."""
    episode: Dict[str, Any] = {}
    last_keys = [
        "last_rewards",
        "last_truncated",
        "last_terminated",
        "bootstrap_values",
    ]
    next_keys = ["values"]

    for k in raw_episode.keys():
        if k in last_keys:
            current_k = k.replace("last_", "")
            episode[current_k] = raw_episode[k][1:]
        else:
            episode[k] = raw_episode[k][:-1]

        if k in next_keys:
            next_k = f"next_{k}"
            episode[next_k] = raw_episode[k][1:]

    batch_episode = {}
    non_tensor_keys = {}
    forward_inputs = {}
    for k, v in list(episode.items()):
        if k in non_tensor_keys:
            continue
        try:
            if "forward_inputs" in k:
                forward_inputs[k.replace("forward_inputs/", "")] = torch.stack(v)
            else:
                batch_episode[k] = torch.stack(v)
        except Exception:
            logger.exception(f"Could not stack key '{k}'. Leaving as list.")
    batch_episode["forward_inputs"] = forward_inputs
    batch_episode = compute_returns_ppo(batch_episode, ppo_cfg=policy_cfg.ppo_cfg)

    return batch_episode


def compute_returns_ppo(
    episode: Dict[str, torch.Tensor],
    ppo_cfg: Any,
) -> Dict[str, Any]:
    """Compute returns and advantages for PPO using GAE."""
    gamma = getattr(ppo_cfg, "gamma", 0.99)
    gae_lambda = getattr(ppo_cfg, "gae_lambda", 0.95)

    rewards = episode["rewards"]
    values = episode["values"]
    next_values = episode["next_values"]
    # Combine terminated and truncated
    dones = torch.logical_or(episode["terminated"], episode["truncated"])
    masks = (~dones).float()

    # Add bootstrap values to rewards for truncated steps
    # Note: masks will be 0 for truncated steps in the GAE calculation below,
    # so we add gamma * bootstrap_value to the reward to account for the terminal value.
    if "bootstrap_values" in episode:
        rewards += gamma * episode["bootstrap_values"]

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    for step in reversed(range(T)):
        delta = rewards[step] + gamma * next_values[step] * masks[step] - values[step]
        gae = delta + gamma * gae_lambda * masks[step] * gae
        returns[step] = gae + values[step]

    # calc adv
    advantages = returns - values
    mean_advantages = advantages.mean()
    std_advantages = advantages.std()

    episode["returns"] = returns
    episode["advantages"] = (advantages - mean_advantages) / (std_advantages + 1e-5)
    return episode


def safe_normalize(array: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Normalize array values using a boolean loss mask."""
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
