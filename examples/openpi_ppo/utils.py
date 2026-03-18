"""Utilities for OpenPI PPO preprocessing and postprocessing."""

from collections import defaultdict
from typing import Any, Dict, List, Optional

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
    # convert (T+1) steps to T steps
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
        elif k in next_keys:
            episode[k] = raw_episode[k]
        else:
            episode[k] = raw_episode[k][:-1]

    # convert_trajectories_to_batch
    batch_episode = {}
    non_tensor_keys = {}
    forward_inputs = {}
    for k, v in list(episode.items()):
        if k in non_tensor_keys:
            continue
        if "forward_inputs" in k:
            forward_inputs[k.replace("forward_inputs/", "")] = torch.stack(v)
        else:
            batch_episode[k] = torch.stack(v)
    batch_episode["forward_inputs"] = forward_inputs

    # compute loss mask
    batch_episode["dones"] = torch.logical_or(batch_episode["terminated"], batch_episode["truncated"])
    loss_mask, loss_mask_sum = compute_loss_mask(batch_episode["dones"])
    if policy_cfg.ppo_cfg.reward_type == "chunk_level":
        loss_mask = loss_mask.any(dim=-1, keepdim=True)
        loss_mask_sum = loss_mask_sum[..., -1:]

    # preprocess_advantages_inputs
    kwargs = {
        "adv_type": policy_cfg.ppo_cfg.adv_type,
        "rewards": batch_episode["rewards"],
        "values": batch_episode["values"],
        "bootstrap_values": batch_episode["bootstrap_values"],
        "dones": batch_episode["dones"],
        "gamma": policy_cfg.ppo_cfg.get("gamma", 1),
        "gae_lambda": policy_cfg.ppo_cfg.get("gae_lambda", 1),
        "reward_type": policy_cfg.ppo_cfg.reward_type,
        "loss_mask": loss_mask,
        "loss_mask_sum": loss_mask_sum,
    }
    kwargs = preprocess_embodied_advantages_inputs(**kwargs)

    # compute_gae_advantages_and_returns
    advantages, returns = compute_gae_advantages_and_returns(**kwargs)
    advantages_and_returns = postprocess_embodied_advantages_outputs(advantages=advantages, returns=returns, **kwargs)
    batch_episode.update(advantages_and_returns)
    # remove last value
    batch_episode["values"] = batch_episode["values"][:-1]

    return batch_episode


def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    bootstrap_values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and advantages for PPO using GAE."""
    # Add bootstrap values to rewards for truncated steps
    # Note: masks will be 0 for truncated steps in the GAE calculation below,
    # so we add gamma * bootstrap_value to the reward to account for the terminal value.
    if bootstrap_values is not None:
        rewards += gamma * bootstrap_values

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = rewards[step] + gamma * values[step + 1] * (~dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (~dones[step]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


def safe_normalize(array: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Normalize array values using a boolean loss mask."""
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    bootstrap_values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    if kwargs["reward_type"] == "chunk_level":
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        # bootstrap_values not support chunk_level
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    # flattened_dones_full = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
    # dones = flattened_dones_full[-(n_steps + 1) :]
    dones = dones.transpose(1, 2).reshape(n_steps, bsz)

    if bootstrap_values is not None:
        bootstrap_values = bootstrap_values.transpose(1, 2).reshape(n_steps, bsz)

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "bootstrap_values": bootstrap_values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def compute_loss_mask(dones):
    """Compute loss mask from dones. Expects dones shape [T, B, C] (already cropped to T steps)."""
    n_chunk_step, actual_bsz, num_action_chunks = dones.shape
    n_steps = n_chunk_step * num_action_chunks

    # [T, B, C] -> [T, C, B] -> [n_steps, B]
    flattened_dones = dones.transpose(1, 2).reshape(n_steps, actual_bsz)
    # Include step i if no done has occurred in steps 0..i
    flattened_loss_mask = flattened_dones.cumsum(dim=0) == 0  # [n_steps, actual_bsz]

    # [n_steps, B] -> [T, C, B] -> [T, B, C]
    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(1, 2)  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum
