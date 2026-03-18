"""Default preprocessing utilities for buffer transitions."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from rlightning.types import EnvRet, PolicyResponse, Processed_EnvRet_fields

from .preprocessors import (
    Preprocessor,
    default_obs_preprocessor,
    default_reward_preprocessor,
)


def default_env_ret_preprocess_fn(
    transition_dict: Dict[str, Any],
    env_ret: EnvRet,
    obs_preprocessor: Preprocessor,
    reward_preprocessor: Preprocessor,
) -> Dict[str, Any]:
    """Populate a transition dict from an EnvRet."""
    if not isinstance(env_ret, EnvRet):
        raise TypeError(f"env_ret must be an instance of EnvRet, got {type(env_ret)}")

    env_ret_dict = env_ret.to_dict()
    # preprocess
    env_ret_dict["observation"] = obs_preprocessor(env_ret.observation)
    env_ret_dict["last_reward"] = reward_preprocessor(env_ret.last_reward)

    for key, value in env_ret_dict.items():
        transition_dict[key] = value

    return transition_dict


def default_policy_resp_preprocess_fn(
    transition_dict: Dict[str, Any],
    policy_resp: PolicyResponse,
) -> Dict[str, Any]:
    """Populate a transition dict from a PolicyResponse."""
    if not isinstance(policy_resp, PolicyResponse):
        raise TypeError(f"policy_resp must be an instance of PolicyResponse, got {type(policy_resp)}")

    policy_resp_dict = policy_resp.to_dict()

    for key, value in policy_resp_dict.items():
        transition_dict[key] = value

    return transition_dict


def default_preprocess_fn(
    transition_dict: Dict[str, Any],
    env_ret: Optional[EnvRet] = None,
    policy_resp: Optional[PolicyResponse] = None,
    obs_preprocessor: Optional[Preprocessor] = default_obs_preprocessor,
    reward_preprocessor: Optional[Preprocessor] = default_reward_preprocessor,
    env_ret_preprocess_fn: Optional[Callable] = default_env_ret_preprocess_fn,
    policy_resp_preprocess_fn: Optional[Callable] = default_policy_resp_preprocess_fn,
) -> Dict[str, Any]:
    """
    Default transition preprocess function. It will use the given obs_preprocessor and
    reward_preprocessor to preprocess both env_ret and policy_resp, or either one of them.
    When adding transition in a sync manner (env_ret and policy_resp are paired in one step of
    rollout), both env_ret and policy_resp should be provided. When adding transition in an async
    manner, only one of them should be provided.

    Args:
        transition_dict: The dict to aggregate transition data from env_ret and policy_resp.
        env_ret (Optional[EnvRet]): The environment return to be preprocessed.
        policy_resp (Optional[PolicyResponse]): The policy response to be preprocessed.
        obs_preprocessor (Optional[Preprocessor]): The preprocessor for observations.
        reward_preprocessor (Optional[Preprocessor]): The preprocessor for rewards.
        env_ret_preprocess_fn (Optional[Callable]): Function to preprocess `env_ret`.
        policy_resp_preprocess_fn (Optional[Callable]): Function to preprocess `policy_resp`.

    Returns:
        The preprocessed transition dict.
    """
    if env_ret is None and policy_resp is None:
        raise ValueError("At least one of env_ret or policy_resp must be provided.")

    if env_ret is not None and policy_resp is not None:
        if env_ret.env_id != policy_resp.env_id:
            raise ValueError(
                f"Mismatched env_id in env_ret and policy_resp, got {env_ret.env_id} and " f"{policy_resp.env_id}"
            )

    if env_ret is not None:
        transition_dict = env_ret_preprocess_fn(transition_dict, env_ret, obs_preprocessor, reward_preprocessor)
    if policy_resp is not None:
        transition_dict = policy_resp_preprocess_fn(transition_dict, policy_resp)

    return transition_dict


def default_postprocess_fn(episode_buffer: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Convert an episode buffer into a flat training-ready dict."""
    data = {}
    for k, v in episode_buffer.items():
        # skip info by default
        if "info" in k:
            continue
        # process keys with "last_" prefix
        if isinstance(v[0], torch.Tensor):
            v = torch.stack(v, dim=0)
        else:
            v = torch.tensor(v)
        if k.startswith("last_"):
            _k = k[5:]
            _v = v[1:]  # support for both 1D and 2D arrays (vector env)
            data[_k] = _v
        else:
            _k, _v = k, v

        # process special keys
        if k == "observation":
            data["next_observation"] = v[1:]
            data["observation"] = v[:-1]

        # process policy_resp keys
        env_fields = set(EnvRet.fields() + Processed_EnvRet_fields)
        policy_fields = [k for k in episode_buffer.keys() if k not in env_fields]
        if k in policy_fields:
            data[k] = v[:-1]

    return data


def default_compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_adv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    This version uses an explicit loop over the batch dimension.

    Args:
        rewards (torch.Tensor): Rewards at each timestep
        values (torch.Tensor): Value function estimates at each timestep
        next_values (torch.Tensor): Value function estimates at the next timestep
        dones (torch.Tensor): Done flags at each timestep
        gamma (float): Discount factor
        lam (float): GAE lambda parameter
        normalize_adv (bool): Whether to normalize the advantages

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Computed advantages and returns
    """

    B, N = rewards.shape  # B=batch_size, N=num_envs
    device = rewards.device

    advantages = torch.zeros_like(rewards, device=device)
    last_advantage = torch.zeros(N, device=device)  # [num_envs,] each env has independent last_adv

    # iterate backwards over batch dimension
    for t in reversed(range(B)):
        # when done = True, (1 - dones[t])=0, cutting off the accumulation chain.
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        # compute advantages[t] for all envs in batch, 2nd term is 0 if done=True
        advantages[t] = delta + gamma * lam * (1.0 - dones[t]) * last_advantage
        # update last_advantage for next step in reverse order
        last_advantage = advantages[t]

    returns = advantages + values

    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def default_gae_no_loop(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_adv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    This version uses matrix operations to eliminate explicit loops for efficiency.

    Args:
        rewards (torch.Tensor): Rewards at each timestep
        values (torch.Tensor): Value function estimates at each timestep
        next_values (torch.Tensor): Value function estimates at the next timestep
        dones (torch.Tensor): Done flags at each timestep
        gamma (float): Discount factor
        lam (float): GAE lambda parameter
        normalize_adv (bool): Whether to normalize the advantages

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Computed advantages and returns
    """

    B, N = rewards.shape  # B=batch_size, N=num_envs
    device = rewards.device
    coeff = gamma * lam

    deltas = rewards + gamma * next_values * (1.0 - dones.float()) - values  # shape [B, N]
    discount = coeff * (1.0 - dones)  # [B, N]，done=True indicates discount=0

    # generate reversed cumulative product mask: cumprod from back to front,
    #   done=True makes all subsequent values 0.
    discount_mask = torch.cat([torch.ones(1, N, device=device), discount[:-1]], dim=0)
    discount_cum = torch.cumprod(discount_mask.flip(0), dim=0).flip(0)  # [B, N]

    # generate weight matrix and compute advantages
    weight = coeff ** torch.arange(B, device=device).view(B, 1)  # [B, 1]
    weight_matrix = weight.unsqueeze(1) * discount_cum  # [B, N]
    advantages = torch.matmul(deltas.T, weight_matrix).T  # [B, N]

    returns = advantages + values

    if normalize_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
