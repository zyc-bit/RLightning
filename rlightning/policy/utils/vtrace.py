from typing import Tuple

import torch


@torch.inference_mode()
def batch_step_correction(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    log_rhos: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
):
    """
    Compute a single-step V-trace style correction in batch form.

    Args:
        rewards: Reward tensor.
        values: Value estimates at current states.
        next_values: Value estimates at next states.
        log_rhos: Log importance weights for behavior vs. target policy.
        dones: Done flags indicating terminal transitions.
        gamma: Discount factor.
        rho_bar: Clipping threshold for importance weights.
        c_bar: Clipping threshold for trace coefficients.

    Returns:
        target_values: Corrected value targets.
        advantages: Policy gradient advantages.
    """
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    deltas = clipped_rhos * (rewards + gamma * (1.0 - dones.float()) * next_values - values)

    vs = values + deltas
    target_values = values + deltas + gamma * cs * (vs - next_values)
    advantages = clipped_rhos * (rewards + gamma * target_values - values)

    return target_values, advantages


@torch.inference_mode()
def vtrace_correction(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    # bootstrap_value: torch.Tensor,
    log_rhos: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace corrections for off-policy learning.

    Args:
        rewards (torch.Tensor): Rewards of shape [T].
        values (torch.Tensor): Value estimates of shape [T].
        next_values (torch.Tensor): Next-state value estimates of shape [T].
        dones (torch.Tensor): Done flags of shape [T].
        log_rhos (torch.Tensor): Log importance weights of shape [T].
        gamma (float): Discount factor.
        rho_bar (float): Threshold for importance weight clipping.
        c_bar (float): Threshold for trace coefficient clipping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: V-trace targets and policy gradient advantages.
    """

    # Clip importance weights
    dones = dones.bool()
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Append bootstrap value
    # values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)])

    # Compute temporal differences
    deltas = clipped_rhos * (rewards + gamma * next_values - values)

    # Compute V-trace values iteratively (backward pass)
    vs = torch.zeros_like(values)
    vs[-1] = values[-1] + deltas[-1]

    for t in reversed(range(len(rewards) - 1)):
        vs_t_plus_1_or_zero = torch.where(dones[t], 0.0, vs[t + 1])
        # vs_t_plus_1_or_zero = 0.0 if dones[t] else vs[t + 1]
        vs[t] = values[t] + deltas[t] + gamma * cs[t] * (vs_t_plus_1_or_zero - next_values[t])

    # Policy gradient advantages
    pg_advantages = clipped_rhos[:-1] * (rewards[:-1] + gamma * vs[1:] - values[:-1])

    # For the last timestep
    last_advantage = clipped_rhos[-1] * (rewards[-1] + gamma * next_values[-1] - values[-1])
    pg_advantages = torch.cat([pg_advantages, last_advantage.unsqueeze(0)])

    return vs, pg_advantages
