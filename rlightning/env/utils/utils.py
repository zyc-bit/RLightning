"""Environment preprocessing utilities."""

from typing import Union

import numpy as np
import torch

from rlightning.types import PolicyResponse


def default_env_preprocess_fn(policy_resp: PolicyResponse) -> Union[np.ndarray, torch.Tensor]:
    """
    Default preprocess function for environment step.
    Args:
        policy_resp (PolicyResponse): The response from the policy, containing the action to take.
    Returns:
        The action to be taken in the environment.
    """
    return policy_resp.action
