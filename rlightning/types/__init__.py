"""Types module for reinforcement learning data structures.

This module provides core data types used throughout the RLightning framework
for representing environment returns, policy responses, and batched data.

Available types:
    - BatchedData: Container for batched environment or policy data.
    - EnvRet: Environment return data structure.
    - MultiAgentEnvRet: Multi-agent environment return.
    - PolicyResponse: Policy action response structure.
    - MultiAgentPolicyResponse: Multi-agent policy response.
    - EnvMeta: Environment metadata container.
"""

from .batched_data import BatchedData
from .env_rets import EnvRet, MultiAgentEnvRet, Processed_EnvRet_fields
from .metadata import EnvMeta
from .policy_response import MultiAgentPolicyResponse, PolicyResponse

__all__ = [
    "BatchedData",
    "EnvRet",
    "MultiAgentEnvRet",
    "PolicyResponse",
    "MultiAgentPolicyResponse",
    "Processed_EnvRet_fields",
    "EnvMeta",
]
