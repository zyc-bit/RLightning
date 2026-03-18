from dataclasses import dataclass, fields
from typing import Optional, Tuple

from gymnasium import spaces


@dataclass
class EnvMeta:
    """Environment metadata container.

    Stores metadata about an environment including its spaces and
    configuration.

    Attributes:
        env_id: Unique identifier for the environment.
        action_space: Gymnasium action space.
        observation_space: Gymnasium observation space.
        num_envs: Number of parallel environments.
    """

    env_id: str = None
    """The environment ID."""
    action_space: Optional[spaces.Space] = None
    """Action space of the environment."""
    observation_space: Optional[spaces.Space] = None
    """Observation space of the environment."""
    num_envs: Optional[int] = None
    """The vectorized number of the environment."""

    @classmethod
    @property
    def _fields(cls) -> Tuple:
        """Get all field names.

        Returns:
            Tuple of field names.
        """
        return tuple(field.name for field in fields(cls))
