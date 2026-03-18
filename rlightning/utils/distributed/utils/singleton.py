"""Singleton metaclass implementation.

This module provides a metaclass for implementing the singleton pattern,
ensuring only one instance of a class exists throughout the application.
"""

from typing import Any, Dict


class SingletonMeta(type):
    """Metaclass that implements the singleton pattern.

    Classes using this metaclass will only have one instance created.
    Subsequent instantiation attempts return the existing instance.

    Attributes:
        _instances: Dictionary mapping classes to their singleton instances.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create or return the singleton instance.

        Args:
            *args: Positional arguments for instance creation.
            **kwargs: Keyword arguments for instance creation.

        Returns:
            The singleton instance of the class.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]
