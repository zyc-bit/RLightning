"""Registry class for component registration and discovery.

This module provides the Registry class for dynamically registering and
retrieving classes or functions by name, enabling plugin-style architectures.
"""

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """A registry for mapping string names to classes or functions.

    Provides a decorator-based API for registering components and
    retrieving them by name at runtime.

    Attributes:
        name: The name of this registry.
        module_dict: Dictionary mapping names to registered items.

    Example:
        >>> MODELS = Registry("models")
        >>> @MODELS.register("my_model")
        ... class MyModel:
        ...     pass
        >>> model_cls = MODELS.get("my_model")
    """

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: The name of the registry.
        """
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    @property
    def name(self) -> str:
        """Get the registry name."""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Any]:
        """Get the dictionary of registered modules."""
        return self._module_dict

    def get(self, key: str) -> Any:
        """Get a registered class or function by name.

        Args:
            key: The name of the registered item.

        Returns:
            The registered class or function.

        Raises:
            KeyError: If the key is not found in the registry.
        """
        if key not in self._module_dict:
            raise KeyError(f"{key} is not in the {self.name} registry")
        return self._module_dict[key]

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Register a module with the registry.

        A decorator for registering classes or functions.

        Args:
            name: The name to register the module with.
                If None, the module's __name__ will be used.

        Returns:
            Decorator function that registers and returns the class.

        Raises:
            KeyError: If the name is already registered.
        """

        def decorator(cls: T) -> T:
            """The decorator function."""
            module_name = name if name is not None else cls.__name__
            if module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self.name} registry.")
            self._module_dict[module_name] = cls
            return cls

        return decorator
