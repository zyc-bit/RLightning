from typing import Any, Dict, Optional, Type

import ray

from rlightning.utils.placement.placement_manager import get_global_resource_manager


def launch_ray_actor(
    cls: Type, *args, role_type: str, worker_index: int, options: Optional[Dict[str, Any]] = None, **kwargs
) -> Any:
    """
    Create a Ray actor with the given options and arguments.

    Args:
        cls: The class to instantiate. Can be a regular class or a ray.remote wrapped class.
        *args: Positional arguments for the class constructor.
        role_type: The role type of the actor.
        worker_index: The worker index of the actor.
        options: A dictionary of options to pass to ray.remote_cls.options(**options).
        **kwargs: Keyword arguments for the class constructor.

    Returns:
        The created Ray actor handle.
    """
    if options is None:
        options = {}

    global_resource_manager = get_global_resource_manager()
    if global_resource_manager is not None:
        strategy = global_resource_manager.get_scheduling_strategy(role_type, worker_index)
        if strategy != "DEFAULT":
            options["scheduling_strategy"] = strategy

    # If cls is not already a remote function/class, make it one
    if not hasattr(cls, "remote"):
        remote_cls = ray.remote(cls)
    else:
        remote_cls = cls

    if role_type == "env":
        kwargs["worker_index"] = worker_index
    elif role_type == "train" or role_type == "eval":
        kwargs["role_type"] = role_type
    else:
        raise ValueError(f"Invalid role type: {role_type}")
    return remote_cls.options(**options).remote(*args, **kwargs)
