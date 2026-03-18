"""Task submission utilities for Ray actors.

This module provides the TaskSubmitter class for managing task submission
to Ray actors with pending task limits.
"""

import threading
from typing import Any, Callable, Dict, List, Union

import ray

from .utils import resolve_object


class TaskSubmitter:
    """Task submitter for local and remote Ray actor methods.

    Manages task submission to either local class functions or remote Ray actors,
    ensuring that the number of pending tasks does not exceed a specified limit.

    This class is thread-safe for concurrent task submissions.

    Attributes:
        max_pending_tasks: Maximum number of pending tasks allowed per actor.
        worker_ref_to_pending_tasks: Mapping from actor handles to pending task refs.
    """

    def __init__(self, max_pending_tasks_each_worker: int = 1) -> None:
        """Initialize the TaskSubmitter.

        Args:
            max_pending_tasks_each_worker: Maximum number of pending tasks
                allowed per Ray actor.

        Raises:
            ValueError: If max_pending_tasks_each_worker is negative.
        """
        self.max_pending_tasks = max_pending_tasks_each_worker
        self.worker_ref_to_pending_tasks: Dict[ray.actor.ActorHandle, List[ray.ObjectRef]] = {}
        self._lock = threading.Lock()

        if not max_pending_tasks_each_worker >= 0:
            raise ValueError("max_pending_tasks_each_worker must be non-negative value")

    def submit(self, method: Callable, *args: Any, _block: bool = False, **kwargs: Any) -> Union[Any, ray.ObjectRef]:
        """Submit a task to a local method or Ray actor.

        This method is thread-safe for concurrent submissions.

        Args:
            method: The method to be called (local or ray.actor.ActorMethod).
            *args: Positional arguments for the method.
            _block: If True, block until the remote task completes and
                return the actual result instead of a future.
            **kwargs: Keyword arguments for the method.

        Returns:
            For local methods or when _block=True: the actual result.
            For remote methods with _block=False: a ray.ObjectRef.

        Raises:
            ValueError: If _block is not a boolean value.
        """
        if not isinstance(_block, bool):
            raise ValueError(
                f"_block must be a boolean value, but got {_block} as a {type(_block)} type. You "
                f"should also check whether you are passing too many arguments to the submitted "
                f"function {method}, more than the expected."
            )

        if not isinstance(method, ray.actor.ActorMethod):
            return method(*resolve_object(args), **resolve_object(kwargs))

        worker_ref = method._actor_ref

        # Thread-safe access to pending task tracking
        with self._lock:
            pending_task_ref_list = self.worker_ref_to_pending_tasks.get(worker_ref, [])
            # query the unfinished tasks
            if len(pending_task_ref_list) > 0:
                _, pending_task_ref_list = ray.wait(pending_task_ref_list, timeout=0)

            if len(pending_task_ref_list) > self.max_pending_tasks - 1:
                _, pending_task_ref_list = ray.wait(
                    pending_task_ref_list,
                    num_returns=len(pending_task_ref_list) - (self.max_pending_tasks - 1),
                )

            ref = method.remote(*args, **kwargs)

            if not _block:
                pending_task_ref_list.append(ref)
                self.worker_ref_to_pending_tasks[worker_ref] = pending_task_ref_list

        # ray.get() outside lock to avoid blocking other threads
        if _block:
            return ray.get(ref)

        return ref
