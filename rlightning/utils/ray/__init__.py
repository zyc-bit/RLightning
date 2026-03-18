"""Ray utilities module for distributed computing.

This module provides utilities for working with Ray actors and tasks,
including task submission, object resolution, and actor mixins.

Available components:
    - RayActorMixin: Mixin class for Ray actor functionality.
    - TaskSubmitter: Task submission utility with pending task management.
    - resolve_object: Utility to resolve Ray ObjectRefs to values.
"""

from .remote_class import RayActorMixin
from .task_submitter import TaskSubmitter
from .utils import resolve_object

__all__ = ["RayActorMixin", "TaskSubmitter", "resolve_object"]
