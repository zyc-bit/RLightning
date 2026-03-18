"""
Placement module for global resource management and component scheduling.

This module provides:
- GlobalResourceManager: Singleton manager for resource-aware component scheduling
- ResourcePoolPlanner: Plans resource pools based on placement strategy (auto/manual)
- ComponentScheduling: Defines resource requirements for each component type
- PlacementStrategy: Strategies for creating Ray placement groups
"""

from .placement_manager import GlobalResourceManager, get_global_resource_manager
from .placement_strategies import (
    PLACEMENT_STRATEGIES,
    ColocatedPlacementStrategy,
    DefaultPlacementStrategy,
    DisaggregatePlacementStrategy,
    PlacementStrategy,
)
from .resource_pool import (
    ComponentAllocation,
    NodeResource,
    ResourcePool,
    ResourcePoolPlanner,
)
from .scheduling import ComponentScheduling, Scheduling, setup_component_scheduling

# Alias for backward compatibility
setup_cluster_scheduling = setup_component_scheduling

__all__ = [
    # Manager
    "GlobalResourceManager",
    "get_global_resource_manager",
    # Resource Pool
    "ComponentAllocation",
    "NodeResource",
    "ResourcePool",
    "ResourcePoolPlanner",
    # Scheduling
    "ComponentScheduling",
    "Scheduling",
    "setup_component_scheduling",
    "setup_cluster_scheduling",
    # Strategies
    "PLACEMENT_STRATEGIES",
    "PlacementStrategy",
    "DefaultPlacementStrategy",
    "DisaggregatePlacementStrategy",
    "ColocatedPlacementStrategy",
]
