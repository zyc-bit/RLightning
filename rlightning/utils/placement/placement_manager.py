"""
Global Resource Manager for resource-aware component scheduling.

This module provides a singleton GlobalResourceManager that:
1. Reads global configuration and generates ComponentScheduling
2. Discovers and manages global node resources
3. Plans resource pools based on placement strategy (disaggregate/colocate)
4. Creates Ray placement groups for components
5. Tracks component distribution across nodes
"""

import os
import pprint
from typing import Any, Dict, List, Optional

import yaml

from rlightning.utils.config.config import ClusterConfig, Config
from rlightning.utils.logger import get_logger
from rlightning.utils.placement.placement_strategies import (
    PLACEMENT_STRATEGIES,
    PlacementStrategy,
    ResourcePoolPlacementStrategy,
)
from rlightning.utils.placement.resource_pool import ResourcePool, ResourcePoolPlanner
from rlightning.utils.placement.scheduling import ComponentScheduling
from rlightning.utils.ray.utils import get_cluster_actor_info, get_cluster_resources

logger = get_logger(__name__)


class GlobalResourceManager:
    """
    Singleton resource manager for global resource management.

    This manager orchestrates the following workflow:
    1. Initialize with configuration and scheduling requirements
    2. Discover cluster resources via ResourcePoolPlanner
    3. Plan resource pools based on placement strategy
    4. Create placement groups via the selected strategy
    5. Track component distribution for monitoring

    Usage:
        manager = GlobalResourceManager.get_instance()
        manager.initialize(placement_config, scheduling)
        strategy = manager.get_scheduling_strategy("train", worker_index=0)
    """

    _instance: Optional["GlobalResourceManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "GlobalResourceManager":
        """Create or return the singleton instance.

        Returns:
            The singleton GlobalResourceManager instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "GlobalResourceManager":
        """Get the singleton instance..

        Returns:
            The singleton GlobalResourceManager instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the placement manager."""
        if not self._initialized:
            self._strategy: Optional[PlacementStrategy] = None
            self._placement_config: Optional[Config] = None
            self._scheduling: Optional[ComponentScheduling] = None
            self._initialized = True
            self._cluster_info: Dict[str, Any] = {}
            self._resource_pools: Dict[str, ResourcePool] = {}
            self._component_distribution: Dict[str, Dict[str, Dict[str, Any]]] = {}
            self._resource_planner: Optional[ResourcePoolPlanner] = None

    @property
    def is_initialized(self) -> bool:
        """Check if the placement manager is initialized with a strategy..

        Returns:
            True if a strategy has been set.
        """
        return self._strategy is not None

    def initialize(
        self,
        cluster_cfg: ClusterConfig,
        scheduling: ComponentScheduling,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the placement manager with configuration.

        This method performs the following steps:
        1. Store configuration and scheduling requirements
        2. Discover cluster resources
        3. Validate scheduling requirements against available resources
        4. Plan resource pools based on placement strategy
        5. Create placement groups via the selected strategy
        6. Track initial component distribution

        Args:
            cluster_cfg: Cluster configuration, including placement settings.
            scheduling: Component scheduling requirements.
            config_path: Optional path to the configuration file.
        """
        self._placement_config = cluster_cfg.placement
        self._config_path = config_path
        self._cluster_info = get_cluster_resources()
        self._scheduling = scheduling

        # Step 1: Determine placement strategy
        placement_mode = str(self._placement_config.mode).lower()
        placement_strategy = self.get_placement_strategy()
        strategy_class = PLACEMENT_STRATEGIES[placement_strategy]
        self._strategy = strategy_class(scheduling=scheduling)

        logger.info(
            f"[GlobalResourceManager] Initializing with mode: {placement_mode}, " f"strategy: {placement_strategy}"
        )
        if not isinstance(self._strategy, ResourcePoolPlacementStrategy):
            self._scheduling.infer_auto_buffer_worker_num(self._cluster_info)
            self._strategy.create_placement_groups()
            return

        # Step 2: Create resource planner and discover cluster resources
        self._resource_planner = ResourcePoolPlanner(scheduling=self._scheduling, cluster_info=self._cluster_info)
        self._resource_planner.discover_cluster_resources()

        # Step 3: Plan resource pools
        if placement_mode == "manual":
            # Manual mode: load resource pools from config
            assert cluster_cfg.resource_pool is not None, "resource_pool must be provided in manual mode"
            pools_cfg_list = cluster_cfg.resource_pool
            if not isinstance(pools_cfg_list, list):
                pools_cfg_list = [pools_cfg_list]
            self._resource_pools = self._resource_planner.load_manual_resource_pools(pools_cfg_list)
        else:
            self._resource_pools = self._resource_planner.plan_resource_pools(
                strategy=placement_strategy,
                env_strategy=self._placement_config.env_strategy,
            )

        # Log resource planning summary
        summary = self._resource_planner.summary()
        logger.debug(f"[GlobalResourceManager] Resource planning summary: {pprint.pformat(summary)}")

        # Before creating placement groups, adjust the buffer worker number
        # to match the train worker number.
        train_node_count = self._resource_planner.get_component_node_count("train")
        self._scheduling.adjust_buffer_worker_num(train_node_count)

        # Step 4: Create placement strategy and placement groups
        placement_groups = self._strategy.create_placement_groups(resource_pools=self._resource_pools)
        self._component_distribution = self._strategy.get_node_component_distribution()
        logger.info(f"[GlobalResourceManager] Component distribution: {self._component_distribution}")
        logger.info(
            f"[GlobalResourceManager] Created {len(placement_groups)} placement groups: "
            f"{list(placement_groups.keys())}"
        )

        # Save resource pool yaml config
        if placement_mode == "auto" and self._config_path is not None:
            self.save_yaml_config(cluster_config_path=self._config_path)

    def get_placement_config(self) -> Optional[Config]:
        """Get the placement configuration.

        Returns:
            The placement configuration object.
        """
        return self._placement_config

    def get_scheduling(self) -> Optional[ComponentScheduling]:
        """Get the component scheduling requirements.

        Returns:
            The component scheduling object.
        """
        return self._scheduling

    def get_scheduling_strategy(self, component_type: str, worker_index: int = 0) -> Any:
        """
        Get Ray scheduling strategy for a component.

        Args:
            component_type: Type of component ("train", "eval", "buffer", "env").
            worker_index: Index of the worker within its type.

        Returns:
            Ray scheduling strategy (PlacementGroupSchedulingStrategy or "DEFAULT").
        """
        if self._strategy is None:
            raise RuntimeError("GlobalResourceManager not initialized. Call initialize() first.")

        strategy = self._strategy.get_scheduling_strategy(component_type, worker_index)
        strategy_info = strategy.__dict__ if not isinstance(strategy, str) else strategy
        logger.debug(
            f"[GlobalResourceManager] Get scheduling strategy for " f"{component_type}[{worker_index}]: {strategy_info}"
        )
        return strategy

    def get_storage_to_train_workers(self) -> Optional[Dict[int, List[int]]]:
        """Get storage -> train worker mapping from the active strategy.

        Returns:
            Dictionary mapping storage indices to train worker indices.

        Raises:
            RuntimeError: If PlacementManager is not initialized.
        """
        if self._strategy is None:
            raise RuntimeError("GlobalResourceManager not initialized. Call initialize() first.")
        return self._strategy.get_storage_to_train_workers()

    def get_resource_pools(self, pool_name: str = None) -> Dict[str, ResourcePool]:
        """Get the planned resource pool by pool_name."""
        return self._resource_planner.get_resource_pools(pool_name)

    def get_pool_for_component(self, component_type: str) -> Optional[ResourcePool]:
        """Get the resource pool containing a specific component type."""
        if self._resource_planner is None:
            return None
        return self._resource_planner.get_pool_for_component(component_type)

    def get_component_distribution(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Return mapping of node_id -> {component_type: {"count": N, "ids": [...]}}.

        Format is aligned with ResourcePoolPlacementStrategy._node_component_distribution.
        """
        if not self._component_distribution:
            logger.warning("[GlobalResourceManager] Component distribution not found.")
        #     self.refresh_component_distribution()
        return self._component_distribution

    def refresh_component_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Refresh and return the current component distribution across nodes.

        This queries Ray's actor registry to get real-time distribution.
        """
        try:
            node_actors = get_cluster_actor_info()
            self._component_distribution = {}

            for node_id, actor_names in node_actors.items():
                component_counts: Dict[str, int] = {}
                for name in actor_names:
                    # Parse component type from actor name
                    # Expected format: "policy-{train|eval}-{type}-{id}" or "buffer-{id}"
                    if "train" in name:
                        component_counts["train"] = component_counts.get("train", 0) + 1
                    elif "eval" in name:
                        component_counts["eval"] = component_counts.get("eval", 0) + 1
                    elif "storage" in name.lower():
                        component_counts["buffer"] = component_counts.get("buffer", 0) + 1
                    elif "env" in name.lower():
                        component_counts["env"] = component_counts.get("env", 0) + 1

                if component_counts:
                    self._component_distribution[node_id] = component_counts

            logger.info(f"[GlobalResourceManager] Component distribution: {self._component_distribution}")
        except Exception as e:
            logger.warning(f"[GlobalResourceManager] Failed to refresh component distribution: {e}")

        return self._component_distribution

    def get_placement_strategy(self) -> str:
        """Get the placement strategy."""
        if self._placement_config is None:
            return "default"
        placement_mode = str(self._placement_config.mode).lower()
        if placement_mode == "manual":
            return "resource_pool"
        return str(self._placement_config.strategy).lower()

    def get_resource_planner(self) -> Optional[ResourcePoolPlanner]:
        """Get the resource planner instance."""
        return self._resource_planner

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get a summary of cluster resources and allocation."""
        if self._resource_planner is None:
            return {"error": "GlobalResourceManager not initialized"}

        summary = self._resource_planner.summary()
        summary["placement_strategy"] = self.get_placement_strategy()
        summary["component_distribution"] = self._component_distribution
        return summary

    def save_yaml_config(
        self,
        cluster_config_path: str,
        filename: str = "resource_pool_auto.yaml",
        subdir: str = "resource_pool",
    ) -> str:
        """
        Save current `resource_pool` yaml to disk.
        """
        if self._resource_planner is None:
            raise RuntimeError("GlobalResourceManager not initialized")
        yaml_dict = self._resource_planner.to_yaml_config()

        out_dir = os.path.join(cluster_config_path, subdir)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_dict, f, sort_keys=False)

        logger.info(f"[ResourcePoolPlanner] Saved resource_pool yaml to: {out_path}")
        return out_path

    def cleanup(self) -> None:
        """Clean up placement groups."""
        if self._strategy is not None:
            self._strategy.cleanup()

    @property
    def strategy(self) -> Optional[PlacementStrategy]:
        """Get current placement strategy.

        Returns:
            The current PlacementStrategy instance or None.
        """
        return self._strategy

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        if cls._instance is not None:
            cls._instance.cleanup()
        cls._instance = None
        cls._initialized = False


def get_global_resource_manager() -> Optional[GlobalResourceManager]:
    """
    Get the initialized global resource manager singleton instance.

    Returns:
        GlobalResourceManager if initialized, None otherwise.
    """
    instance = GlobalResourceManager.get_instance()
    if instance.is_initialized:
        return instance
    return None
