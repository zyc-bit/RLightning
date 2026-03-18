"""
Placement strategies for component scheduling.

This module provides different strategies for placing components on cluster resources:
- DefaultPlacementStrategy: No specific placement, uses Ray's default scheduling
- ResourcePoolPlacementStrategy: Base class for resource pool-based placement
- DisaggregatePlacementStrategy: Separate pools for Train+Buffer and Eval+Env
- ColocatedPlacementStrategy: Shared pool for all components
"""

import pprint
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rlightning.utils.logger import get_logger
from rlightning.utils.placement.resource_pool import ResourcePool
from rlightning.utils.placement.scheduling import ComponentScheduling
from rlightning.utils.ray.utils import get_cluster_resources

logger = get_logger(__name__)

# Component types that use CPU-only bundles (no GPU)
CPU_ONLY_COMPONENTS = {"buffer"}


def _pack_workers_on_gpu_units(
    *,
    allocations: List[Tuple[int, int]],
    node_id: str,
    component_type: str,
    node_component_distribution: Dict[str, Dict[str, Dict[str, Any]]],
    pg_key: str,
    capacity_gpus: int,
    unit_cpu: List[int],
    worker_locations: List[tuple],
    workers_placed: int,
    workers_total: int,
    gpu_req_list: List[float],
    cpu_req: int,
) -> int:
    """
    Expand allocations[component_name] to GPU unit list, then pack workers on GPU=1 bundles.

    - If allocations is empty, default to full [0..capacity-1].
    - `gpu_req_list` is per-worker requirement list (0 < gpu <= 1).
      For constant req, pass `[req] * total`.

    Returns:
        updated_workers_placed count.
    """
    if not allocations:
        return workers_placed

    unit_remaining = [1.0 for _ in range(capacity_gpus)]

    units: List[int] = []
    for s, e in allocations:
        assert s <= e, f"Invalid allocation: {s} <= {e}"
        units.extend(list(range(int(s), int(e) + 1)))

    for u in units:
        if u < 0 or u >= capacity_gpus:
            raise RuntimeError(f"Invalid GPU unit index: {u} for capacity: {capacity_gpus}")
        while workers_placed < workers_total and unit_remaining[u] >= float(gpu_req_list[workers_placed]):
            component_id = workers_placed
            worker_locations.append((pg_key, u))
            node_component_distribution.setdefault(node_id, {})
            comp_entry = node_component_distribution[node_id].setdefault(component_type, {"count": 0, "ids": []})
            comp_entry["ids"].append(component_id)
            comp_entry["count"] += 1
            unit_remaining[u] -= float(gpu_req_list[component_id])
            unit_cpu[u] += int(cpu_req)
            workers_placed += 1

    return workers_placed


class PlacementStrategy(ABC):
    """Abstract base class for placement strategies.

    Defines the interface for creating Ray placement groups and
    determining scheduling strategies for different component types.

    Attributes:
        scheduling: Cluster scheduling configuration.
        placement_groups: Created placement groups by name.
        storage_to_train_workers: Mapping from storage index to train worker indices.
    """

    def __init__(self, scheduling: ComponentScheduling):
        """Initialize the placement strategy.

        Args:
            scheduling: Component scheduling configuration.
        """
        self.scheduling: ComponentScheduling = scheduling
        self.placement_groups: Dict[str, PlacementGroup] = {}
        self._storage_to_train_workers: Dict[int, List[int]] = {}

    @abstractmethod
    def create_placement_groups(self) -> Dict[str, PlacementGroup]:
        """Create placement groups based on the strategy.

        Returns:
            Dictionary mapping group names to PlacementGroup instances.
        """
        pass

    @abstractmethod
    def get_scheduling_strategy(self, component_type: str, worker_index: int = 0) -> Any:
        """Get scheduling strategy for a specific component.

        Args:
            component_type: Type of component ('env', 'train', 'eval', 'buffer').
            worker_index: Index of the worker within its type.

        Returns:
            Scheduling strategy (PlacementGroupSchedulingStrategy or 'DEFAULT').
        """
        pass

    def get_storage_to_train_workers(self) -> Dict[int, List[int]]:
        """Return storage -> train worker mapping (may be empty if not used).

        Returns:
            Dictionary mapping storage indices to lists of train worker indices.
        """
        return self._storage_to_train_workers

    def cleanup(self) -> None:
        """Clean up placement groups."""
        for pg in self.placement_groups.values():
            try:
                ray.util.remove_placement_group(pg)
            except Exception as e:
                logger.warning(f"Failed to remove placement group: {e}")
        self.placement_groups.clear()

    def _print_placement_group_details(self, pg: PlacementGroup, group_name: str) -> None:
        """Print placement group details for debugging.

        Args:
            pg: Placement group to print details for.
            group_name: Name of the placement group.
        """
        try:
            logger.debug(f"[{group_name}] Placement group details:")
            logger.debug(pprint.pformat(ray.util.placement_group_table(pg)))
        except Exception as error:
            logger.error(f"Failed to get placement group details: {error}")


class DefaultPlacementStrategy(PlacementStrategy):
    """Default placement strategy with no specific placement groups.

    Uses node affinity for buffer workers when multiple are needed,
    but otherwise relies on Ray's default scheduling.

    Attributes:
        buffer_strategys: List of scheduling strategies for buffer workers.
        _node_info: Node information for resource-aware choices.
    """

    def __init__(self, scheduling: ComponentScheduling):
        """Initialize the default placement strategy.

        Args:
            scheduling: Component scheduling configuration.
        """
        super().__init__(scheduling)
        self.buffer_strategies: List[Any] = []
        # Cache node info so derived strategies can make resource-aware choices
        self._node_info = get_cluster_resources()

    def create_placement_groups(self) -> Dict[str, PlacementGroup]:
        """Create placement groups based on the strategy.

        For multiple buffer storages, creates node affinity strategies
        to place each buffer on a different node.

        Returns:
            Empty dictionary (no placement groups created).
        """
        num_buffer_storages = self.scheduling.buffer_worker.worker_num
        if num_buffer_storages > 1:
            node_ids = list(self._node_info["node_id_to_resources"].keys())
            assert len(node_ids) >= num_buffer_storages, (
                f"Not enough nodes to place buffer storages, required: {num_buffer_storages}, "
                f"available: {len(node_ids)}."
            )
            for storage_index in range(num_buffer_storages):
                node_id = node_ids[storage_index]
                self.buffer_strategies.append(
                    ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    )
                )
        return {}

    def get_scheduling_strategy(self, component_type: str, worker_index: int = 0) -> Any:
        """Get scheduling strategy for a component.

        Args:
            component_type: Type of component.
            worker_index: Index of the worker.

        Returns:
            Node affinity strategy for buffers, 'DEFAULT' otherwise.
        """
        if component_type == "buffer" and len(self.buffer_strategies) > 0:
            return self.buffer_strategies[worker_index]
        return "DEFAULT"


class ResourcePoolPlacementStrategy(PlacementStrategy):
    """
    Base class for resource pool-based placement strategies.

    Provides common functionality for creating placement groups from resource pools
    by reading component_types from pool config and scheduling info from self.scheduling.
    """

    def __init__(self, scheduling: ComponentScheduling):
        super().__init__(scheduling)
        # Maps component_type -> list of (pg_key, bundle_index) tuples
        self._worker_locations: Dict[str, List[tuple]] = {
            "train": [],
            "buffer": [],
            "eval": [],
            "env": [],
        }
        # component_type -> placed worker count (global across pools)
        self._workers_placed: Dict[str, int] = {}
        # component_type -> total worker count (global)
        self._workers_total: Dict[str, int] = {}
        # node_id -> {component_type: {"count": int, "ids": [int, ...]}}
        self._node_component_distribution: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _ensure_component_tracking(self, component_types: List[str]) -> None:
        for comp_type in component_types:
            if comp_type not in self._workers_total:
                total, _, _ = self._get_component_scheduling(comp_type)
                self._workers_total[comp_type] = total
            self._workers_placed.setdefault(comp_type, 0)

    def _get_component_scheduling(self, component_type: str) -> Tuple[int, float | List[float], int]:
        """
        Get scheduling info for a component type from self.scheduling.

        Returns:
            Tuple of (worker_total, num_gpus, num_cpus).
            num_gpus can be a single float or a list for heterogeneous workers (env).
        """
        scheduling_dict = self.scheduling.to_dict()
        if component_type == "env":
            # env_worker is a list of Scheduling, build per-worker GPU list
            worker_total = 0
            num_gpus_list: List[float] = []
            num_cpus = 1  # default
            for env_sched in self.scheduling.env_worker:
                worker_total += env_sched.worker_num
                num_gpus_list.extend([env_sched.num_gpus] * env_sched.worker_num)
            return worker_total, num_gpus_list, num_cpus
        else:
            worker_sched = scheduling_dict[component_type]
            num_gpus_list = [worker_sched["num_gpus"]] * worker_sched["worker_num"]
            return worker_sched["worker_num"], num_gpus_list, worker_sched["num_cpus"]

    def _create_pool_placement_groups(
        self,
        pool_name: str,
        resource_pool: ResourcePool,
    ) -> None:
        """
        Create placement groups for a resource pool.

        This method reads component_types from pool_config and gets component
        scheduling info from self.scheduling automatically.

        Args:
            pool_name: Name prefix for placement groups.
            resource_pool: Resource pool object providing nodes and component types.
        """
        nodes = resource_pool.nodes
        if not nodes:
            logger.warning(f"[{pool_name}] No nodes specified in pool config")
            return
        component_types = resource_pool.component_types
        if not component_types:
            logger.warning(f"[{pool_name}] No component_types specified in pool config")
            return

        self._ensure_component_tracking(component_types)

        # Separate GPU components and CPU-only components
        gpu_components: List[str] = []
        for comp_type in component_types:
            if comp_type not in CPU_ONLY_COMPONENTS:
                gpu_components.append(comp_type)

        for node_info in nodes:
            node_id = node_info.node_id
            # Use total_gpus as allocated GPUs for this pool(maybe more than max_allocated_gpus)
            allocated_gpus = int(node_info.total_gpus)
            allocations = node_info.allocations

            assert allocated_gpus > 0, f"Node {node_id} has no allocated GPUs"
            pg_key = f"{pool_name}_{node_id}"

            # Build per-GPU-unit bundles (GPU=1 as unit)
            unit_cpu = [0 for _ in range(allocated_gpus)]
            bundles = [{"GPU": 1, "CPU": 0} for _ in range(allocated_gpus)]

            # Pack GPU components
            for comp_type in gpu_components:
                if comp_type not in allocations:
                    continue
                total, num_gpus_list, num_cpus = self._get_component_scheduling(comp_type)
                locations = self._worker_locations[comp_type]

                self._workers_placed[comp_type] = _pack_workers_on_gpu_units(
                    allocations=allocations[comp_type],
                    node_id=node_id,
                    component_type=comp_type,
                    node_component_distribution=self._node_component_distribution,
                    pg_key=pg_key,
                    capacity_gpus=allocated_gpus,
                    unit_cpu=unit_cpu,
                    worker_locations=locations,
                    workers_placed=self._workers_placed[comp_type],
                    workers_total=total,
                    gpu_req_list=num_gpus_list,
                    cpu_req=int(num_cpus),
                )

            # Finalize CPU per GPU-unit bundle
            for u in range(allocated_gpus):
                bundles[u]["CPU"] = unit_cpu[u]

            # Remove bundle with CPU=0 (no workers placed on this GPU unit)
            bundles = [b for b in bundles if b["CPU"] > 0]

            # Place CPU-only components (e.g., buffer) - separate bundle per worker
            if "train" in allocations and len(allocations["train"]) > 0:
                buffer_worker_id = self._workers_placed["buffer"]

                locations = self._worker_locations["buffer"]
                bundle_idx = len(bundles)
                bundles.append({"CPU": 1})
                locations.append((pg_key, bundle_idx))
                self._node_component_distribution.setdefault(node_id, {})
                comp_entry = self._node_component_distribution[node_id].setdefault("buffer", {"count": 0, "ids": []})
                comp_entry["ids"].append(buffer_worker_id)
                comp_entry["count"] += 1

                self._workers_placed["buffer"] += 1

                # Verify that the mapping from storage (buffer) workers to train workers is
                # evenly distributed among all buffer workers
                train_worker_ids = self._node_component_distribution[node_id]["train"]["ids"]
                self._storage_to_train_workers[buffer_worker_id] = train_worker_ids

            if bundles:
                pg = ray.util.placement_group(
                    bundles,
                    name=pg_key,
                    strategy="STRICT_PACK",
                    _soft_target_node_id=node_id,
                )
                ray.get(pg.ready())
                self.placement_groups[pg_key] = pg
                self._print_placement_group_details(pg, pg_key)

    def create_placement_groups(
        self, resource_pools: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs
    ) -> Dict[str, PlacementGroup]:
        """Create placement groups based on the strategy."""
        for pool_name, resource_pool in resource_pools.items():
            self._create_pool_placement_groups(pool_name, resource_pool)

        self._verify_workers_placed()

        return self.placement_groups

    def _verify_workers_placed(self) -> None:
        """Verify all workers placed."""
        for comp_type, total in self._workers_total.items():
            if self._workers_placed[comp_type] != total:
                raise RuntimeError(
                    f"{comp_type} workers placed number mismatch: " f"{self._workers_placed[comp_type]}/{total}"
                )

    def get_scheduling_strategy(self, component_type: str, worker_index: int = 0) -> Any:
        """Get scheduling strategy for a component."""
        locations = self._worker_locations.get(component_type, [])
        if worker_index < len(locations):
            pg_key, bundle_idx = locations[worker_index]
            if pg_key in self.placement_groups:
                return PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_groups[pg_key],
                    placement_group_bundle_index=bundle_idx,
                )
        return "DEFAULT"

    def get_node_component_distribution(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return the planned component distribution per node."""
        return {
            node_id: {comp: dict(info) for comp, info in counts.items()}
            for node_id, counts in self._node_component_distribution.items()
        }


class DisaggregatePlacementStrategy(ResourcePoolPlacementStrategy):
    """
    Disaggregate placement strategy.

    Allocates separate resource pools for:
    - train_pool: Train workers + Buffer workers (colocated)
    - rollout_pool: Eval workers + Env workers (colocated)

    This provides resource isolation between training and evaluation.
    """

    def create_placement_groups(
        self,
        resource_pools: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, PlacementGroup]:
        """
        Create placement groups based on resource pools.

        Args:
            resource_pools: Dictionary with "train_pool" and "rollout_pool" configurations.
            **kwargs: Additional strategy options kept for interface compatibility.
        """
        # Create placement groups for train pool
        if "train_pool" not in resource_pools:
            raise ValueError("train_pool not found in resource pools")
        train_pool = resource_pools["train_pool"]
        self._create_pool_placement_groups(pool_name="train_pool", resource_pool=train_pool)

        # Create placement groups for rollout pool
        if "rollout_pool" not in resource_pools:
            raise ValueError("rollout_pool not found in resource pools")
        rollout_pool = resource_pools["rollout_pool"]
        self._create_pool_placement_groups(pool_name="rollout_pool", resource_pool=rollout_pool)

        self._verify_workers_placed()

        logger.info(
            "[DisaggregatePlacementStrategy] Created placement groups:\n"
            f"  - Train worker locations: {len(self._worker_locations['train'])}\n"
            f"  - Buffer worker locations: {len(self._worker_locations['buffer'])}\n"
            f"  - Eval worker locations: {len(self._worker_locations['eval'])}\n"
            f"  - Env worker locations: {len(self._worker_locations['env'])}"
        )

        return self.placement_groups


class ColocatedPlacementStrategy(ResourcePoolPlacementStrategy):
    """
    Colocated placement strategy.

    All components share a global resource pool. Workers are distributed
    across nodes with consideration for resource utilization.
    """

    def create_placement_groups(
        self,
        resource_pools: Optional[Dict[str, Dict[str, Any]]] = None,
        max_colocate_count: int = 10,
        **kwargs,
    ) -> Dict[str, PlacementGroup]:
        """
        Create placement groups for colocated components.

        Args:
            resource_pools: Dictionary with "global_pool" configuration.
            max_colocate_count: Maximum number of components per node.
            **kwargs: Additional strategy options kept for interface compatibility.
        """
        global_pool = resource_pools["global_pool"]
        self._create_pool_placement_groups(pool_name="global_pool", resource_pool=global_pool)

        self._verify_workers_placed()

        logger.info(
            "[ColocatedPlacementStrategy] Created placement groups:\n"
            f"  - Train worker locations: {self._worker_locations['train']}\n"
            f"  - Buffer worker locations: {self._worker_locations['buffer']}\n"
            f"  - Eval worker locations: {self._worker_locations['eval']}\n"
            f"  - Env worker locations: {self._worker_locations['env']}"
        )

        return self.placement_groups


# Strategy registry
PLACEMENT_STRATEGIES = {
    "resource_pool": ResourcePoolPlacementStrategy,
    "colocate": ColocatedPlacementStrategy,
    "disaggregate": DisaggregatePlacementStrategy,
    "default": DefaultPlacementStrategy,
}
