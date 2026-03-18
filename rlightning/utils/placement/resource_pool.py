"""
Resource Pool Planner for global resource management.

This module provides functionality for:
1. Discovering and managing cluster node resources
2. Planning resource pools based on component scheduling requirements
3. Supporting different allocation strategies (disaggregate, colocate)
4. Tracking fine-grained component-to-resource mappings
"""

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from rlightning.utils.logger import get_logger
from rlightning.utils.placement.scheduling import ComponentScheduling
from rlightning.utils.ray.utils import get_cluster_resources

logger = get_logger(__name__)


@dataclasses.dataclass
class NodeResource:
    """
    Represents a node's resources.

    This class is used in two contexts:
    - **Cluster discovery**: total_* reflects node total capacity; available_* is computed.
    - **ResourcePool membership**: total_* represents the node's capacity,
      and `allocations` records per-component GPU index ranges within the node's index space.

    Resource tracking uses `gpu_cursor` to track the next available GPU index.
    `available_gpus` is computed as `total_gpus - gpu_cursor`.
    """

    node_id: str
    ip: str
    total_cpus: int
    total_gpus: int
    # component_type -> list of (start_idx, end_idx) in this node-local GPU index space
    allocations: Dict[str, List[Tuple[int, int]]] = dataclasses.field(default_factory=dict)
    # cursor for assigning node-local GPU indices [0, total_gpus)
    gpu_cursor: int = 0
    max_allocated_gpus: int = 0

    @property
    def available_gpus(self) -> int:
        """Remaining GPUs available for allocation."""
        return self.total_gpus - self.gpu_cursor

    def allocate(
        self,
        gpus: int | List[int],
        component_types: Optional[List[str]] = None,
        consume: bool = True,
    ) -> None:
        """
        Allocate resources directly on this node.

        Modifies self.allocations and advances gpu_cursor.

        Args:
            gpus: GPU units to allocate. Can be int (same for all) or list (per-component).
            component_types: Components to record allocations for. If None, just advances cursor.
            consume: If True, consume the GPUs from the node.
        """
        component_types = component_types or []

        # Calculate total GPUs to allocate
        if not component_types:
            gpus_to_allocate = sum(gpus) if isinstance(gpus, list) else int(gpus)
        else:
            if isinstance(gpus, list):
                if len(gpus) != len(component_types):
                    raise RuntimeError(
                        f"GPU list length {len(gpus)} != "
                        f"component_types length {len(component_types)}"
                    )
                gpus_to_allocate = sum(gpus)
            else:
                gpus_to_allocate = int(gpus)

        if gpus_to_allocate <= 0:
            logger.warning(
                "The number of GPUs to allocate must be greater than 0, "
                f"but got {gpus_to_allocate}"
            )
            return

        if gpus_to_allocate > self.available_gpus:
            raise RuntimeError(
                "The number of GPUs to allocate must be <= the available GPUs, "
                f"but got {gpus_to_allocate} > {self.available_gpus}"
            )

        # Record allocations using current cursor position
        if component_types:
            cursor = self.gpu_cursor
            for i, comp in enumerate(component_types):
                req = int(gpus[i]) if isinstance(gpus, list) else gpus_to_allocate
                if req == 0:
                    continue
                if req < 0:
                    raise RuntimeError(
                        f"The number of GPUs to allocate must be greater than 0, but got {req}"
                    )
                start, end = cursor, cursor + req - 1
                self.allocations.setdefault(comp, []).append((start, end))
                cursor += req

        self.max_allocated_gpus = max(self.max_allocated_gpus, self.gpu_cursor + gpus_to_allocate)
        # Advance cursor
        if consume:
            self.gpu_cursor += gpus_to_allocate

    def has_resources(self, cpus: int = 0, gpus: int = 0) -> bool:
        """Check if node has sufficient available resources."""
        return self.total_cpus >= cpus and self.available_gpus >= gpus

    @property
    def is_empty(self) -> bool:
        """Check if node has no more allocatable resources."""
        return self.available_gpus == 0

    @property
    def component_types(self) -> List[str]:
        """Get list of component types that have allocations on this node."""
        return list(self.allocations.keys())

    def copy(self) -> "NodeResource":
        """Create a deep copy of this node resource."""
        return NodeResource(
            node_id=self.node_id,
            ip=self.ip,
            total_cpus=self.total_cpus,
            total_gpus=self.total_gpus,
            allocations=copy.deepcopy(self.allocations),
            gpu_cursor=self.gpu_cursor,
            max_allocated_gpus=self.max_allocated_gpus,
        )


@dataclasses.dataclass
class ComponentAllocation:
    """
    Represents resource allocation for a specific component type.

    Attributes:
        component_type: Type of component (train, eval, env, buffer)
        gpu_indices: List of GPU index ranges, e.g., ["0-7", "8-15"]
        total_gpus: Total number of GPUs allocated
        worker_count: Number of workers for this component
    """

    component_type: str
    # node_id -> list of GPU index ranges (strings) for that node,
    # e.g. {"nodeA": ["0-3"], "nodeB": ["4-7"]}
    node_to_gpu_indices: Dict[str, List[str]]
    total_gpus: int
    worker_count: int

    def to_index_string(self) -> str:
        """Convert GPU indices to a compact string format like '0-7, 8-15'."""
        flat: List[str] = []
        for _, ranges in self.node_to_gpu_indices.items():
            flat.extend(ranges)
        return ", ".join(flat)

    def get_node_index_string(self, node_id: str) -> str:
        """Get index string for a specific node_id."""
        ranges = self.node_to_gpu_indices.get(node_id, [])
        return ", ".join(ranges)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component_type": self.component_type,
            "node_to_gpu_indices": self.node_to_gpu_indices,
            "index_string": self.to_index_string(),
            "total_gpus": self.total_gpus,
            "worker_count": self.worker_count,
        }


@dataclasses.dataclass
class ResourcePool:
    """
    A resource pool containing allocated nodes for specific components.

    This class tracks:
    - Which nodes belong to this pool
    - Which component types use this pool
    - Fine-grained GPU index allocations per component per node

    component_types is auto-inferred from node allocations if not provided.
    If 'train' exists in allocations, 'buffer' is automatically added.
    """

    name: str
    nodes: List[NodeResource]
    _component_types: Optional[List[str]] = dataclasses.field(default=None)
    # optional pre-built view (used for YAML export / debugging)
    component_allocations: Dict[str, ComponentAllocation] = dataclasses.field(default_factory=dict)

    @property
    def component_types(self) -> List[str]:
        """
        Get component types for this pool.

        Auto-infers from node allocations if not explicitly set.
        Auto-adds 'buffer' if 'train' exists.
        """
        if self._component_types is not None:
            return self._component_types

        # Infer from node allocations
        types_set: set[str] = set()
        for node in self.nodes:
            types_set.update(node.component_types)

        # Auto-add buffer if train exists
        if "train" in types_set:
            types_set.add("buffer")

        self._component_types = list(types_set)

        return list(types_set)

    @property
    def total_gpus(self) -> int:
        """Total available GPUs in this pool."""
        return sum(n.total_gpus for n in self.nodes)

    @property
    def total_cpus(self) -> int:
        """Total available CPUs in this pool."""
        return sum(n.total_cpus for n in self.nodes)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in this pool."""
        return len(self.nodes)

    @property
    def node_ids(self) -> List[str]:
        """List of node IDs in this pool."""
        return [n.node_id for n in self.nodes]

    def get_component_node_count(self, component_type: str) -> int:
        """Get the number of nodes that have the component type."""
        return sum(1 for n in self.nodes if component_type in n.component_types)

    def get_component_indices(self, component_type: str) -> str:
        """
        Get the GPU index string for a component type across all nodes.

        Returns:
            Index string like "0-7" or "0-7, 8-15" for multi-node.
        """
        if component_type not in self.component_types:
            return ""
        if not self.component_allocations:
            self._rebuild_component_allocations()
        ca = self.component_allocations.get(component_type)
        return ca.to_index_string() if ca else ""

    def _rebuild_component_allocations(self) -> None:
        """
        Build ComponentAllocation view from per-node allocations.

        We export indices in a *pool-global* index space by concatenating nodes in order
        and offsetting each node-local range by the cumulative node capacity.
        """
        self.component_allocations = {}
        offset = 0
        for node in self.nodes:
            for comp, ranges in node.allocations.items():
                for s, e in ranges:
                    gs, ge = offset + s, offset + e
                    idx_str = f"{gs}-{ge}" if gs != ge else str(gs)
                    ca = self.component_allocations.get(comp)
                    if ca is None:
                        ca = ComponentAllocation(
                            component_type=comp,
                            node_to_gpu_indices={},
                            total_gpus=0,
                            worker_count=0,
                        )
                        self.component_allocations[comp] = ca
                    ca.node_to_gpu_indices.setdefault(node.node_id, []).append(idx_str)
                    ca.total_gpus += ge - gs + 1
            offset += node.total_gpus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "num_node": self.num_nodes,
            "num_gpus": self.total_gpus,
            "node_ids": self.node_ids,
            "component_types": self.component_types,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "ip": n.ip,
                    "total_gpus": n.total_gpus,
                    "total_cpus": n.total_cpus,
                    "allocations": copy.deepcopy(n.allocations),
                }
                for n in self.nodes
            ],
        }
        # Add component index mappings
        for comp_type in self.component_types:
            indices = self.get_component_indices(comp_type)
            if indices:
                result[comp_type] = indices

        return result

    def to_yaml_dict(self) -> Dict[str, Any]:
        """
        Convert to YAML-friendly dictionary format.

        Output format:
            name: "train_pool"
            num_node: 1
            num_gpus: 8   # per-node allocated gpus, NOT total
            train: "0-7"
        """
        # Use total_gpus as it represents total GPUs per node
        per_node_gpus = [n.total_gpus for n in self.nodes]
        num_gpus: Union[int, List[int]]
        if len(set(per_node_gpus)) == 1:
            num_gpus = per_node_gpus[0]
        else:
            # Heterogeneous pools: keep per-node gpus list for reversibility
            num_gpus = per_node_gpus

        result = {
            "name": self.name,
            "num_node": self.num_nodes,
            "num_gpus": num_gpus,
        }
        for comp_type in self.component_types:
            indices = self.get_component_indices(comp_type)
            if indices:
                result[comp_type] = indices
        return result

    @staticmethod
    def _parse_index_str(s: Any) -> List[Tuple[int, int]]:
        """
        Parse index string like:
          - "0-7"
          - "0-3, 8-11"
          - 5
        into list of (start, end) inclusive.
        """
        if s is None:
            return []
        if isinstance(s, int):
            return [(s, s)]
        if not isinstance(s, str):
            return []
        s = s.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        ranges: List[Tuple[int, int]] = []
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                start, end = int(a.strip()), int(b.strip())
            else:
                start = end = int(p)
            if end < start:
                start, end = end, start
            ranges.append((start, end))
        return ranges

    @staticmethod
    def _split_global_range_by_nodes(
        start: int, end: int, node_offsets: List[int]
    ) -> List[Tuple[int, int, int]]:
        """
        Split a global [start,end] into per-node segments.

        node_offsets: cumulative starts per node, length num_nodes+1.
        Returns list of (node_idx, local_start, local_end).
        """
        out: List[Tuple[int, int, int]] = []
        cur = start
        while cur <= end:
            # find node containing cur
            node_idx = max(i for i in range(len(node_offsets) - 1) if node_offsets[i] <= cur)
            node_start = node_offsets[node_idx]
            node_end = node_offsets[node_idx + 1] - 1
            seg_end = min(end, node_end)
            out.append((node_idx, cur - node_start, seg_end - node_start))
            cur = seg_end + 1
        return out

    @classmethod
    def from_yaml_dict(
        cls,
        pool_cfg: Dict[str, Any],
        cluster_nodes: Dict[str, "NodeResource"],
        *,
        used_node_ids: Optional[set[str]] = None,
    ) -> "ResourcePool":
        """
        Build a ResourcePool from a YAML pool dict.

        Supported input fields:
          - name: str
          - num_node: int
          - num_gpus: int (per-node) OR List[int] (per-node gpus list)
          - optional node_ids: List[str] to pin pools to specific nodes
          - component keys: train/eval/env/buffer/... with values like "0-7, 8-15"

        component_types is auto-inferred from allocations if not explicitly listed.
        'buffer' is auto-added if 'train' exists.
        """
        # pool_cfg may be a dict or a Config-like object; normalize to dict.
        if not isinstance(pool_cfg, dict):
            if hasattr(pool_cfg, "to_dict"):
                pool_cfg = pool_cfg.to_dict()
            elif hasattr(pool_cfg, "model_dump"):
                pool_cfg = pool_cfg.model_dump()
            else:
                pool_cfg = dict(pool_cfg)

        # (warning) Simplified assumption: all cluster nodes have identical GPU capacity.
        if pool_cfg.get("node_ids") is None:
            node_caps = [int(n.total_gpus) for n in cluster_nodes.values()]
            if len(set(node_caps)) != 1:
                raise ValueError(
                    "Manual mode without specified node_ids requires "
                    "per-node num_gpus to be a single value."
                )

        name = str(pool_cfg.get("name"))
        num_node = int(pool_cfg.get("num_node"))

        node_ids = pool_cfg.get("node_ids", None)
        if node_ids is not None:
            num_gpus_field = pool_cfg.get("num_gpus")
            node_ids = [str(x) for x in node_ids]
            assert isinstance(
                num_gpus_field, list
            ), "if node_ids is provided, num_gpus must be a list"
            if len(node_ids) != num_node or len(num_gpus_field) != num_node:
                raise ValueError(
                    f"[{name}] node_ids or num_gpus length {len(node_ids)} != num_node {num_node}"
                )
            per_node_gpus = [int(x) for x in num_gpus_field]
        else:
            used = used_node_ids or set()
            candidates = [n for n in cluster_nodes.values() if n.node_id not in used]
            if len(candidates) < num_node:
                raise ValueError(f"[{name}] Not enough nodes to satisfy manual pool request")

            # Deterministically pick the first N unused nodes.
            candidates = sorted(candidates, key=lambda n: n.node_id)[:num_node]

            per_node_gpus = [int(n.total_gpus) for n in candidates]
            node_ids = [n.node_id for n in candidates]

        # build pool nodes (node-local indexing 0..per_node_gpus-1)
        nodes: List[NodeResource] = []
        for nid in node_ids:
            nodes.append(cluster_nodes[nid].copy())

        # build offsets for global indices -> node-local indices
        offsets = [0]
        for g in per_node_gpus:
            offsets.append(offsets[-1] + g)

        # parse component allocations (auto-infers component_types)
        reserved_keys = {"name", "num_node", "num_gpus", "node_ids"}
        node_allocated_gpus = [0] * num_node
        for k, v in pool_cfg.items():
            if k in reserved_keys:
                continue
            if v is None:
                continue
            for gs, ge in cls._parse_index_str(v):
                for node_idx, ls, le in cls._split_global_range_by_nodes(gs, ge, offsets):
                    nodes[node_idx].allocations.setdefault(str(k), []).append((ls, le))
                    node_allocated_gpus[node_idx] = max(node_allocated_gpus[node_idx], le + 1)

        # component_types will be auto-inferred from node allocations
        pool = cls(name=name, nodes=nodes)
        pool._rebuild_component_allocations()
        return pool


class ResourcePoolPlanner:
    """
    Plans and manages resource pools for component placement.

    This planner:
    1. Discovers cluster resources
    2. Validates scheduling requirements against available resources
    3. Allocates resources to pools based on the placement strategy
    4. Tracks fine-grained component-to-GPU mappings
    """

    def __init__(
        self,
        scheduling: ComponentScheduling,
        cluster_info: Optional[Dict[str, Any]] = None,
    ):
        self.scheduling = scheduling
        self._cluster_info: Dict[str, Any] = cluster_info
        self._node_resources: Dict[str, NodeResource] = {}
        self._resource_pools: Dict[str, ResourcePool] = {}
        self.resource_summary: Dict[str, Any] = {}

    def discover_cluster_resources(self) -> Dict[str, NodeResource]:
        """
        Discover and cache cluster node resources.

        Returns:
            Dictionary mapping node_id to NodeResource.
        """
        if self._cluster_info is None:
            self._cluster_info = get_cluster_resources()
        node_id_to_resources = self._cluster_info.get("node_id_to_resources", {})

        self._node_resources = {}
        for node_id, info in node_id_to_resources.items():
            self._node_resources[node_id] = NodeResource(
                node_id=node_id,
                ip=info.get("ip", ""),
                total_cpus=info.get("CPU", 0),
                total_gpus=info.get("GPU", 0),
            )

        logger.info(
            f"[ResourcePoolPlanner] Discovered {len(self._node_resources)} nodes, "
            f"total GPUs: {sum(n.total_gpus for n in self._node_resources.values())}, "
            f"total CPUs: {sum(n.total_cpus for n in self._node_resources.values())}"
        )

        return self._node_resources

    def validate_scheduling(
        self, strategy: str = "disaggregate", env_strategy: str = "default"
    ) -> Tuple[bool, str]:
        """
        Validate that cluster resources can satisfy scheduling requirements.

        Args:
            strategy: The placement strategy to use.
            env_strategy: The environment strategy to use.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self._node_resources:
            self.discover_cluster_resources()

        total_gpus = sum(n.total_gpus for n in self._node_resources.values())
        total_cpus = sum(n.total_cpus for n in self._node_resources.values())

        # Calculate required resources
        train_gpus_required, train_cpus_required = self.scheduling.train_pool_requirements()
        env_gpus_required, env_cpus_required = self.scheduling.get_component_requirements("env")
        eval_gpus_required, eval_cpus_required = self.scheduling.get_component_requirements("eval")

        if env_strategy == "device-colocate":
            rollout_gpus_required = max(env_gpus_required, eval_gpus_required)
        else:
            rollout_gpus_required = env_gpus_required + eval_gpus_required

        # Validate GPU requirements
        if strategy == "disaggregate":
            required_gpus = train_gpus_required + rollout_gpus_required
        elif strategy == "colocate":
            required_gpus = max(train_gpus_required, rollout_gpus_required)
        else:
            raise ValueError(f"Unknown placement strategy: {strategy}")

        if required_gpus > total_gpus:
            return (
                False,
                f"Insufficient GPUs: required {required_gpus}, available {total_gpus}",
            )

        # Validate CPU requirements
        required_cpus = train_cpus_required + eval_cpus_required + env_cpus_required
        if required_cpus > total_cpus:
            return (
                False,
                f"Insufficient CPUs: required {required_cpus}, available {total_cpus}",
            )

        # Log resource validation results
        logger.info(
            f"[ResourcePoolPlanner] Resource validation passed: "
            f"GPUs {required_gpus}/{total_gpus}, CPUs {required_cpus}/{total_cpus}"
        )

        self.resource_summary = {
            "train_pool_required_gpus": train_gpus_required,
            "rollout_pool_required_gpus": rollout_gpus_required,
            "env_required_gpus": env_gpus_required,
            "eval_required_gpus": eval_gpus_required,
        }

        return True, ""

    def plan_resource_pools(
        self, strategy: str = "disaggregate", env_strategy: str = "default"
    ) -> Dict[str, ResourcePool]:
        """
        Plan resource pools based on placement strategy.

        Args:
            strategy: The placement strategy to use.
            env_strategy: The environment placement strategy.

        Returns:
            Dictionary mapping pool name to ResourcePool.
        """
        if not self._node_resources:
            self.discover_cluster_resources()

        # Validate first
        is_valid, error_msg = self.validate_scheduling(strategy, env_strategy)
        if not is_valid:
            raise ValueError(f"Resource validation failed: {error_msg}")

        if strategy == "disaggregate":
            self._plan_disaggregate_pools(env_strategy)
        elif strategy == "colocate":
            self._plan_colocate_pools(env_strategy)
        else:
            raise ValueError(f"Unknown placement strategy: {strategy}")

        # Log pool allocation results
        for pool_name, pool in self._resource_pools.items():
            logger.info(
                f"[ResourcePoolPlanner] Pool '{pool_name}': "
                f"{pool.num_nodes} nodes, {pool.total_gpus} GPUs, "
                f"components: {pool.component_types}"
            )
            for comp_type in pool.component_types:
                indices = pool.get_component_indices(comp_type)
                if indices:
                    logger.info(f"  - {comp_type}: {indices}")

        return self._resource_pools

    def load_manual_resource_pools(
        self, resource_pool_cfg: List[Dict[str, Any]]
    ) -> Dict[str, ResourcePool]:
        """
        Load resource pools from a manual YAML config list.
        """
        if not self._node_resources:
            self.discover_cluster_resources()

        used: set[str] = set()
        pools: Dict[str, ResourcePool] = {}
        for p in resource_pool_cfg:
            pool = ResourcePool.from_yaml_dict(p, self._node_resources, used_node_ids=used)
            used.update(pool.node_ids)
            pools[pool.name] = pool
        self._resource_pools = pools
        return pools

    def _plan_disaggregate_pools(self, env_strategy: str = "default") -> None:
        """
        Plan separate resource pools for train and rollout.

        - train_pool: Contains Train workers + Buffer workers (colocated)
        - rollout_pool: Contains Eval workers + Env workers (colocated)
        """
        if not self.resource_summary:
            raise ValueError("Resource summary not available. Call validate_scheduling() first.")
        train_gpus_remaining = self.resource_summary["train_pool_required_gpus"]
        env_gpus_remaining = self.resource_summary["env_required_gpus"]
        eval_gpus_remaining = self.resource_summary["eval_required_gpus"]

        # Sort nodes by GPU count (descending) for better allocation
        sorted_nodes = sorted(
            self._node_resources.values(),
            key=lambda n: n.total_gpus,
            reverse=True,
        )

        # ===== Allocate train pool =====
        train_pool_nodes: List[NodeResource] = []

        for node in sorted_nodes:
            if node.is_empty:
                continue
            if train_gpus_remaining <= 0:
                break

            gpus_to_allocate = min(node.available_gpus, train_gpus_remaining)

            # Allocate train slice directly on the node
            node.allocate(
                gpus=gpus_to_allocate,
                component_types=["train"],
                consume=True,
            )
            train_pool_nodes.append(node)

            # Update tracking
            train_gpus_remaining -= gpus_to_allocate

        # ===== Allocate rollout pool =====
        rollout_pool_nodes: List[NodeResource] = []

        for node in sorted_nodes:
            if node.is_empty:
                continue
            if env_gpus_remaining <= 0 and eval_gpus_remaining <= 0:
                break
        
            eval_gpus_to_allocate = min(node.available_gpus, eval_gpus_remaining)
            if eval_gpus_to_allocate > 0:
                if env_strategy == "device-colocate":
                    node.allocate(
                        gpus=eval_gpus_to_allocate,
                        component_types=["eval"],
                        consume=False,
                    )
                else:
                    node.allocate(
                        gpus=eval_gpus_to_allocate,
                        component_types=["eval"],
                        consume=True,
                    )
                eval_gpus_remaining -= eval_gpus_to_allocate

            env_gpus_to_allocate = min(node.available_gpus, env_gpus_remaining)
            if env_gpus_to_allocate > 0:
                node.allocate(
                    gpus=env_gpus_to_allocate,
                    component_types=["env"],
                    consume=True,
                )
                env_gpus_remaining -= env_gpus_to_allocate

            # Only add to rollout pool if not already in train pool
            if node not in train_pool_nodes:
                rollout_pool_nodes.append(node)

        # Create resource pools with auto-inferred component_types
        self._resource_pools["train_pool"] = ResourcePool(
            name="train_pool",
            nodes=train_pool_nodes,
        )
        self._resource_pools["rollout_pool"] = ResourcePool(
            name="rollout_pool",
            nodes=rollout_pool_nodes,
        )

        # Build component allocation views from node.allocations
        self._resource_pools["train_pool"]._rebuild_component_allocations()
        self._resource_pools["rollout_pool"]._rebuild_component_allocations()

    def _plan_colocate_pools(self, env_strategy: str = "default") -> None:
        """
        Plan a single global resource pool for all components.

        All components share the same pool, with GPU indices potentially overlapping
        for colocated components.
        """
        if not self.resource_summary:
            raise ValueError("Resource summary not available. Call validate_scheduling() first.")
        train_gpus_remaining = self.resource_summary["train_pool_required_gpus"]
        eval_gpus_remaining = self.resource_summary["eval_required_gpus"]
        env_gpus_remaining = self.resource_summary["env_required_gpus"]

        # In colocate mode, we assign the full node GPU slice as a shared allocation
        # for all components.
        sorted_nodes = sorted(
            self._node_resources.values(),
            key=lambda n: n.total_gpus,
            reverse=True,
        )
        global_pool_nodes: List[NodeResource] = []

        for node in sorted_nodes:
            if node.is_empty:
                continue
            gpus_on_node = node.available_gpus
            if train_gpus_remaining <= 0 and env_gpus_remaining <= 0 and eval_gpus_remaining <= 0:
                break

            # Allocate train slice (shared with rollout in colocate mode)
            if train_gpus_remaining > 0:
                train_gpus_to_allocate = min(gpus_on_node, train_gpus_remaining)
                # In colocate, train allocation doesn't advance cursor (shared with rollout)
                node.allocate(
                    gpus=train_gpus_to_allocate,
                    component_types=["train"],
                    consume=False,
                )
                train_gpus_remaining -= train_gpus_to_allocate
            
            # Allocate eval slice (overlaps with train in colocate mode)
            eval_gpus_to_allocate = min(gpus_on_node, eval_gpus_remaining)
            if eval_gpus_to_allocate > 0:
                if env_strategy == "device-colocate":
                    node.allocate(
                        gpus=eval_gpus_to_allocate,
                        component_types=["eval"],
                        consume=False,
                    )
                else:
                    node.allocate(
                        gpus=eval_gpus_to_allocate,
                        component_types=["eval"],
                        consume=True,
                    )
                eval_gpus_remaining -= eval_gpus_to_allocate
            global_pool_nodes.append(node)

            # Allocate env slice (overlaps with train in colocate mode)
            env_gpus_to_allocate = min(gpus_on_node, env_gpus_remaining)
            if env_gpus_to_allocate > 0:
                node.allocate(
                    gpus=env_gpus_to_allocate,
                    component_types=["env"],
                    consume=True,
                )
                env_gpus_remaining -= env_gpus_to_allocate
        # Create resource pool with auto-inferred component_types
        self._resource_pools["global_pool"] = ResourcePool(
            name="global_pool",
            nodes=global_pool_nodes,
        )
        self._resource_pools["global_pool"]._rebuild_component_allocations()

    def get_component_node_count(self, component_type: str) -> int:
        """Get the number of nodes that have the component type."""
        return sum(
            pool.get_component_node_count(component_type) for pool in self._resource_pools.values()
        )

    def get_resource_pools(self, pool_name: str = None) -> Dict[str, ResourcePool]:
        """Get the planned resource pools."""
        if pool_name is None:
            return self._resource_pools
        return self._resource_pools.get(pool_name)

    def get_pool_for_component(self, component_type: str) -> Optional[ResourcePool]:
        """Get the resource pool that contains a specific component type."""
        for pool in self._resource_pools.values():
            if component_type in pool.component_types:
                return pool
        return None

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get raw cluster information."""
        return self._cluster_info

    def get_node_resources(self) -> Dict[str, NodeResource]:
        """Get node resources dictionary."""
        return self._node_resources

    def to_yaml_config(self) -> Dict[str, Any]:
        """
        Generate a YAML-compatible configuration representing the resource pools.

        Output format:
            resource_pool:
              - name: "train_pool"
                num_node: 1
                num_gpus: 8  # per-node GPUs
                train: "0-7"
              - name: "rollout_pool"
                num_node: 2
                num_gpus: 8  # per-node GPUs
                eval: "0-7, 8-15"
                env: "0-7, 8-15"
        """
        pools_config = []
        for pool in self._resource_pools.values():
            pools_config.append(pool.to_yaml_dict())

        return pools_config

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the resource planning."""
        return {
            "cluster": {
                "n_nodes": len(self._node_resources),
                "total_gpus": sum(n.total_gpus for n in self._node_resources.values()),
                "total_cpus": sum(n.total_cpus for n in self._node_resources.values()),
            },
            "pools": {name: pool.to_dict() for name, pool in self._resource_pools.items()},
            "yaml_config": self.to_yaml_config(),
        }
