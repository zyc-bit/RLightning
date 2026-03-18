import heapq
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ray

from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class AsyncRouter(ABC):
    """Abstract base class for request routing strategies."""

    @abstractmethod
    def assign(self, current_loads: List[int], num_tasks: int, env_ids: Optional[Sequence[str]] = None) -> List[int]:
        """
        Assign num_tasks to policy indices.

        Args:
            current_loads: List of current loads for each policy.
            num_tasks: Number of tasks to assign.
            env_ids: Optional sequence of environment IDs for affinity-based routing.

        Returns:
            List of policy indices corresponding to the chosen target for each task.
        """
        pass


class SimpleRouter(AsyncRouter):
    """Load-balancing router that assigns tasks to policies with minimum load."""

    def assign(self, current_loads: List[int], num_tasks: int) -> List[int]:
        """
        Assign num_tasks to indices based on current_loads using a min-heap.
        Returns a list of indices corresponding to the chosen target for each task in order.
        """
        heap = [(int(current_loads[i]), i) for i in range(len(current_loads))]
        if not heap:
            raise ValueError("No policies available for routing")
        heapq.heapify(heap)

        assignments: List[int] = []
        for _ in range(num_tasks):
            load, idx = heapq.heappop(heap)
            assignments.append(idx)
            heapq.heappush(heap, (load + 1, idx))
        return assignments


class NodeAffinityRouter(AsyncRouter):
    """Route tasks to policies on the same node as env workers."""

    def __init__(
        self,
        component_distribution: Dict[str, Dict[str, Dict[str, Any]]],
        policy_node_ids: Optional[List[str]] = None,
    ) -> None:
        self._env_worker_to_node: Dict[int, str] = {}
        for node_id, components in component_distribution.items():
            for env_worker_id in components.get("env", {}).get("ids", []):
                self._env_worker_to_node[int(env_worker_id)] = node_id
        self._policy_node_ids = policy_node_ids or []

    @staticmethod
    def _parse_env_worker_index(env_id: str) -> Optional[int]:
        try:
            suffix = env_id.rsplit("-", 1)[-1]
            return int(suffix)
        except (ValueError, AttributeError):
            return None

    def assign(self, current_loads: List[int], num_tasks: int, env_ids: Optional[Sequence[str]] = None) -> List[int]:
        if not self._policy_node_ids or not env_ids:
            return SimpleRouter().assign(current_loads, num_tasks)

        assignments: List[int] = []
        for env_id in env_ids:
            worker_idx = self._parse_env_worker_index(env_id)
            node_id = self._env_worker_to_node.get(worker_idx) if worker_idx is not None else None
            candidates = [idx for idx, policy_node_id in enumerate(self._policy_node_ids) if policy_node_id == node_id]
            if not candidates:
                idx = int(min(range(len(current_loads)), key=lambda i: current_loads[i]))
            else:
                idx = int(min(candidates, key=lambda i: current_loads[i]))
            assignments.append(idx)
            current_loads[idx] += 1
        return assignments


class SyncRouter(ABC):
    """Abstract base class for sync rollout routing strategies."""

    @abstractmethod
    def select_policy(self, env_id: Optional[str] = None) -> Any:
        """Select a policy for a single env_id in sync mode."""
        pass


class SyncSimpleRouter(SyncRouter):
    """Sync router using a single idle deque."""

    def __init__(self, eval_policies: Sequence[Any]) -> None:
        self._eval_policies = list(eval_policies)
        self._idle_deque = deque(self._eval_policies)
        self._lock = threading.Lock()

    def _flush_idle(self) -> None:
        """Refresh the idle deque by probing busy eval policies."""
        if not self._eval_policies:
            return
        busy_list = list(set(self._eval_policies) - set(self._idle_deque))

        if not busy_list:
            return

        if not isinstance(busy_list[0], ray.actor.ActorHandle):
            # local policy, all are idle
            self._idle_deque.extend(busy_list)
            return

        futures = [p.check_idle.remote() for p in busy_list]
        idle_list, _ = ray.wait(futures, num_returns=len(futures), timeout=0.01)
        future_to_idx = {f.hex(): i for i, f in enumerate(futures)}
        self._idle_deque.extend([busy_list[future_to_idx[f.hex()]] for f in idle_list])

    def select_policy(self, env_id: Optional[str] = None) -> Any:
        """Select policy with internal locking (simple router uses single global lock)."""
        with self._lock:
            while len(self._idle_deque) == 0:
                self._flush_idle()
        return self._idle_deque.popleft()


class SyncNodeAffinityRouter(SyncRouter):
    """Sync router that prefers policies on the same node as env workers.

    This router supports concurrent scheduling across different nodes while
    maintaining sequential scheduling within each node through per-node locks.
    """

    def __init__(
        self,
        eval_policies: Sequence[Any],
        component_distribution: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> None:
        self._eval_policies = list(eval_policies)
        self._env_worker_to_node: Dict[int, str] = {}
        self._node_to_eval_policies: Dict[str, List[Any]] = {}

        # idle deques for each node (kept for backward compat but no longer used for routing)
        self._idle_deques_by_node: Dict[str, deque] = {}
        # global idle deque
        self._idle_deque = deque(self._eval_policies)

        # build node mappings
        self.env_worker_num = 0
        self._build_node_mappings(component_distribution)

        # Round-robin counters per node (replaces idle-check mechanism)
        self._rr_counters: Dict[str, int] = {
            node_id: 0 for node_id in self._node_to_eval_policies
        }
        self._global_rr_counter = 0

        # per-node locks for concurrent cross-node scheduling
        self._node_locks: Dict[str, threading.Lock] = {
            node_id: threading.Lock() for node_id in self._node_to_eval_policies
        }
        # global lock for global idle deque
        self._global_lock = threading.Lock()

    @staticmethod
    def _parse_env_worker_index(env_id: str) -> Optional[int]:
        try:
            suffix = env_id.rsplit("-", 1)[-1]
            return int(suffix)
        except (ValueError, AttributeError):
            return None

    def _build_node_mappings(
        self, component_distribution: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Tuple[Dict[int, str], Dict[str, List[Any]], Dict[str, deque]]:
        """Build env_worker -> node, node -> eval_policies, and node -> idle_deque mappings.

        Args:
            component_distribution: Component distribution from placement strategy.

        Returns:
            Tuple of (env_worker_to_node, node_to_eval_policies, idle_deques_by_node).
        """
        for node_id, components in component_distribution.items():
            env_worker_ids = components.get("env", {}).get("ids", [])
            eval_worker_ids = components.get("eval", {}).get("ids", [])

            # Build env_worker -> node mapping
            if env_worker_ids:
                assert len(eval_worker_ids) > 0, "eval workers must be assigned to the same node as env workers"
                self.env_worker_num += len(env_worker_ids)
                for env_worker_id in env_worker_ids:
                    self._env_worker_to_node[int(env_worker_id)] = node_id

            # Build node -> eval_policies and idle_deques mappings
            if eval_worker_ids:
                assert len(env_worker_ids) > 0, "env workers must be assigned to the same node as eval workers"
                eval_policies = [self._eval_policies[int(worker_id)] for worker_id in eval_worker_ids]
                self._node_to_eval_policies[node_id] = eval_policies
                self._idle_deques_by_node[node_id] = deque(eval_policies)

        logger.info(f"node_to_eval_policies: {self._node_to_eval_policies}")
        logger.info(f"env_worker_to_node: {self._env_worker_to_node}")

    def _flush_idle(self) -> None:
        """Refresh the global idle deque by probing busy eval policies."""
        busy_list = list(set(self._eval_policies) - set(self._idle_deque))
        if not busy_list:
            return
        if not isinstance(busy_list[0], ray.actor.ActorHandle):
            self._idle_deque.extend(busy_list)
            return
        futures = [p.check_idle.remote() for p in busy_list]
        idle_list, _ = ray.wait(futures, num_returns=len(futures), timeout=0.01)
        future_to_idx = {f.hex(): i for i, f in enumerate(futures)}
        self._idle_deque.extend([busy_list[future_to_idx[f.hex()]] for f in idle_list])

    def _flush_idle_by_node(self, node_id: str) -> None:
        """Refresh the idle deque for a specific node by probing busy eval policies."""
        idle_deque = self._idle_deques_by_node[node_id]
        policies = self._node_to_eval_policies[node_id]
        busy_list = list(set(policies) - set(idle_deque))
        if not busy_list:
            return
        if not isinstance(busy_list[0], ray.actor.ActorHandle):
            idle_deque.extend(busy_list)
            return

        futures = [p.check_idle.remote() for p in busy_list]
        idle_list, _ = ray.wait(futures, num_returns=len(futures), timeout=0.01)
        future_to_idx = {f.hex(): i for i, f in enumerate(futures)}
        idle_deque.extend([busy_list[future_to_idx[f.hex()]] for f in idle_list])

    def _get_target_node(self, env_id: str) -> Optional[str]:
        """Get the target node for an env_id based on worker placement."""
        worker_idx = self._parse_env_worker_index(env_id)
        if worker_idx is None:
            return None
        return self._env_worker_to_node.get(worker_idx)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes with eval policies."""
        return len(self._node_locks)

    def select_policy(self, env_id: str) -> Any:
        """Select policy using round-robin within the target node.

        With max_pending_tasks > 1 on the eval TaskSubmitter, backpressure is
        handled by TaskSubmitter itself. The router only needs to distribute
        tasks evenly across eval policies on the correct node — no idle-check
        RPCs needed.

        Args:
            env_id: Environment ID to determine target node.

        Returns:
            Selected policy instance.
        """
        target_node_id = self._get_target_node(env_id)

        if target_node_id is not None and target_node_id in self._node_to_eval_policies:
            policies = self._node_to_eval_policies[target_node_id]
            with self._node_locks[target_node_id]:
                idx = self._rr_counters[target_node_id] % len(policies)
                self._rr_counters[target_node_id] += 1
            result = policies[idx]
        else:
            # Fallback to global round-robin
            with self._global_lock:
                idx = self._global_rr_counter % len(self._eval_policies)
                self._global_rr_counter += 1
            result = self._eval_policies[idx]

        return result


def create_router(
    rollout_mode: str,
    router_type: str,
    eval_policies: Sequence[Any],
    global_resource_manager=None,
) -> Union[SyncRouter, AsyncRouter]:
    """Factory for sync routers.

    Args:
        rollout_mode: Rollout mode ("sync" or "async").
        router_type: Type of router ("simple" or "node_affinity").
        eval_policies: List of eval policy instances.
        global_resource_manager: Global resource manager. Required for "node_affinity" router.

    Returns:
        A SyncRouter or AsyncRouter instance.
    """
    if rollout_mode == "sync":
        if router_type == "node_affinity":
            assert global_resource_manager is not None, "global_resource_manager is required for node_affinity router"
            component_distribution = global_resource_manager.get_component_distribution()
            assert len(component_distribution) > 0, "component_distribution is required for node_affinity router"
            return SyncNodeAffinityRouter(
                eval_policies=eval_policies,
                component_distribution=component_distribution,
            )
        return SyncSimpleRouter(eval_policies)
    elif rollout_mode == "async":
        return SimpleRouter()
    else:
        raise ValueError(f"Invalid rollout mode: {rollout_mode}")
