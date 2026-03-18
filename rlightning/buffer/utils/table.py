"""Episode-to-storage shard assignment table."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


class EpisodeTable:
    """
    Track episode -> storage shard mapping and storage -> train worker mapping.

    The MVP version uses a simple, even distribution strategy for both:
    - env_ids are assigned to storage shards in round-robin order with load balance.
    - train workers are assigned to storage shards in contiguous, even blocks.

    When node affinity is enabled, envs and train workers are bound to storages
    on the same node using the component distribution view.
    """

    def __init__(
        self,
        num_storages: int,
        env_ids: Optional[Sequence[str]] = None,
        num_train_workers: Optional[int] = None,
        component_distribution: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        node_affinity_env: bool = False,
        node_affinity_train: bool = False,
    ) -> None:
        """Initialize the table.

        Args:
            num_storages: Number of storage shards.
            env_ids: Optional list of env IDs to register.
            num_train_workers: Optional number of train workers to assign.
            component_distribution: Optional node-level component ID mapping for affinity assignment.
            node_affinity_env: Whether to enable node-affinity assignment for env workers.
            node_affinity_train: Whether to enable node-affinity assignment for train workers.
        """
        if num_storages < 1:
            raise ValueError(f"num_storages must be positive, got {num_storages}")
        self.num_storages = num_storages
        self._component_distribution = component_distribution or {}
        self._node_affinity_env = node_affinity_env
        self._node_affinity_train = node_affinity_train

        self._env_to_storage: Dict[str, int] = {}
        self._storage_env_count: List[int] = [0 for _ in range(num_storages)]
        self._rr_cursor: int = 0
        self._storage_to_train_workers: Dict[int, List[int]] = {}
        self._env_id_to_worker_index: Dict[str, int] = {}

        if self._node_affinity_env or self._node_affinity_train:
            assert (
                len(self._component_distribution) > 0
            ), "component_distribution must be provided when node affinity is enabled"
        node_to_storage_ids, node_to_train_ids, node_to_env_ids = self._extract_node_component_ids()

        self._init_env_mapping(env_ids, node_to_storage_ids, node_to_env_ids)
        self._init_train_mapping(num_train_workers, node_to_storage_ids, node_to_train_ids)

    @staticmethod
    def _parse_env_worker_index(env_id: str, fallback_index: int) -> int:
        """Extract worker index from env_id suffix, falling back to order."""
        try:
            suffix = env_id.rsplit("-", 1)[-1]
            return int(suffix)
        except (ValueError, AttributeError):
            return fallback_index

    def _init_env_mapping(
        self,
        env_ids: Optional[Sequence[str]],
        node_to_storage_ids: Dict[str, List[int]],
        node_to_env_ids: Dict[str, List[int]],
    ) -> None:
        if not env_ids:
            return

        self._env_id_to_worker_index = {
            env_id: self._parse_env_worker_index(env_id, idx) for idx, env_id in enumerate(env_ids)
        }
        if self._node_affinity_env:
            self._validate_node_worker_counts(node_to_storage_ids, node_to_env_ids, "env")
            self._assign_envs_with_node_affinity(env_ids, node_to_storage_ids, node_to_env_ids)
        else:
            self.register_envs(env_ids)
        logger.debug(f"env_to_storage: {self._env_to_storage}")

    def _init_train_mapping(
        self,
        num_train_workers: Optional[int],
        node_to_storage_ids: Dict[str, List[int]],
        node_to_train_ids: Dict[str, List[int]],
    ) -> None:
        if num_train_workers is None:
            return
        if self._node_affinity_train:
            self._validate_node_worker_counts(node_to_storage_ids, node_to_train_ids, "train")
            self._assign_train_workers_with_node_affinity(node_to_storage_ids, node_to_train_ids)
        else:
            self._assign_train_workers(num_train_workers)
        logger.debug(f"storage_to_train_workers: {self._storage_to_train_workers}")

    def register_envs(self, env_ids: Iterable[str]) -> None:
        """Register a batch of env_ids with the current assignment strategy."""
        for env_id in env_ids:
            self._register_env(env_id)

    def _register_env(self, env_id: str) -> int:
        """Register a single env_id and return its assigned storage index."""
        if env_id in self._env_to_storage:
            return self._env_to_storage[env_id]

        storage_idx = self._select_storage_for_new_env()
        self._env_to_storage[env_id] = storage_idx
        self._storage_env_count[storage_idx] += 1
        return storage_idx

    def _select_storage_for_new_env(self) -> int:
        """Pick the shard with the lowest load, breaking ties in round-robin order."""
        min_load = min(self._storage_env_count)
        for _ in range(self.num_storages):
            idx = self._rr_cursor
            self._rr_cursor = (self._rr_cursor + 1) % self.num_storages
            if self._storage_env_count[idx] == min_load:
                return idx

        # Fallback, should never hit.
        return 0

    def _assign_envs_with_node_affinity(
        self,
        env_ids: Sequence[str],
        node_to_storage_ids: Dict[str, List[int]],
        node_to_env_ids: Dict[str, List[int]],
    ) -> None:
        worker_index_to_node: Dict[int, str] = {}
        for node_id, env_worker_ids in node_to_env_ids.items():
            for worker_id in env_worker_ids:
                worker_index_to_node[worker_id] = node_id

        for env_id in env_ids:
            worker_idx = self._env_id_to_worker_index.get(env_id)
            node_id = worker_index_to_node.get(worker_idx)
            assert node_id in node_to_storage_ids, f"Missing storage for env worker {worker_idx}"

            storage_idx = node_to_storage_ids[node_id][0]
            assert (
                0 <= storage_idx < self.num_storages
            ), f"Storage index {storage_idx} out of range for {self.num_storages} storages"

            self._env_to_storage[env_id] = storage_idx
            self._storage_env_count[storage_idx] += 1

    def _assign_train_workers(self, num_train_workers: int) -> None:
        """Assign train workers to storages using even, contiguous allocation."""
        if num_train_workers < 1:
            raise ValueError(f"num_train_workers must be positive, got {num_train_workers}")
        self._storage_to_train_workers = {}
        assert num_train_workers % self.num_storages == 0, "num_train_workers must be divisible by num_storages"
        num_workers_per_storage = num_train_workers // self.num_storages
        worker_id = 0
        for storage_idx in range(self.num_storages):
            self._storage_to_train_workers[storage_idx] = list(range(worker_id, worker_id + num_workers_per_storage))
            worker_id += num_workers_per_storage

    def _assign_train_workers_with_node_affinity(
        self,
        node_to_storage_ids: Dict[str, List[int]],
        node_to_train_ids: Dict[str, List[int]],
    ) -> None:
        self._storage_to_train_workers = {idx: [] for idx in range(self.num_storages)}

        for node_id, storage_ids in node_to_storage_ids.items():
            train_ids = node_to_train_ids.get(node_id)
            storage_idx = storage_ids[0]
            assert (
                0 <= storage_idx < self.num_storages
            ), f"Storage index {storage_idx} out of range for {self.num_storages} storages"
            self._storage_to_train_workers[storage_idx] = list(train_ids)

    def get_envs_for_storage(self, storage_idx: int) -> List[str]:
        """List env_ids currently mapped to the given storage shard."""
        self._validate_storage_idx(storage_idx)
        return [env_id for env_id, idx in self._env_to_storage.items() if idx == storage_idx]

    def get_storage_idx_for_env(self, env_id: str) -> int:
        """Return storage index for env_id, assigning it if unseen."""
        if env_id not in self._env_to_storage:
            return self._register_env(env_id)
        return self._env_to_storage[env_id]

    def get_train_workers(self, storage_idx: int) -> List[int]:
        """Return train workers assigned to the given storage shard."""
        self._validate_storage_idx(storage_idx)
        return self._storage_to_train_workers.get(storage_idx)

    def get_storage_to_train_workers(self) -> Dict[int, List[int]]:
        """Return storage -> train worker mapping."""
        return self._storage_to_train_workers

    def get_env_to_storage(self) -> Dict[str, int]:
        """Return env -> storage mapping."""
        return self._env_to_storage

    def _validate_storage_idx(self, storage_idx: int) -> None:
        """Validate that storage_idx is within range."""
        if storage_idx < 0 or storage_idx >= self.num_storages:
            raise IndexError(f"Storage idx {storage_idx} out of range for {self.num_storages} storages")

    def _extract_node_component_ids(
        self,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
        node_to_storage_ids: Dict[str, List[int]] = {}
        node_to_train_ids: Dict[str, List[int]] = {}
        node_to_env_ids: Dict[str, List[int]] = {}

        for node_id, components in self._component_distribution.items():
            buffer_ids = components.get("buffer", {}).get("ids", [])
            train_ids = components.get("train", {}).get("ids", [])
            env_ids = components.get("env", {}).get("ids", [])

            if buffer_ids:
                assert (
                    len(buffer_ids) == 1
                ), f"Node {node_id} has multiple buffer storages {buffer_ids}, expected at most one."
                node_to_storage_ids[node_id] = list(buffer_ids)
            if train_ids:
                node_to_train_ids[node_id] = list(train_ids)
            if env_ids:
                node_to_env_ids[node_id] = list(env_ids)

        return node_to_storage_ids, node_to_train_ids, node_to_env_ids

    def _validate_node_worker_counts(
        self,
        node_to_storage_ids: Dict[str, List[int]],
        node_to_component_ids: Dict[str, List[int]],
        component_type: str,
    ) -> None:
        expected_count: Optional[int] = None
        for node_id in node_to_storage_ids.keys():
            component_ids = node_to_component_ids.get(node_id, [])
            if not component_ids:
                raise RuntimeError(
                    f"Node {node_id} has storage but no {component_type} workers when node affinity enabled."
                )
            count = len(component_ids)
            if expected_count is None:
                expected_count = count
            elif count != expected_count:
                raise RuntimeError(
                    f"Node {node_id} has {count} {component_type} workers; "
                    f"expected {expected_count} for node affinity."
                )
