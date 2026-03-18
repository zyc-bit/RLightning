"""Policy group module for managing multiple policy workers.

This module provides the PolicyGroup class for coordinating multiple
train and eval policies, including weight distribution, communication
group initialization, and rollout management.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import ray
from ray.actor import ActorHandle

from rlightning.env import EnvMeta
from rlightning.types import BatchedData
from rlightning.utils.config import PolicyConfig, TrainConfig
from rlightning.utils.distributed.group_initializer import ParallelMode
from rlightning.utils.logger import get_logger
from rlightning.utils.placement import get_global_resource_manager
from rlightning.utils.ray import TaskSubmitter, resolve_object
from rlightning.utils.utils import InternalFlag
from rlightning.weights.utils import build_weight_buffer

from .base_policy import BasePolicy, PolicyRole
from .utils.router import SyncNodeAffinityRouter, create_router

logger = get_logger(__name__)


class PolicyGroup:
    """Manager for collections of train and eval policies.

    Coordinates multiple policies for distributed training and evaluation,
    handling weight distribution, communication groups, and rollout batching.

    """

    def __init__(
        self,
        train_policy_list: List[BasePolicy | ActorHandle],
        eval_policy_list: List[BasePolicy | ActorHandle],
        config: PolicyConfig,
    ) -> None:
        """Create policy group with given training policy (actor) list and eva policy (actor) list.

        Args:
            train_policy_list (List[BasePolicy  |  ActorHandle]): A list of training policy (actor)
            eval_policy_list (List[BasePolicy  |  ActorHandle]): A list of policy (actor) for evalutation
            config (PolicyConfig): Policy configuration instance.
        """

        self.config = config

        self.policy_list: Dict[PolicyRole, List[BasePolicy | ActorHandle]] = {
            PolicyRole.TRAIN: train_policy_list,
            PolicyRole.EVAL: eval_policy_list,
        }

        self.rollout_mode = config.rollout_mode

        self.placement_info: Dict[str, Dict[str, List]] = {}
        self.weight_buffer_strategy: Optional[str] = None
        self.transfer_info: Dict[str, List[Dict]] = {}
        self.router_map: Dict[str, List[str]] = {}
        self.shared_weight_buffer_map: Dict[str, Dict] = {}
        self._global_resource_manager = get_global_resource_manager()

        # Eval submitter allows pipelining _rollout and _postprocess on the same actor
        self._eval_task_submitter = TaskSubmitter(max_pending_tasks_each_worker=8)
        # Train submitter keeps max_pending=1 to guarantee ordering
        self._train_task_submitter = TaskSubmitter(max_pending_tasks_each_worker=1)

        self.router_type = config.router_type
        self.router = create_router(
            rollout_mode=self.rollout_mode,
            router_type=self.router_type,
            eval_policies=self.eval_list,
            global_resource_manager=self._global_resource_manager,
        )
        logger.info(f"[PolicyGroup] Using router: {type(self.router).__name__}")

        # Pre-initialized thread pool for node_affinity router to avoid per-call overhead
        self._rollout_executor: Optional[ThreadPoolExecutor] = None
        if isinstance(self.router, SyncNodeAffinityRouter):
            num_workers = self.router.env_worker_num
            self._rollout_executor = ThreadPoolExecutor(max_workers=num_workers)
            logger.info(f"[PolicyGroup] Initialized rollout executor with {num_workers} workers")

    def _submitter_for(self, policy) -> TaskSubmitter:
        """Return the appropriate TaskSubmitter for a given policy."""
        if policy in self.policy_list[PolicyRole.TRAIN]:
            return self._train_task_submitter
        return self._eval_task_submitter

    @property
    def eval_list(self) -> List[BasePolicy | ActorHandle]:
        """Return the list of policies for evaluation

        Returns:
            List of evaluation policy instances.
        """
        return self.policy_list[PolicyRole.EVAL]

    @property
    def train_list(self) -> List[BasePolicy | ActorHandle]:
        """Return the list of policies for training

        Returns:
            List of training policy instances.
        """
        return self.policy_list[PolicyRole.TRAIN]

    def init(
        self,
        train_config: TrainConfig,
        env_meta: Optional[EnvMeta] = None,
        eval_config: Optional[Any] = None,
    ) -> None:
        """Initialize both training and evaluation policies."""
        self.init_eval(eval_config, env_meta)
        self.init_train(train_config, env_meta)

    def init_placement_info(self, is_colocated: bool = False) -> None:
        """Initialize the placement info of the policy group with the consideration of given
        policy config.

        `placement_info` is a dict that mapping from ray_node_id to a dict of training/evaluation
        policy (actor).
        """
        # build transfer info
        if is_colocated:
            self.transfer_info = build_transfer_info_for_colocated(self.train_list, self.eval_list)
            return

        for policy in self.train_list:
            if InternalFlag.REMOTE_TRAIN:
                node_id = ray.get(policy._get_node_id.remote())
            else:
                node_id = "local"
            if node_id not in self.placement_info:
                self.placement_info[node_id] = {
                    "train_policies": [],
                    "eval_policies": [],
                }
            self.placement_info[node_id]["train_policies"].append(policy)

        for policy in self.eval_list:
            if InternalFlag.REMOTE_EVAL:
                node_id = ray.get(policy._get_node_id.remote())
            else:
                node_id = "local"
            if node_id not in self.placement_info:
                self.placement_info[node_id] = {
                    "train_policies": [],
                    "eval_policies": [],
                }
            self.placement_info[node_id]["eval_policies"].append(policy)

        logger.info(f"placement_info: {self.placement_info}")

        self.transfer_info = build_transfer_info_for_disaggregated(
            self.placement_info, self.config.weight_buffer.buffer_strategy
        )

    def init_weight_buffer(self) -> None:
        """Initialize the weight buffer based on the configured strategy.

        The weights buffer is created by the buffer strategy:
        None: no weight buffer
        Double: each eval policy has its own weight buffer
        Shared: shared weight buffer between eval policies on the same node
        Sharded: not supported now

        Raises:
            ValueError: If the weight buffer strategy is unsupported.
        """

        self.weight_buffer_strategy = self.config.weight_buffer.buffer_strategy
        if self.weight_buffer_strategy == "None":
            pass
        elif self.weight_buffer_strategy == "Double":
            for policy in self.eval_list:
                self._eval_task_submitter.submit(policy.init_weight_buffer, _block=True)
        elif self.weight_buffer_strategy == "Shared":
            self.shared_weight_buffer_map = {}
            for node_id, policies_info in self.placement_info.items():
                if len(policies_info["eval_policies"]) == 0:
                    continue
                shared_weight_buffer = build_weight_buffer(
                    self.config.weight_buffer.type,
                    self.config.weight_buffer,
                    node_id=node_id,
                )
                for policy in policies_info["eval_policies"]:
                    self._eval_task_submitter.submit(policy.init_weight_buffer, shared_weight_buffer, _block=True)
                self.shared_weight_buffer_map[node_id] = {
                    "shared_weight_buffer": shared_weight_buffer,
                    "eval_policies": policies_info["eval_policies"],
                }
        else:
            raise ValueError(f"Unsupported weight buffer strategy: {self.weight_buffer_strategy}")

    def init_comm_group(self, backend: str = "nccl", is_colocated: bool = False) -> None:
        """Initialize communication groups for distributed training.

        Sets up distributed environment, intra-node groups, weight transfer
        groups, and DDP communication groups.

        Args:
            backend: Communication backend ('nccl' or 'gloo').
            is_colocated: If True, initialize only for training policies.
        """
        if not self.train_list and not self.eval_list:
            return

        if is_colocated:
            all_policies = self.train_list
        else:
            all_policies = self.train_list + self.eval_list

        # init distributed environment
        actors_by_node = defaultdict(list)

        for policy in all_policies:
            node_id = self._submitter_for(policy).submit(policy._get_node_id, _block=True)
            actors_by_node[node_id].append(policy)

        # Sort actors by node and gpu id
        sorted_actors = []
        local_rank_list = []
        local_world_size_list = []
        for node_id in actors_by_node:
            actors_by_node[node_id] = sorted(
                actors_by_node[node_id],
                key=lambda x: self._submitter_for(x).submit(x._get_gpu_ids, _block=True)[0],
            )
            sorted_actors.extend(actors_by_node[node_id])
            local_world_size = len(actors_by_node[node_id])
            local_rank_list.extend(range(local_world_size))
            local_world_size_list.extend([local_world_size] * local_world_size)

        # Choose master from rank-0 actor's node to match env:// requirement
        master_addr, master_port = ray.get(sorted_actors[0]._get_addr_and_port.remote())
        world_size = len(sorted_actors)
        ref = [
            policy.init_distributed_env.remote(
                rank=rank,
                world_size=world_size,
                local_rank=local_rank_list[rank],
                local_world_size=local_world_size_list[rank],
                master_addr=master_addr,
                master_port=master_port,
                backend=backend,
            )
            for rank, policy in enumerate(sorted_actors)
        ]
        ray.get(ref)

        # init intra node communication group
        ref = []
        for node_actors in actors_by_node.values():
            intra_node_ranks = [ray.get(policy.get_rank.remote()) for policy in node_actors]
            ref.extend(
                [
                    policy.init_single_comm_group.remote(intra_node_ranks, ParallelMode.INTRA_NODE, backend)
                    for policy in all_policies
                ]
            )
        ray.get(ref)
        ray.get([policy.dist_barrier.remote() for policy in all_policies])

        # init weight transfer group
        if self.weight_buffer_strategy != "None" and self.transfer_info:
            ref.clear()
            for node_id, transfer_info in self.transfer_info.items():
                for transfer_info_item in transfer_info:
                    if len(transfer_info_item["receiver"]) == 0:
                        continue
                    weight_transfer_ranks = [
                        ray.get(policy.get_rank.remote())
                        for policy in [transfer_info_item["sender"]] + transfer_info_item["receiver"]
                    ]
                    ref.extend(
                        [
                            policy.init_single_comm_group.remote(
                                weight_transfer_ranks,
                                ParallelMode.WEIGHT_TRANSFER,
                                backend,
                            )
                            for policy in all_policies
                        ]
                    )
            ray.get(ref)
            ray.get([policy.dist_barrier.remote() for policy in all_policies])

        # if no train policy, only init eval policy
        if not self.train_list:
            return

        # init ddp communication group
        ddp_train_ranks = [ray.get(policy.get_rank.remote()) for policy in self.train_list]
        ref = [
            policy.init_single_comm_group.remote(ddp_train_ranks, ParallelMode.TRAIN_DATA_PARALLEL, backend)
            for policy in all_policies
        ]
        ray.get(ref)
        ray.get([policy.dist_barrier.remote() for policy in all_policies])

    def reset_training_state(
        self, train_config: TrainConfig, env_meta: Optional[Any] = None, seed: Optional[int] = None
    ) -> None:
        """Reset training policies after warm_up."""
        # Submit all policy reset tasks in parallel (non-blocking) to avoid DDP deadlock
        refs = [
            self._train_task_submitter.submit(policy.reset_training_state, train_config, env_meta, seed, _block=False)
            for policy in self.train_list
        ]
        # Wait for all tasks to complete (returns ObjectRef if Ray actor, otherwise direct result)
        if refs and isinstance(refs[0], ray.ObjectRef):
            ray.get(refs)
            # Synchronize only within train DDP group (don't use global barrier to avoid deadlock with eval ranks)
            ray.get([policy.dist_barrier.remote(ParallelMode.TRAIN_DATA_PARALLEL) for policy in self.train_list])

    def print_timing_summary(self, reset: bool = False) -> None:
        """
        Print the timing summary of the policy group.
        """
        for policy in self.train_list + self.eval_list:
            self._submitter_for(policy).submit(policy.print_timing_summary, reset, _block=True)

    def shutdown(self) -> None:
        """Shutdown the policy group and cleanup resources."""
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=False)
            self._rollout_executor = None

    def rollout_batch(self, batched_env_ret: BatchedData) -> BatchedData:
        """Perform batched rollout across eval policies.

        Args:
            batched_env_ret: Batched environment returns.

        Returns:
            BatchedData containing policy responses.
        """
        policy_resp_list = []
        if self.rollout_mode == "async":
            # with load balancing, assign requests to the actor with the least load
            current_loads = ray.get([p.get_num_requests.remote() for p in self.eval_list])
            assignments = self.router.assign(current_loads, len(batched_env_ret))
            for env_ret, idx in zip(batched_env_ret.values(), assignments):
                policy = self.eval_list[idx]
                # use the asynchronous rollout method to enable concurrent requests on the actor
                policy_resp_list.append(policy._rollout_async.remote(env_ret))
        else:
            env_ids = batched_env_ret.ids()
            env_rets = batched_env_ret.values()

            if isinstance(self.router, SyncNodeAffinityRouter):
                # For node_affinity router: use pre-initialized ThreadPoolExecutor to enable
                # concurrent scheduling across different nodes. The router uses per-node locks
                # internally, so max parallelism is limited by the number of nodes.
                def _submit_rollout(env_id: str, env_ret: Any) -> Any:
                    policy = self.router.select_policy(env_id)
                    return self._eval_task_submitter.submit(policy._rollout, env_ret)

                policy_resp_list = list(self._rollout_executor.map(_submit_rollout, env_ids, env_rets))
            else:
                # For simple router: sequential execution is more efficient because
                # all selections are serialized by a single global lock anyway.
                # Using ThreadPoolExecutor would only add overhead without benefit.
                for env_id, env_ret in zip(env_ids, env_rets):
                    policy = self.router.select_policy(env_id)
                    policy_resp_list.append(self._eval_task_submitter.submit(policy._rollout, env_ret))

        return BatchedData(batched_env_ret.ids(), policy_resp_list)

    def postprocess(
        self,
        batched_env_ret: Optional[BatchedData] = None,
        batched_policy_resp: Optional[BatchedData] = None,
    ) -> BatchedData:
        """Submit postprocess tasks to eval policies.

        Args:
            batched_env_ret: Batched environment returns.
            batched_policy_resp: Batched policy responses.

        Returns:
            BatchedData containing processed results.
        """
        batched_args = []
        if batched_env_ret is not None:
            batched_args.append(batched_env_ret.values())
        if batched_policy_resp is not None:
            batched_args.append(batched_policy_resp.values())

        env_ids = batched_env_ret.ids() if batched_env_ret is not None else batched_policy_resp.ids()
        zipped_args = list(zip(*batched_args))

        if isinstance(self.router, SyncNodeAffinityRouter) and self._rollout_executor is not None:
            # Parallel postprocess across nodes, mirroring rollout_batch's approach

            def _submit_postprocess(env_id: str, args: tuple) -> Any:
                policy = self.router.select_policy(env_id)
                return self._eval_task_submitter.submit(policy._postprocess, *args)

            processed_policy_resp_list = list(
                self._rollout_executor.map(_submit_postprocess, env_ids, zipped_args)
            )
        else:
            # Sequential fallback for simple router
            processed_policy_resp_list = []
            for env_id, args in zip(env_ids, zipped_args):
                policy = self.router.select_policy(env_id)
                processed_policy_resp_list.append(self._eval_task_submitter.submit(policy._postprocess, *args))

        return BatchedData(env_ids, processed_policy_resp_list)

    def update_dataset(self, sample_data: List) -> List[ray.ObjectRef]:
        """
        Update the dataset in the policy group by getting sampled data from the buffer.

        Args:
            sample_data: List of sampled data objects, one per train policy.
        """
        assert len(sample_data) == len(self.train_list)
        ret = []
        for policy, data in zip(self.train_list, sample_data):
            ret.append(self._train_task_submitter.submit(policy.update_dataset, data))
        return ret

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Train all training policies.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Training info from the first policy.
        """
        train_info = resolve_object(
            [self._train_task_submitter.submit(policy.train, *args, **kwargs) for policy in self.train_list]
        )
        return train_info[0]

    def push(self, policy_worker: BasePolicy | ActorHandle, role: PolicyRole) -> None:
        """Add new policy worker to existing group.

        Args:
            policy_worker (BasePolicy): Worker instance
            role (PolicyRole): Policy role, indicating training or evaluation.
        """

        self.policy_list[role].append(policy_worker)

    def pop(self, role: PolicyRole) -> BasePolicy | ActorHandle:
        """
        Pop the last policy worker from the list.
        """
        return self.policy_list[role].pop()

    def convert_train_to_eval(self, num: int) -> None:
        """Convert training policies to evaluation policies.

        Args:
            num: Number of policies to convert.
        """
        pass

    def convert_eval_to_train(self, num: int) -> None:
        """Convert evaluation policies to training policies.

        Args:
            num: Number of policies to convert.
        """
        pass

    def notify_update_weights(self) -> None:
        """Signal all eval policies that new weights are available."""
        for policy in self.eval_list:
            self._eval_task_submitter.submit(policy.notify_update_weights)

    def wait_for_eval_weight_update(self, timeout: float = 60.0) -> None:
        """Block until every eval policy finishes its background weight update.

        No-op when ``weight_buffer_strategy == "None"`` because weights are
        transferred synchronously and no background thread is involved.

        Args:
            timeout: Per-policy wait timeout in seconds.
        """
        refs = [
            self._eval_task_submitter.submit(policy.wait_for_weight_update_done, timeout)
            for policy in self.eval_list
        ]
        if refs and isinstance(refs[0], ray.ObjectRef):
            results = ray.get(refs)
            for i, ok in enumerate(results):
                if not ok:
                    logger.warning(f"[WeightConsistency] Eval policy {i} weight update timed out after {timeout}s")

    def verify_eval_weight_consistency(self, rtol: float = 1e-3) -> bool:
        """Wait for eval weight updates and verify they match the first train policy.

        For each eval policy, per-parameter L2 norms are collected and compared
        against those of the first train policy.  A relative tolerance *rtol* is
        applied: a mismatch is reported when

            |train_norm - eval_norm| / max(|train_norm|, 1e-8) > rtol

        Args:
            rtol: Relative tolerance for per-parameter norm comparison.

        Returns:
            ``True`` if every eval policy is consistent with the train policy,
            ``False`` if any mismatch is detected.
        """
        if self.weight_buffer_strategy != "None":
            self.wait_for_eval_weight_update()

        ref_fingerprint: dict = self._train_task_submitter.submit(self.train_list[0].get_param_fingerprint, _block=True)

        all_consistent = True
        for i, policy in enumerate(self.eval_list):
            eval_fingerprint: dict = self._eval_task_submitter.submit(policy.get_param_fingerprint, _block=True)
            mismatches = []
            for key, train_norm in ref_fingerprint.items():
                if key not in eval_fingerprint:
                    mismatches.append(f"{key}: missing in eval policy")
                    continue
                eval_norm = eval_fingerprint[key]
                base = max(abs(train_norm), 1e-8)
                rel_diff = abs(train_norm - eval_norm) / base
                if rel_diff > rtol:
                    mismatches.append(
                        f"{key}: train_norm={train_norm:.6f}, " f"eval_norm={eval_norm:.6f}, rel_diff={rel_diff:.2e}"
                    )

            if mismatches:
                all_consistent = False
                logger.warning(
                    f"[WeightConsistency] Eval policy {i} has {len(mismatches)} "
                    f"mismatched parameter(s) (showing first 10):"
                )
                for m in mismatches[:10]:
                    logger.warning(f"  {m}")
            else:
                logger.info(f"[WeightConsistency] Eval policy {i} weights match train policy.")

        return all_consistent

    def save_checkpoint(self, path: str) -> None:
        """Save the checkpoint of train policy.

        If multiple train policies exist, only save the checkpoint of the first one.

        Args:
            path (str): The path to save the checkpoint.
        """
        self._train_task_submitter.submit(self.train_list[0].save_checkpoint, path, _block=True)

    def send_weights(self) -> None:
        """Broadcast weights from train to eval policies.

        Send weights from train to eval policies based on the configured weight buffer strategy.
        """
        refs = []
        _transfer_descriptions = []
        for node_id, transfer_info in self.transfer_info.items():
            for transfer_info_item in transfer_info:
                # train_policy is an actor of WeightTransferManager
                train_policy = transfer_info_item["sender"]
                n_receivers = len(transfer_info_item["receiver"]) if isinstance(transfer_info_item["receiver"], list) else 1
                intra = transfer_info_item.get("intra_node", False)
                _transfer_descriptions.append(f"node={node_id},intra={intra},receivers={n_receivers}")
                if self.weight_buffer_strategy == "Shared" and transfer_info_item["intra_node"]:
                    shared_weight_buffer = self.shared_weight_buffer_map[node_id]["shared_weight_buffer"]
                    if InternalFlag.REMOTE_TRAIN:
                        refs.append(
                            train_policy.send_weights.remote(transfer_info_item["receiver"], shared_weight_buffer)
                        )
                    else:
                        train_policy.send_weights(transfer_info_item["receiver"], shared_weight_buffer)
                else:
                    if InternalFlag.REMOTE_TRAIN:
                        refs.append(train_policy.send_weights.remote(transfer_info_item["receiver"]))
                    else:
                        # for single process mode
                        train_policy.send_weights(transfer_info_item["receiver"])

        if len(refs) > 0:
            ray.get(refs)

    def sync_weights(self):
        """
        Broadcast the state dict to the eval policy and store into the
        weight_buffer for later loading.
        """
        # (warining): may have bug when params is offload to cpu (in warm_up)
        if self.weight_buffer_strategy == "None":
            refs = []
            for transfer_info_item in self.transfer_info["all"]:
                train_policy = transfer_info_item["sender"]
                eval_policy = transfer_info_item["receiver"]
                serialized_state_dict = train_policy.send_weights_ipc.remote()
                ref = self._eval_task_submitter.submit(
                    eval_policy.recv_weights_ipc, serialized_state_dict
                )
                refs.append(ref)
            resolve_object(refs)
        else:
            self.send_weights()
            self.notify_update_weights()

    def init_eval(self, eval_config: Optional[Any] = None, env_meta: Optional[EnvMeta] = None) -> List:
        """Initialize all evaluation policies.

        Args:
            eval_config: Evaluation configuration.
            env_meta: Environment metadata.

        Returns:
            List of initialization results.
        """
        resolve_object(
            [self._eval_task_submitter.submit(policy.init_eval, eval_config, env_meta) for policy in self.eval_list]
        )

    def init_train(self, train_config: Any, env_meta: Optional[EnvMeta] = None) -> List:
        """Initialize all training policies.

        Args:
            train_config: Training configuration.
            env_meta: Environment metadata.

        Returns:
            List of initialization results.
        """

        resolve_object(
            [self._train_task_submitter.submit(policy.init_train, train_config, env_meta) for policy in self.train_list]
        )

    def offload_eval_model(self):
        """
        Offload the eval model to cpu.
        """
        resolve_object([self._eval_task_submitter.submit(policy.offload_model) for policy in self.eval_list])

    def reload_eval_model(self):
        """
        Reload the eval model to gpu.
        """
        resolve_object([self._eval_task_submitter.submit(policy.reload_model) for policy in self.eval_list])

    def offload_model_optimizer(self):
        """
        Offload the model optimizer to cpu.
        """
        resolve_object([self._train_task_submitter.submit(policy.offload_optimizer) for policy in self.train_list])

    def offload_model_param_and_grad(self, offload_grad: bool = True, offload_optimizer: bool = True):
        """
        Offload the model parameters and gradients to cpu.
        """
        resolve_object(
            [
                self._train_task_submitter.submit(policy.offload_model_param_and_grad, offload_grad=offload_grad)
                for policy in self.train_list
            ]
        )
        if offload_optimizer:
            resolve_object([self._train_task_submitter.submit(policy.offload_optimizer) for policy in self.train_list])

    def reload_model_param_and_grad(self, load_grad: bool = True, load_optimizer: bool = True):
        """
        Reload the model parameters and gradients to gpu.
        """
        resolve_object(
            [
                self._train_task_submitter.submit(policy.reload_model_param_and_grad, load_grad=load_grad)
                for policy in self.train_list
            ]
        )
        if load_optimizer:
            resolve_object([self._train_task_submitter.submit(policy.reload_optimizer) for policy in self.train_list])


def build_transfer_info_for_disaggregated(
    placement_info: Dict[str, Dict[str, List]], weight_buffer_strategy: str
) -> tuple[Dict[str, List[str]], Dict[str, List[Dict]]]:
    """Build weight transfer info for disaggregated..

    Determines how weights should be transferred from train to eval policies
    based on their node placement and the weight buffer strategy.

    Args:
        placement_info: Dictionary mapping node IDs to policy lists.
        weight_buffer_strategy: Strategy for weight buffer management.

    Returns:
        A tuple containing:
            - router_map: Mapping from train nodes to receiver nodes.
            - transfer_info: Detailed transfer information per node.

    Raises:
        ValueError: If no train policy is found or unsupported strategy.
    """
    # init transfer info
    router_map = {}
    transfer_info = {}
    node_info = {"train_nodes": [], "eval_nodes": [], "mixed_nodes": []}

    # classify nodes into train/eval/mixed nodes
    for node_id, policies_info in placement_info.items():
        if len(policies_info["train_policies"]) == 0:
            node_info["eval_nodes"].append(node_id)
        elif len(policies_info["eval_policies"]) == 0:
            node_info["train_nodes"].append(node_id)
        else:
            node_info["mixed_nodes"].append(node_id)
            router_map[node_id] = [node_id]

    # process the router of the node which only has eval_policy
    if len(node_info["train_nodes"]) > 0:
        # if exist independent train_node，average assign eval_policy to train_node
        for idx, eval_node_id in enumerate(node_info["eval_nodes"]):
            train_node_id = node_info["train_nodes"][idx % len(node_info["train_nodes"])]
            if train_node_id not in router_map:
                router_map[train_node_id] = []
            router_map[train_node_id].append(eval_node_id)
    elif len(node_info["mixed_nodes"]) > 0:
        # if not exist independent train_node，average assign eval_policy to mixed_node
        for idx, eval_node_id in enumerate(node_info["eval_nodes"]):
            mixed_node_id = node_info["mixed_nodes"][idx % len(node_info["mixed_nodes"])]
            router_map[mixed_node_id].append(eval_node_id)
    else:
        raise ValueError("No train policy found")

    for node_id, router_list in router_map.items():
        train_policies = placement_info[node_id]["train_policies"]
        transfer_info[node_id] = []
        for idx, router_node_id in enumerate(router_list):
            # average assign receiver_node to the train_policy of sender_node
            transfer_info_item = {
                "sender": train_policies[idx % len(train_policies)],
                "receiver": [],
                "intra_node": False,
            }
            if router_node_id == node_id:
                transfer_info_item["intra_node"] = True
            if weight_buffer_strategy in ["None", "Double"]:
                transfer_info_item["receiver"].extend(placement_info[router_node_id]["eval_policies"])
            elif weight_buffer_strategy == "Shared":
                if router_node_id != node_id:
                    transfer_info_item["receiver"].append(placement_info[router_node_id]["eval_policies"][0])
            else:
                raise ValueError(f"Unsupported weight buffer strategy: {weight_buffer_strategy}")
            transfer_info[node_id].append(transfer_info_item)

    logger.debug(f"router_map: {router_map}")
    logger.debug(f"transfer_info: {transfer_info}")
    return transfer_info


def build_transfer_info_for_colocated(train_list: List[BasePolicy], eval_list: List[BasePolicy]):
    """
    Build transfer info for colocation.

    Args:
        train_list: List of train policies.
        eval_list: List of eval policies.

    Returns:
        Transfer info for colocation.
    """
    transfer_info_list = []
    assert len(train_list) >= len(eval_list), "train_list must have more policies than eval_list for colocated mode"
    # here we assume the train_list and eval_list are in the same order
    for idx, policy in enumerate(eval_list):
        transfer_info_item = {
            "sender": train_list[idx],
            "receiver": policy,
        }
        transfer_info_list.append(transfer_info_item)
    return {"all": transfer_info_list}
