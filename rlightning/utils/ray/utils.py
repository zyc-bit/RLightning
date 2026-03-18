"""Ray utility functions for distributed computing.

This module provides utility functions for working with Ray actors,
object references, and cluster resources.
"""

from collections.abc import Sequence
from typing import Any, Dict, List, Mapping, Union

import ray
from ray._private.state import actors

from rlightning.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_object(
    obj: Union[ray.ObjectRef, Sequence[ray.ObjectRef], Mapping[Any, ray.ObjectRef], Any],
) -> Any:
    """Top-level resolution of Ray ObjectRef(s).

    Resolves ObjectRef instances to their actual values. No recursion
    into nested containers - only handles top-level ObjectRefs.

    Args:
        obj: An ObjectRef, a non-empty Sequence of ObjectRef, or a non-empty
             Mapping[Any, ObjectRef].

    Returns:
        The resolved value or a container of the same type when batch-resolved;
        otherwise, returns the input unchanged.
    """
    if isinstance(obj, ray.ObjectRef):
        return ray.get(obj)

    elif isinstance(obj, Sequence) and len(obj) > 0:
        if isinstance(obj, (str, bytes)):
            return obj
        elif all(isinstance(o, ray.ObjectRef) for o in obj):
            return type(obj)(ray.get(list(obj)))
        elif any(isinstance(o, ray.ObjectRef) for o in obj):
            return type(obj)(resolve_object(o) for o in obj)

    elif isinstance(obj, Mapping) and len(obj) > 0:
        if all(isinstance(v, ray.ObjectRef) for v in obj.values()):
            keys = list(obj.keys())
            vals = ray.get(list(obj.values()))
            return type(obj)(zip(keys, vals))
        elif any(isinstance(v, ray.ObjectRef) for v in obj.values()):
            return type(obj)((k, resolve_object(v)) for k, v in obj.items())

    return obj


def get_cluster_nodes() -> List[Dict[str, Any]]:
    """Get the nodes in the cluster.

    Returns:
        List of dictionaries containing node information, only including
        nodes that are currently alive.
    """
    ray_nodes = [n for n in ray.nodes() if n["Alive"]]
    return ray_nodes


def get_cluster_resources() -> Dict[str, Any]:
    """Get cluster resources with per-node details.

    Returns a summary compatible with existing callers and adds detailed
    per-node resource information.

    Returns:
        Dictionary containing:
            - n_nodes: Number of alive nodes in the cluster.
            - node_id_to_resources: Mapping from node ID to resource info
              (ip, CPU count, GPU count).
    """
    # List all alive nodes and their declared total resources
    ray_nodes = [n for n in ray.nodes() if n["Alive"]]

    node_id_to_resources = {}

    for n in ray_nodes:
        node_id = n.get("NodeID")
        ip = n.get("NodeManagerAddress")

        # Ray versions may expose either "Resources" or "TotalResources"
        resources_total = n.get("Resources") or n.get("TotalResources") or {}

        cpu_total = int(resources_total.get("CPU", 0))
        gpu_total = int(resources_total.get("GPU", 0))

        info = {
            "ip": ip,
            "CPU": cpu_total,
            "GPU": gpu_total,
        }
        node_id_to_resources[node_id] = info

    return {
        "n_nodes": len(ray_nodes),
        "node_id_to_resources": node_id_to_resources,
    }


def get_cluster_actor_info() -> Dict[str, List[str]]:
    """Get actor info for all actors in the cluster.

    Collects information about all non-dead actors and groups them by node.
    Logs the actor counts per node for debugging.

    Returns:
        Dictionary mapping node IDs to lists of actor names on that node.
    """
    node_map = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]}
    actor_meta = actors()

    node_actors = {}
    for _, meta in actor_meta.items():
        node_id = meta["Address"]["NodeID"]
        if meta["State"] != "DEAD":
            node_actors.setdefault(node_id, []).append(meta["Name"])

    for nid, actor_list in node_actors.items():
        logger.info(f"Node {nid[:8]} @ {node_map[nid]} has {len(actor_list)} actors: {actor_list}")
    return node_actors
