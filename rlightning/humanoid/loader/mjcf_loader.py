import xml.etree.ElementTree as ETree
from collections import OrderedDict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


# recursively adding all nodes into the skel_tree
def _add_xml_node(
    node_names: List[str],
    local_translation: List[Sequence],
    parent_indices: List[int],
    local_rotation: List[Sequence],
    joints_range: List[Sequence],
    body_to_joint: Dict[str, Any],
    xml_node,
    parent_index,
    node_index,
) -> int:
    node_name = xml_node.attrib.get("name")
    # parse the local translation into float list
    pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
    quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
    node_names.append(node_name)
    parent_indices.append(parent_index)
    local_translation.append(pos)
    local_rotation.append(quat)
    curr_index = node_index
    node_index += 1
    all_joints = xml_node.findall("joint")  # joints need to remove the first 6 joints
    if len(all_joints) == 6:
        all_joints = all_joints[6:]

    if all_joints:
        for joint in all_joints:
            if not joint.attrib.get("range") is None:
                joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            else:
                if not joint.attrib.get("type") == "free":
                    joints_range.append([-np.pi, np.pi])

            body_to_joint[node_name] = joint.attrib.get("name")
    else:
        pass

    for next_node in xml_node.findall("body"):
        node_index = _add_xml_node(next_node, curr_index, node_index)

    return node_index


def load_mjcf(mjcf_path: str, num_dof: int, device: str = "cpu") -> Dict[str, Any]:
    """
    Load a MJCF file and return the corresponding Mujoco model.

    Args:
        mjcf_path (str): Path to the MJCF file.
        num_dof (int): Expected number of DoF joints in the MJCF.
        device (str): Target device for returned tensors.

    Returns:
        A dict of the following keys

            - node_names: List of node names in the model.
            - parent_indices: Tensor of parent indices for each node.
            - local_translation: Tensor of local translations for each node.
            - local_rotation: Tensor of local rotations for each node.
            - joints_range: Tensor of joint ranges.
            - body_to_joint: A dictionary mapping body names to joint names.
    """

    tree = ETree.parse(mjcf_path)

    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")

    if xml_world_body is None:
        raise ValueError("MJCF parsed incorrectly please verify it.")

    # assume this is the root
    xml_body_root = xml_world_body.find("body")

    if xml_body_root is None:
        raise ValueError("MJCF parsed incorrectly please verify it.")

    node_names = []
    parent_indices = []
    local_translation = []
    local_rotation = []
    joints_range = []
    body_to_joint = OrderedDict()

    _add_xml_node(
        node_names,
        local_translation,
        parent_indices,
        local_rotation,
        joints_range,
        body_to_joint,
        xml_body_root,
        -1,
        0,
    )

    assert len(joints_range) == num_dof

    return {
        "node_names": node_names,
        "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)).to(device),
        "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)).to(
            device
        ),
        "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)).to(device),
        "joints_range": torch.from_numpy(np.array(joints_range)).to(device),
        "body_to_joint": body_to_joint,
    }
