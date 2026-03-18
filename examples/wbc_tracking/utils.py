from typing import Dict, List

import torch


def episode_postprocess_fn(episode_buffer: List[Dict]) -> Dict:
    assert (
        "_extra" not in episode_buffer
    ), "_extra should not be postprocessed here, check the configuration of related preprocess_fn"

    data = {}

    # firstly we fill 'next_observation' into episode_buffer manually, and do not forget to ensure the length
    #   consistency by duplicating the last observation
    episode_buffer["next_observation"] = episode_buffer["observation"][1:] + [episode_buffer["observation"][-1]]

    for k, v in episode_buffer.items():
        # skip info by default
        if "info" in k:
            continue

        # items which tagged with 'last_' should ignore the first frame
        if k.startswith("last_"):
            k = k[5:]
            v = v[1:]  # support for both 1D and 2D arrays (vector env)
        else:  # otherwise, ignore the last frame to treat them as current states
            v = v[:-1]

        data[k] = torch.stack(v, dim=0)

    return data
