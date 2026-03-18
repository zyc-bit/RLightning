# Third-Party Notices

This repository includes code copied from or derived from third-party projects.
The items below record the immediate upstream source that is currently confirmed for the
Git-tracked files in this repository.

## RLinf

- Upstream repository: <https://github.com/RLinf/RLinf>
- Upstream repository license: Apache License 2.0
- Local usage: copied and adapted code listed below

### Copied directories

- `examples/openvla_ppo/maniskill/`
  Source: `RLinf/examples/openvla_ppo/maniskill/`
  Notes: the currently tracked contents under this path are Python source files that retain the
  RLinf copyright and Apache 2.0 license headers.

- `third_party/openvla/`
  Source: `RLinf/third_party/openvla/`
  Notes: vendored implementation copied from RLinf.

### Copied or adapted files

- `rlightning/env/maniskill_env.py`
  Source: `RLinf/rlightning/env/maniskill_env.py`
  Status: copied from RLinf and modified in this repository

- `rlightning/models/openvla/openvla_model.py`
  Source: `RLinf/rlightning/models/openvla/openvla_model.py`
  Status: copied from RLinf and modified in this repository

- `rlightning/models/openvla/value_head.py`
  Source: `RLinf/rlightning/models/openvla/value_head.py`
  Status: copied from RLinf

- `rlightning/policy/utils/losses.py`
  Source: `RLinf/rlightning/policy/utils/losses.py`
  Status: copied from RLinf

- `rlightning/policy/utils/utils.py`
  Source: `RLinf/rlightning/policy/utils/utils.py`
  Status: copied from RLinf

## Notes

- The list above reflects files and directories present in the current working tree.
- If the exact upstream commit or tag used for the copy is known, add it here for a stronger audit trail.
- Non-tracked local assets or caches are not inventoried here.
- If you later identify separate original licenses for nested third-party materials, preserve those notices alongside this file.
