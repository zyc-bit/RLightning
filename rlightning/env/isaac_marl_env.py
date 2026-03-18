# from typing import List, Dict, Optional, Callable

# try:
#     from isaaclab.app import AppLauncher
# except ImportError as e:
#     raise e

# import time

# import gymnasium as gym
# import numpy as np
# import torch

# from rlightning.types import EnvRet, MultiAgentEnvRet, MultiAgentPolicyResponse
# from rlightning.utils.profiler import profiler
# from rlightning.utils.registry import ENVS
# from rlightning.utils.config import EnvConfig

# from rlightning.utils.logger import get_logger

# from .base_env import BaseEnv
# from .utils.utils import default_env_preprocess_fn

# logger = get_logger(__name__)


# @ENVS.register("isaac_marl")
# class IsaacMarlEnv(BaseEnv):

#     def __init__(
#         self, config: EnvConfig, env_worker_index: int = 0, preprocess_fn: Optional[Callable] = default_env_preprocess_fn
#     ):
#         super().__init__(config, preprocess_fn)

#         app_launcher = None
#         simulation_app = None

#         try:
#             device = config.env_kwargs.pop("device", f"cuda:{torch.cuda.current_device()}")
#             headless = config.env_kwargs.pop("headless", True)
#             app_launcher = AppLauncher(headless=headless, device=device)
#             simulation_app = app_launcher.app

#             from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

#             import unitree_rl_lab.tasks
#             from unitree_rl_lab.envs import mix_marl_env
#         except Exception as e:
#             logger.warning(f"no module named 'unitree_rl_lab' found")
#             raise e

#         env_cfg = load_cfg_from_registry(config.task, "env_cfg_entry_point")

#         assert isinstance(env_cfg, mix_marl_env.MixMarlEnvCfg), type(env_cfg)

#         self.has_critic = None
#         self.env: gym.Env = gym.make(config.task, cfg=env_cfg, render_mode=None)
#         self.simulation_app = simulation_app

#     @property
#     def possible_agents(self) -> List[str]:
#         """A list of possible agent ids

#         Returns:
#             List[str]: A list of strings
#         """

#         return self.unwrapped.possible_agents

#     @property
#     def unwrapped(self):
#         """Returns the base environment of the wrapper.

#         This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
#         """
#         return self.env.unwrapped

#     def get_observation_space(self) -> Dict[str, gym.Space]:
#         """Retrieve a dict of agent observation space

#         Returns:
#             Dict[str, gym.Space]: A dict of agent observation space
#         """

#         return self.unwrapped.observation_spaces

#     def get_action_space(self) -> Dict[str, gym.Space]:
#         """Retrive a dict of agent action spaces

#         Returns:
#             Dict[str, gym.Space]: A dict of agent actions spaces
#         """

#         return self.unwrapped.action_spaces

#     @profiler.timer_wrap("env_reset", level="debug")
#     def reset(self, *args, **kwargs) -> EnvRet:
#         obs_dict, _ = self.env.reset()

#         recompute_obs = {}
#         recompute_obs["policy"] = {agent: obs["policy"] for agent, obs in obs_dict.items()}
#         if obs_dict[self.possible_agents[0]].get("critic", None) is not None:
#             self.has_critic = True
#             recompute_obs["critic"] = {agent: obs["critic"] for agent, obs in obs_dict.items()}
#         else:
#             self.has_critic = False

#         return MultiAgentEnvRet(
#             env_id=self.env_id,
#             observation=recompute_obs["policy"],
#             last_reward={
#                 agent: torch.zeros(
#                     self.unwrapped.num_envs, dtype=torch.float32, device=self.unwrapped.device
#                 )
#                 for agent in self.possible_agents
#             },
#             last_terminated={
#                 agent: torch.zeros(
#                     self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
#                 ).bool()
#                 for agent in self.possible_agents
#             },
#             last_truncated={
#                 agent: torch.zeros(
#                     self.unwrapped.num_envs, dtype=torch.long, device=self.unwrapped.device
#                 ).bool()
#                 for agent in self.possible_agents
#             },
#             last_info={"observations": recompute_obs},
#         )

#     @profiler.timer_wrap("env_step", level="debug")
#     def step(self, policy_resp: MultiAgentPolicyResponse) -> EnvRet:
#         """Step the environment with the given policy response.

#         The returned EnvRet has items as a series of dict (mapping from agent ids to entries)

#         Args:
#             policy_resp (PolicyResponse): The response from the policy, containing the action to take

#         Returns:
#             EnvRet: The return from the environment after taking the action
#         """

#         actions = self._preprocess_fn(policy_resp)

#         # record step information
#         obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
#         # compute dones for compatibility with RSL-RL
#         dones = {
#             agent: (terminated[agent] & truncated[agent]).to(dtype=torch.long)
#             for agent in self.possible_agents
#         }
#         truncated = sum(truncated.values()) > 0

#         dones = sum(dones.values()) > 0

#         # move extra observations to the extras dict
#         recompute_obs = {}
#         recompute_obs["policy"] = {agent: obs["policy"] for agent, obs in obs_dict.items()}
#         if self.has_critic:
#             recompute_obs["critic"] = {agent: obs["critic"] for agent, obs in obs_dict.items()}
#         extras["observations"] = recompute_obs
#         # move time out information to the extras dict
#         # this is only needed for infinite horizon tasks
#         if not self.unwrapped.cfg.is_finite_horizon:
#             extras["time_outs"] = truncated

#         return MultiAgentEnvRet(
#             env_id=self.env_id,
#             observation=recompute_obs["policy"],
#             last_reward=rew,
#             last_terminated=dones,
#             last_truncated=truncated,
#         )

#     def close(self):
#         self.env.close()
#         self.simulation_app.close()
