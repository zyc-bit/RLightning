import ray
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from rlightning.buffer.base_buffer import DataBuffer
from rlightning.models.toy_model import NatureCNN
from rlightning.policy.base_policy import BasePolicy
from rlightning.utils.utils import to_device


class SimpleSupervisedPolicy(BasePolicy):
    """A simple supervised policy with a CNN encoder for testing purposes."""

    def __init__(self, config, role_type):
        super().__init__(config, role_type)

        super().init()

        self._is_ready = True

    def is_ready(self):
        return self._is_ready

    def init_train(self, train_config=None, env_meta=None):
        env_metadata = self.config.env_metadata
        sample_obs = env_metadata["obs"]
        action_dim = env_metadata["act_dim"]

        self.encoder = NatureCNN(sample_obs)
        self.actor_mean = nn.Linear(self.encoder.out_features, action_dim)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        self._find_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def construct_network(self, env_meta=None, *args, **kwargs):
        raise NotImplementedError

    def setup_optimizer(self, optim_cfg):
        raise NotImplementedError

    def forward(self, obs):
        x = self.encoder(obs)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def get_action(self, obs):
        return self.forward(obs)

    def get_value(self, obs):
        pass

    def get_action_value(self, obs):
        pass

    def get_action_mean(self, obs):
        x = self.encoder(obs)
        return self.actor_mean(x)

    def rollout_step(self, env_ret, **kwargs):
        raise NotImplementedError

    def postprocess(self, env_ret=None, policy_resp=None):
        raise NotImplementedError

    def update_dataset(self, data):
        raise NotImplementedError

    def train(self, sl_buffer: DataBuffer):
        self.sl_batch_size = 128
        data_length = ray.get(sl_buffer.get_data_length.remote())
        if data_length < self.sl_batch_size:
            return
        obs, act = ray.get(sl_buffer.sample.remote(self.sl_batch_size))
        obs = to_device(obs, torch.device("cuda"))
        act = to_device(act, torch.device("cuda"))

        pred = self.get_action_mean(obs)
        loss = F.mse_loss(pred, act)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self, obs):
        return self.get_action(obs)

    def get_trainable_parameters(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True, assign=False):
        raise NotImplementedError
