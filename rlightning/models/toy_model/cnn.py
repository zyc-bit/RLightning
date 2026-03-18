from typing import Union

import numpy as np
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
    a simple CNN model for test purpose
    """

    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()
            n_flatten = cnn(dummy).shape[1]

        fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, obs: Union[torch.Tensor, np.ndarray]):
        encoded = []
        for key, extractor in self.extractors.items():
            x = obs[key]
            if key == "rgb":
                x = x.float().permute(0, 3, 1, 2) / 255.0
            encoded.append(extractor(x))
        return torch.cat(encoded, dim=1)
