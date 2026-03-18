from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NatureCNN(nn.Module):
    """
    Construct a CNN with given dummy RGB
    """

    def __init__(
        self,
        standard_input_size: Tuple[int, int] = (84, 84),
        in_channels: int = 3,
        image_format: str = "CHW",
    ):
        super().__init__()

        feature_size = 256

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
            rgb = torch.rand((1, in_channels) + standard_input_size)
            n_flatten = cnn(rgb).shape[1]

        fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        self.model = nn.Sequential(cnn, fc)

        self.image_format = image_format
        self.in_channels = in_channels
        self.out_feature_dim = feature_size
        self.standard_input_size = standard_input_size

        self.transform_pipeline = transforms.Compose(
            [
                transforms.Resize(standard_input_size, interpolation=F.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def forward(self, obs: Union[torch.Tensor, np.ndarray]):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dtype != torch.float32:
            obs = obs.float()

        if self.image_format == "HWC" and len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif self.image_format == "CHW" and len(obs.shape) == 4:
            pass
        else:
            raise ValueError(
                f"Unsupported image_format {self.image_format} or obs shape {obs.shape}"
            )

        obs = obs.to(next(self.parameters()).device)

        if obs.max() > 1.0:
            obs = obs / 255.0

        obs = self.transform_pipeline(obs)
        features = self.model(obs)

        return features
