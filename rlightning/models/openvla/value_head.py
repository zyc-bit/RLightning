# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from transformers.generation import LogitsProcessor


class VLALogitsProcessor(LogitsProcessor):
    """
    Only sample the action token.
    """

    def __init__(self, action_num_bins, filter_value: float = -torch.inf):
        super().__init__()
        self.action_num_bins = action_num_bins
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        - scores: [B, vocab-size]
        """
        scores_processed = scores.clone()
        # scores_processed[:, :-self.action_num_bins] = self.filter_value
        scores_processed[:, : 32000 - self.action_num_bins] = self.filter_value
        scores_processed[:, 32000:] = self.filter_value
        return scores_processed


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",  # 'relu' or 'gelu'
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # tmp fix
                    nit_nonlin = "relu" if nonlinearity == "gelu" else nonlinearity
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nit_nonlin)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    value_head = ValueHead(
        input_dim=4096, hidden_sizes=(512, 128), output_dim=1, activation="gelu", bias_last=False
    )
    x = torch.randn(1, 4096)
    print(value_head(x))
