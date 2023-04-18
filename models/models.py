from typing import *
import collections

import torch
import torch.nn as nn
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear, MetaConv2d, MetaBatchNorm2d

class Conv4(MetaModule):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super(Conv4, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = 64

        self.convs = MetaSequential(
            self.convBlock(self.in_channels, self.hidden_size, 3),
            self.convBlock(self.hidden_size, self.hidden_size, 3),
            self.convBlock(self.hidden_size, self.hidden_size, 3),
            self.convBlock(self.hidden_size, self.hidden_size, 3),
        )

        self.linear = MetaLinear(self.hidden_size, self.out_features)

    @classmethod
    def convBlock(
        cls, in_channels: int, out_channels: int, kernel_size: int
    ) -> MetaSequential:
        return MetaSequential(
            MetaConv2d(in_channels, out_channels, kernel_size, padding=1, stride=3),
            MetaBatchNorm2d(out_channels, momentum=1.0, track_running_stats=False),
            nn.ReLU(),
        )

    def forward(
        self, x: torch.Tensor, params: Optional[collections.OrderedDict] = None
    ) -> torch.Tensor:
        x_convs = self.convs(x, params=self.get_subdict(params, "convs"))
        prob = self.linear(
            x_convs.flatten(start_dim=1), params=self.get_subdict(params, "linear")
        )
        return prob