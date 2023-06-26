from more_itertools import pairwise
from typing import Optional, Tuple

import torch.nn as nn

from minipi.utils.misc import infer_fc_input_dim


class ImpalaBlock(nn.Module):
    """
    Conv sequence in Impala CNN
    """

    def __init__(self, in_channels, depth):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=depth, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
        )
        self.res2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """
    CNN network used in the paper
    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        depths: Tuple[int, ...] = (16, 32, 32),
        hiddens: Optional[Tuple[int, ...]] = (256,),
    ):
        super().__init__()
        c, h, w = input_shape

        depths = (c, *depths)
        self.convs = nn.Sequential(
            *(ImpalaBlock(n_in, depth) for n_in, depth in pairwise(depths)),
            nn.Flatten(),
            nn.ReLU(),
        )
        output_dim = infer_fc_input_dim(self.convs, input_shape)
        if hiddens is not None:
            layers = []
            for n_channel in hiddens:
                layers.append(nn.Linear(in_features=output_dim, out_features=n_channel))
                layers.append(nn.ReLU())
                output_dim = n_channel
            self.fc = nn.Sequential(*layers)
        self.hiddens = hiddens
        self.output_dim = output_dim
        self.is_recurrent = False

    def forward(self, x):
        x = self.convs(x)
        if self.hiddens is not None:
            x = self.fc(x)
        return x
