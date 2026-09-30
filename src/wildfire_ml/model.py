import torch
from torch import nn


class TinyWildfireNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=12,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        return x
