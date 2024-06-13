import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size, bias=False),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(),
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class Caps(nn.Module):
    def __init__(self, num_capsules, num_channels, num_classes):
        super().__init__()
        self.num_capsules = num_capsules  # num_capsules = width x height
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.linear = nn.Linear(num_capsules, num_classes, bias=False)
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        # [..., channels, height, width] -> [..., channels, width x height]
        x = x.flatten(-2)
        # [..., channels, width x height] -> [..., channels, num_classes]
        x = self.linear(x)
        # batchnorm channel features (bn classes or channels*classes instead?)
        x = self.activation(self.bn(x))
        # [..., channels, num_classes] -> [..., num_classes]
        branch_logits = x.sum(-2)
        return branch_logits


class CapsNet(nn.Module):
    """Reproduction of net architecture from "No Routing Needed Between Capsules"
    by Adam Byerly, Tatiana Kalganova, Ian Dear
    https://arxiv.org/abs/2001.09136

    choices/changes made from original paper:
    1. ReLU -> LeakyReLU
    2. Z-Derived Capsules
    3. BatchNorm1d on channel features (classes or channels*classes instead?)
    4. Merge branch_logits with sum rather than learnable weights
    """

    def __init__(self, w, h, c, num_classes, kernel_size=3):
        super().__init__()
        self.w, self.h, self.c = w, h, c
        self.convs = nn.ModuleList(
            [
                Conv([c, 32, 48, 64], kernel_size=kernel_size),
                Conv([64, 80, 96, 112], kernel_size=kernel_size),
                Conv([112, 128, 144, 160], kernel_size=kernel_size),
            ]
        )
        n_caps = []
        for c in self.convs:
            gap = (kernel_size - 1) * len(c.convs)
            w, h = w - gap, h - gap
            n_caps.append(w * h)
        self.caps = nn.ModuleList(
            [Caps(nc, c.channels[-1], num_classes) for c, nc in zip(self.convs, n_caps)]
        )

    def forward(self, x):
        logits = 0
        for conv, cap in zip(self.convs, self.caps):
            x = conv(x)
            logits += cap(x)
        return logits

    def probabilities(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)