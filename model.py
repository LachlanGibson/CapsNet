import torch
import torch.nn as nn


class HVCapsNet(nn.Module):
    """Reproduction of net architecture from "No Routing Needed Between Capsules"
    by Adam Byerly, Tatiana Kalganova, Ian Dear
    https://doi.org/10.48550/arXiv.2001.09136
    https://github.com/AdamByerly/BMCNNwHFCs

    choices made from original paper:
    1. Z-Derived Capsules
    2. Merge branch_logits with sum rather than learnable weights
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
        for b in self.convs:
            gap = (kernel_size - 1) * len(b.convs)
            w, h = w - gap, h - gap
            n_caps.append(w * h)
        self.caps = nn.ModuleList(
            [Caps(b.channels[-1], nc, num_classes) for b, nc in zip(self.convs, n_caps)]
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


class Conv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size, bias=False),
                    nn.BatchNorm2d(channels[i + 1], eps=0.001, momentum=0.0003),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class Caps(nn.Module):
    def __init__(self, num_channels, num_capsules, num_classes):
        super().__init__()
        self.num_capsules = num_capsules  # num_capsules = width x height
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.weights = nn.Parameter(
            data=torch.Tensor(num_channels, num_capsules, num_classes),
            requires_grad=True,
        )
        stdv = num_capsules**-0.5
        self.weights.data.uniform_(-stdv, stdv)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_channels * num_classes, eps=0.001, momentum=0.0003)

    def forward(self, x):
        # [..., channels, height, width] -> [..., channels, 1, width x height]
        x = x.flatten(-2).unsqueeze(-2)
        # [..., channels, 1, capsules] @ [channels, capsules, classes] -> [..., channels, 1, classes]
        x = x @ self.weights
        # [..., channels, 1, classes] -> [..., channels*classes]
        x = x.flatten(-3)
        # batchnorm channels*classes
        x = self.activation(self.bn(x))
        # [..., channels*classes] -> [..., channels, classes]
        x = x.view(*x.shape[:-1], self.num_channels, self.num_classes)
        # [..., channels, classes] -> [..., classes]
        branch_logits = x.sum(-2)
        return branch_logits
