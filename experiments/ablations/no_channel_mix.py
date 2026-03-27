import argparse

import torch
import torch.nn as nn

from experiments.convmixer_models import Residual, build_patch_stem


class ConvMixerNoChannelMix(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 5,
        patch_size: int = 2,
        n_classes: int = 10,
    ) -> None:
        super().__init__()

        self.stem = build_patch_stem(dim=dim, patch_size=patch_size)

        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=dim,
                                out_channels=dim,
                                kernel_size=kernel_size,
                                groups=dim,
                                padding=kernel_size // 2,
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(num_features=dim),
                        )
                    ),
                    nn.Identity(),
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def validate_no_channel_mix_args(args: argparse.Namespace) -> None:
    if args.model != "convmixer":
        raise ValueError("--ablation no_channel_mix is only supported for --model convmixer")


def build_convmixer_no_channel_mix(args: argparse.Namespace) -> nn.Module:
    return ConvMixerNoChannelMix(
        dim=args.dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        patch_size=args.patch_size,
        n_classes=10,
    )
