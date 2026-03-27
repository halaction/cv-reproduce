import argparse

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


def build_patch_stem(dim: int, patch_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        ),
        nn.GELU(),
        nn.BatchNorm2d(num_features=dim),
    )


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 5,
        patch_size: int = 2,
        n_classes: int = 10,
        stem: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.stem = stem if stem is not None else build_patch_stem(dim=dim, patch_size=patch_size)

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
                    nn.Conv2d(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                    nn.BatchNorm2d(num_features=dim),
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


def build_convmixer(args: argparse.Namespace) -> nn.Module:
    return ConvMixer(
        dim=args.dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        patch_size=args.patch_size,
        n_classes=10,
    )
