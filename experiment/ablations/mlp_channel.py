import argparse

import torch
import torch.nn as nn

from experiment.convmixer_models import Residual, build_patch_stem


class ChannelMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim < dim:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} gives hidden_dim={hidden_dim}, "
                f"but hidden_dim must be >= dim={dim}."
            )

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(num_features=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvMixerMLPChannel(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 5,
        patch_size: int = 2,
        n_classes: int = 10,
        mlp_ratio: float = 4.0,
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
                    ChannelMLP(dim=dim, mlp_ratio=mlp_ratio),
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


def add_mlp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mlp-ratio", type=float, default=4.0)


def validate_mlp_args(args: argparse.Namespace) -> None:
    if args.ablation == "mlp_channel":
        if args.model != "convmixer":
            raise ValueError("--ablation mlp_channel is only supported for --model convmixer")
    elif args.mlp_ratio != 4.0:
        raise ValueError("--mlp-ratio is only supported when --ablation mlp_channel")


def build_convmixer_mlp_channel(args: argparse.Namespace) -> nn.Module:
    return ConvMixerMLPChannel(
        dim=args.dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        patch_size=args.patch_size,
        n_classes=10,
        mlp_ratio=args.mlp_ratio,
    )


def infer_mlp_save_path(args: argparse.Namespace) -> str:
    return (
        "./checkpoints/"
        f"mlp_channel_mixer_"
        f"d{args.dim}_dep{args.depth}_"
        f"k{args.kernel_size}_p{args.patch_size}_"
        f"mlpr{args.mlp_ratio}.pt"
    )
