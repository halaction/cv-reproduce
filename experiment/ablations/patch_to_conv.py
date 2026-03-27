import argparse

import torch.nn as nn

from experiment.convmixer_models import ConvMixer


def build_patch_to_conv_stem(dim: int) -> nn.Sequential:
    hidden_dim = max(1, dim // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.GELU(),
        nn.BatchNorm2d(num_features=hidden_dim),
        nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.GELU(),
        nn.BatchNorm2d(num_features=dim),
    )


def validate_patch_to_conv_args(args: argparse.Namespace) -> None:
    if args.model != "convmixer":
        raise ValueError("--ablation patch_to_conv is only supported for --model convmixer")
    if args.patch_size != 2:
        raise ValueError("--ablation patch_to_conv currently requires --patch-size 2")


def build_convmixer_patch_to_conv(args: argparse.Namespace) -> nn.Module:
    return ConvMixer(
        dim=args.dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        patch_size=args.patch_size,
        n_classes=10,
        stem=build_patch_to_conv_stem(dim=args.dim),
    )
