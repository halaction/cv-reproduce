import argparse

import torch.nn as nn

from experiments.convmixer_models import ConvMixer


def build_patch_to_conv_stem(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.GELU(),
        nn.BatchNorm2d(num_features=dim),
        nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=dim,
            bias=False,
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
