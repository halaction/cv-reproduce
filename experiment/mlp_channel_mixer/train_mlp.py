import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

from experiment.train import (
    count_parameters,
    evaluate,
    get_device,
    make_loaders,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ChannelMLP(nn.Module):
    """
    MLP-kinda channel mixer:
    1x1 conv -> GELU -> 1x1 conv -> GELU -> BatchNorm2d
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim < dim:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} gives hidden_dim={hidden_dim}, "
                f"but hidden_dim must be >= dim={dim}."
            )

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=dim,
                kernel_size=1,
            ),
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

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.GELU(),
            nn.BatchNorm2d(num_features=dim),
        )

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
                    ChannelMLP(
                        dim=dim,
                        mlp_ratio=mlp_ratio,
                    ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvMixer with MLP channel mixer"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="disable",
        choices=("enable", "disable"),
    )

    return parser.parse_args()


def infer_save_path(args: argparse.Namespace) -> str:
    if args.save_path:
        return args.save_path

    return (
        "./checkpoints/"
        f"mlp_channel_mixer_"
        f"d{args.dim}_dep{args.depth}_"
        f"k{args.kernel_size}_p{args.patch_size}_"
        f"mlpr{args.mlp_ratio}.pt"
    )


def maybe_init_wandb(
    args: argparse.Namespace,
    device: torch.device,
    param_count: int,
    save_path: str,
):
    if args.wandb != "enable":
        return None

    import wandb

    load_dotenv(override=False)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    run = wandb.init(
        project=project,
        entity=entity,
        config={
            "experiment": "mlp_channel_mixer",
            "model": "convmixer_mlp_channel",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dim": args.dim,
            "depth": args.depth,
            "kernel_size": args.kernel_size,
            "patch_size": args.patch_size,
            "mlp_ratio": args.mlp_ratio,
            "grad_clip": args.grad_clip,
            "strong_aug": args.strong_aug,
            "seed": args.seed,
            "device": str(device),
            "param_count": param_count,
            "save_path": save_path,
        },
    )

    return run


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_strong_aug=args.strong_aug,
    )

    model = ConvMixerMLPChannel(
        dim=args.dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        patch_size=args.patch_size,
        n_classes=10,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    param_count = count_parameters(model)
    save_path = infer_save_path(args)

    print(f"Trainable params: {param_count:,}")
    print(f"Checkpoint path: {save_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
    )

    best_test_acc = 0.0
    wandb_run = maybe_init_wandb(
        args=args,
        device=device,
        param_count=param_count,
        save_path=save_path,
    )

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        test_result = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if test_result.accuracy > best_test_acc:
            best_test_acc = test_result.accuracy
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_test_acc,
                path=save_path,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_result.loss:.4f} | "
            f"train_acc={train_result.accuracy:.4f} | "
            f"test_loss={test_result.loss:.4f} | "
            f"test_acc={test_result.accuracy:.4f} | "
            f"best_test_acc={best_test_acc:.4f} | "
            f"train_time={train_result.epoch_time_sec:.1f}s | "
            f"eval_time={test_result.epoch_time_sec:.1f}s"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/loss": train_result.loss,
                    "train/accuracy": train_result.accuracy,
                    "test/loss": test_result.loss,
                    "test/accuracy": test_result.accuracy,
                    "best_test_accuracy": best_test_acc,
                    "time/train_sec": train_result.epoch_time_sec,
                    "time/eval_sec": test_result.epoch_time_sec,
                }
            )

    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best model saved to: {save_path}")

    if wandb_run is not None:
        wandb_run.summary["best_test_accuracy"] = best_test_acc
        wandb_run.summary["best_checkpoint"] = save_path
        wandb_run.finish()


if __name__ == "__main__":
    main()