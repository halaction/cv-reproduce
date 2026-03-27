import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from timm import create_model

from experiments.ablations.data_size import (
    add_data_size_args,
    infer_data_size_save_path,
    make_data_size_loaders,
    validate_data_size_args,
)
from experiments.ablations.mlp_channel import (
    add_mlp_args,
    build_convmixer_mlp_channel,
    infer_mlp_save_path,
    validate_mlp_args,
)
from experiments.ablations.patch_to_conv import (
    build_convmixer_patch_to_conv,
    validate_patch_to_conv_args,
)
from experiments.convmixer_models import build_convmixer
from experiments.train_utils import (
    count_parameters,
    evaluate,
    get_device,
    make_loaders,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image models on CIFAR-10")

    parser.add_argument(
        "--model",
        type=str,
        default="convmixer",
        choices=("convmixer", "deit_tiny", "resnet18"),
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
    add_mlp_args(parser)
    add_data_size_args(parser)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=("none", "patch_to_conv", "mlp_channel", "data_size"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument(
        "--wandb",
        type=str,
        default="disable",
        choices=("enable", "disable"),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_mlp_args(args)
    validate_data_size_args(args)

    if args.ablation == "patch_to_conv":
        validate_patch_to_conv_args(args)

    if args.ablation == "mlp_channel" and args.model != "convmixer":
        raise ValueError("--ablation mlp_channel is only supported for --model convmixer")

    if args.model in ("deit_tiny", "resnet18") and args.ablation in ("patch_to_conv", "mlp_channel"):
        raise ValueError("--ablation patch_to_conv/mlp_channel is only supported for --model convmixer")


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "convmixer":
        if args.ablation == "patch_to_conv":
            return build_convmixer_patch_to_conv(args)
        if args.ablation == "mlp_channel":
            return build_convmixer_mlp_channel(args)
        return build_convmixer(args)

    if args.model == "deit_tiny":
        return create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=10,
            img_size=32,
        )

    if args.model == "resnet18":
        return create_model(
            "resnet18",
            pretrained=False,
            num_classes=10,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def infer_save_path(args: argparse.Namespace) -> str:
    if args.save_path:
        return args.save_path

    if args.ablation == "data_size":
        return infer_data_size_save_path(args)

    if args.model == "convmixer":
        if args.ablation == "patch_to_conv":
            return "./checkpoints/convmixer_patch_to_conv_best.pt"
        if args.ablation == "mlp_channel":
            return infer_mlp_save_path(args)
        return "./checkpoints/convmixer_best.pt"

    return f"./checkpoints/{args.model}_best.pt"


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
            "model": args.model,
            "ablation": args.ablation,
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
            "fraction": args.fraction,
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
    validate_args(args)
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")

    if args.ablation == "data_size":
        train_loader, test_loader = make_data_size_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_strong_aug=args.strong_aug,
            fraction=args.fraction,
            seed=args.seed,
        )
        print(
            "Using data-size ablation with "
            f"fraction={args.fraction} | train_samples={len(train_loader.dataset)}"
        )
    else:
        train_loader, test_loader = make_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_strong_aug=args.strong_aug,
        )

    save_path = infer_save_path(args)
    model = build_model(args).to(device)

    param_count = count_parameters(model)
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

        if test_result.accuracy > best_test_acc:
            best_test_acc = test_result.accuracy
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_test_acc,
                path=save_path,
            )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_result.loss:.4f} | "
            f"train_acc={train_result.accuracy:.4f} | "
            f"test_loss={test_result.loss:.4f} | "
            f"test_acc={test_result.accuracy:.4f} | "
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
                    "time/train_sec": train_result.epoch_time_sec,
                    "time/eval_sec": test_result.epoch_time_sec,
                    "best_test_accuracy": best_test_acc,
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
