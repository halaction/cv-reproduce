import argparse
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from timm import create_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import datasets, transforms


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_conv_stem(dim: int) -> nn.Sequential:
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


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        kernel_size: int = 5,
        patch_size: int = 2,
        n_classes: int = 10,
        stem_type: str = "patch",
    ) -> None:
        super().__init__()

        if stem_type == "patch":
            self.stem = build_patch_stem(dim=dim, patch_size=patch_size)
        elif stem_type == "conv":
            self.stem = build_conv_stem(dim=dim)
        else:
            raise ValueError(f"Unsupported stem type: {stem_type}")

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


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    epoch_time_sec: float


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> EpochResult:
    model.train()
    start_time = time.time()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in tqdm(
        loader,
        desc="Train",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        running_total += batch_size

    epoch_time = time.time() - start_time
    return EpochResult(
        loss=running_loss / running_total,
        accuracy=running_correct / running_total,
        epoch_time_sec=epoch_time,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()
    start_time = time.time()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in tqdm(
        loader,
        desc="Eval",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        running_total += batch_size

    epoch_time = time.time() - start_time
    return EpochResult(
        loss=running_loss / running_total,
        accuracy=running_correct / running_total,
        epoch_time_sec=epoch_time,
    )


def get_transforms(
    use_strong_aug: bool = False,
) -> tuple[transforms.Compose, transforms.Compose]:
    if use_strong_aug:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    return train_transform, test_transform


def make_loaders(
    batch_size: int,
    num_workers: int,
    use_strong_aug: bool,
) -> tuple[DataLoader, DataLoader]:
    train_transform, test_transform = get_transforms(use_strong_aug=use_strong_aug)

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_acc: float,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        path,
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

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=("none", "patch_to_conv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="./checkpoints/convmixer_best.pt")
    parser.add_argument(
        "--wandb",
        type=str,
        default="disable",
        choices=("enable", "disable"),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ablation == "patch_to_conv":
        if args.model != "convmixer":
            raise ValueError("--ablation patch_to_conv is only supported for --model convmixer")
        if args.patch_size != 2:
            raise ValueError("--ablation patch_to_conv currently requires --patch-size 2")


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "convmixer":
        stem_type = "conv" if args.ablation == "patch_to_conv" else "patch"
        return ConvMixer(
            dim=args.dim,
            depth=args.depth,
            kernel_size=args.kernel_size,
            patch_size=args.patch_size,
            n_classes=10,
            stem_type=stem_type,
        )

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


def maybe_init_wandb(args: argparse.Namespace, device: torch.device, param_count: int):
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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dim": args.dim,
            "depth": args.depth,
            "kernel_size": args.kernel_size,
            "patch_size": args.patch_size,
            "grad_clip": args.grad_clip,
            "strong_aug": args.strong_aug,
            "ablation": args.ablation,
            "seed": args.seed,
            "device": str(device),
            "param_count": param_count,
            "save_path": args.save_path,
        },
    )
    return run


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_strong_aug=args.strong_aug,
    )

    save_path_default = "./checkpoints/convmixer_best.pt"
    if args.save_path == save_path_default and args.model != "convmixer":
        args.save_path = f"./checkpoints/{args.model}_best.pt"
    if (
        args.save_path == save_path_default
        and args.model == "convmixer"
        and args.ablation == "patch_to_conv"
    ):
        args.save_path = "./checkpoints/convmixer_patch_to_conv_best.pt"

    model = build_model(args).to(device)

    param_count = count_parameters(model)
    print(f"Trainable params: {param_count:,}")

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
    wandb_run = maybe_init_wandb(args=args, device=device, param_count=param_count)

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
                path=args.save_path,
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
    print(f"Best model saved to: {args.save_path}")
    if wandb_run is not None:
        wandb_run.summary["best_test_accuracy"] = best_test_acc
        wandb_run.summary["best_checkpoint"] = args.save_path
        wandb_run.finish()


if __name__ == "__main__":
    main()
