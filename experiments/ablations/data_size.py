import argparse
import random
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from experiments.train_utils import get_transforms


def add_data_size_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fraction", type=float, default=1.0)


def validate_data_size_args(args: argparse.Namespace) -> None:
    if args.ablation == "data_size":
        if not (0.0 < args.fraction <= 1.0):
            raise ValueError("--fraction must be in the range (0, 1] for --ablation data_size")
    elif args.fraction != 1.0:
        raise ValueError("--fraction is only supported when --ablation data_size")


def _build_stratified_indices(
    targets: list[int],
    fraction: float,
    seed: int,
) -> list[int]:
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, target in enumerate(targets):
        class_to_indices[target].append(idx)

    rng = random.Random(seed)
    sampled_indices: list[int] = []

    for class_id in sorted(class_to_indices):
        class_indices = class_to_indices[class_id]
        class_size = len(class_indices)
        take_n = max(1, round(class_size * fraction))
        take_n = min(class_size, take_n)
        sampled_indices.extend(rng.sample(class_indices, k=take_n))

    rng.shuffle(sampled_indices)
    return sampled_indices


def make_data_size_loaders(
    batch_size: int,
    num_workers: int,
    use_strong_aug: bool,
    fraction: float,
    seed: int,
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

    sampled_indices = _build_stratified_indices(
        targets=train_dataset.targets,
        fraction=fraction,
        seed=seed,
    )
    train_subset = Subset(dataset=train_dataset, indices=sampled_indices)

    train_loader = DataLoader(
        dataset=train_subset,
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


def infer_data_size_save_path(args: argparse.Namespace) -> str:
    fraction_label = str(args.fraction).replace(".", "p")
    return f"./checkpoints/data_size_{args.model}_frac{fraction_label}.pt"
