# cv-reproduce

Reproduced paper: https://arxiv.org/pdf/2201.09792

Main experiment: Running Convmixer, ResNet and ViT on CIFAR10.

Ablations:
1) Partial dataset - running the same models only with the part of CIFAR10. It is assumed that target architecture (Convmixer) will show better robustness.
2) Remove patches - replace patch embedding with a simple convolution sequence, which matches the output dimensions.
3) Remove mixing - ?

## Training

```bash
git clone https://github.com/halaction/cv-reproduce.git
cd cv-reproduce
uv sync
```

Optional `.env` for WandB (`--wandb enable`):

```bash
cat > .env << 'EOF'
WANDB_PROJECT=cv-reproduce
WANDB_ENTITY=your_wandb_username_or_team
EOF
```

Run ConvMixer:

```bash
uv run python experiment/train.py \
  --model convmixer \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run DeiT-Tiny:

```bash
uv run python experiment/train.py \
  --model deit_tiny \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run ResNet18:

```bash
uv run python experiment/train.py \
  --model resnet18 \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

