# cv-reproduce

Reproduced paper: https://arxiv.org/pdf/2201.09792

Report is placed at `report/main.pdf`.

Main experiment: Running Convmixer, ResNet and ViT on CIFAR10.

Ablations:
1) Remove patches - replace patch embedding with a simple convolution sequence, which matches the output dimensions.
2) Non-linear channel mixing - replace linear channel mixing with its non-linear variant (two convolution with a non-linearity in between).
3) No channel mixing - replace linear channel mixing with an identity function.
4) Partial dataset - running the same models only with the part of CIFAR10. It is assumed that target architecture (Convmixer) will show better robustness.

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
WANDB_ENTITY=halaction
EOF
```

Run ConvMixer:

```bash
uv run python experiments/train.py \
  --model convmixer \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run DeiT-Tiny:

```bash
uv run python experiments/train.py \
  --model deit_tiny \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run ResNet18:

```bash
uv run python experiments/train.py \
  --model resnet18 \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run ConvMixer with patch-to-conv ablation:

```bash
uv run python experiments/train.py \
  --model convmixer \
  --ablation patch_to_conv \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run ConvMixer with non-linear channel mixing ablation:

```bash
uv run python experiments/train.py \
  --model convmixer \
  --ablation non_linear_channel_mix \
  --mlp-ratio 1.0 \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```

Run ConvMixer with no-channel-mix ablation:

```bash
uv run python experiments/train.py \
  --model convmixer \
  --ablation no_channel_mix \
  --epochs 50 \
  --batch-size 128 \
  --wandb enable
```
