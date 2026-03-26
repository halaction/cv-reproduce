# cv-reproduce

Reproduced paper: https://arxiv.org/pdf/2201.09792

Main experiment: Running Convmixer, ResNet and ViT on CIFAR10.

Ablations:
1) Partial dataset - running the same models only with the part of CIFAR10. It is assumed that target architecture (Convmixer) will show better robustness.
2) Remove patches - replace patch embedding with a simple convolution sequence, which matches the output dimensions.
3) Remove mixing - ?
