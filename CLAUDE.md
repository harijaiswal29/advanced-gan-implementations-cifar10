# Assignment 02 - GAN Variants on CIFAR-10

## Project Overview

This is an academic assignment for the **Advanced Deep Learning** course (BITS Pilani M.Tech AI/ML, Semester 3). It implements and compares four GAN variants trained on the CIFAR-10 dataset (automobile class), evaluating image generation quality via Inception Score (IS) and Frechet Inception Distance (FID).

## Repository Structure

```
Final Submission/
  Task1_Conditional_WGAN.html    # Conditional Wasserstein GAN (notebook export)
  Task2_SNGAN.html               # Spectral Normalization GAN (notebook export)
  Task3_SAGAN_NoSN.html          # SAGAN without SN and TTUR (notebook export)
  Task4_CompleteSAGAN.html       # Complete SAGAN (notebook export)
  Task5_Comparison_GAN.pdf       # Observations and comparison summary
  Task1_WGAN_Images.zip          # Generated automobile images (WGAN)
  Task2_SNGAN_Images.zip         # Generated automobile images (SNGAN)
  Task3_SAGAN_NoSN_Images.zip    # Generated images (SAGAN w/o SN)
  Task4_CompleteSAGAN_Images.zip # Generated images (Complete SAGAN)
Assignment_Problem_Statement.md  # Problem statement
```

## Tech Stack

- **Framework**: PyTorch
- **Dataset**: CIFAR-10 (torchvision), filtered to automobile class (label 1)
- **Evaluation**: InceptionV3 (pretrained) for IS and FID computation
- **Environment**: Google Colab with CUDA GPU, Google Drive for checkpoints
- **Dependencies**: torch, torchvision, numpy, scipy, tqdm, logging

## Task Summary

| Task | Model | Key Techniques | Epochs | Best FID |
|------|-------|----------------|--------|----------|
| 1 | Conditional WGAN-GP | Gradient penalty, class conditioning, spectral norm (critic), self-attention | 250 | ~63.2 |
| 2 | SNGAN | Spectral normalization (discriminator), hinge loss, self-attention | 150 | ~35.1 |
| 3 | SAGAN (no SN) | Self-attention, NO spectral norm, TTUR | 150 | ~92.1 |
| 4 | Complete SAGAN | Self-attention, spectral norm (both G & D), TTUR, gradient penalty | 250 | ~85.3 |

## Common Patterns

- All models use **Adam optimizer** with beta1=0.0 (or 0.5), beta2=0.9 (or 0.999)
- All use **CosineAnnealingLR** scheduler
- Evaluation uses **10,000 generated samples** resized to 299x299 for InceptionV3
- FID computed via Frechet distance on 2048-dim inception features
- IS computed via 10-split KL divergence
- All notebooks include full checkpoint save/resume logic
- Image size: 32x32 RGB (CIFAR-10 native)

## Key Architectural Components

- **Self-Attention**: Query/Key reduce channels by 8x, learnable gamma (init=0), applied at 16x16 resolution
- **Spectral Normalization**: `nn.utils.spectral_norm` wrapper on Conv2d/Linear layers
- **ResBlock patterns**: Generator uses BN + ReLU + upsample; Discriminator uses LeakyReLU + downsample
- **Hinge Loss** (Tasks 2-4): D_loss = mean(ReLU(1-real)) + mean(ReLU(1+fake)); G_loss = -mean(fake)
- **WGAN-GP Loss** (Task 1): Wasserstein distance + gradient penalty (lambda=10)
