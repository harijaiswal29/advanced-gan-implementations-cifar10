# Assignment 02 - Comparative Study of GAN Variants on CIFAR-10

**Course**: Advanced Deep Learning | **Program**: M.Tech AI/ML, BITS Pilani | **Semester**: 3

## Problem Statement

Implement, train, and evaluate four GAN variants on the CIFAR-10 dataset, generating images of the **automobile** class. Report Inception Score (IS) and Frechet Inception Distance (FID) for each model and summarize observations.

## Tasks

### Task 1: Conditional Wasserstein GAN (WGAN-GP)

A class-conditional WGAN with gradient penalty, using ResBlock-based generator and critic with self-attention.

| Component | Details |
|-----------|---------|
| **Generator** | ResBlock-based, self-attention at 16x16, BatchNorm, label embedding (128-dim) concatenated with noise |
| **Critic** | ResBlock-based, spectral normalization, self-attention at 16x16, label embedding in final layer |
| **Loss** | Wasserstein loss + gradient penalty (lambda=10) |
| **Training** | 250 epochs, batch size 64, lr_g=0.0001, lr_c=0.0002, n_critic=5 |
| **Extras** | Instance noise (first 50 epochs), ColorJitter augmentation |

**Results**: IS = 1.167, FID = 64.25 (best FID ~63.2 at epoch 180)

---

### Task 2: Spectral Normalization GAN (SNGAN)

An SNGAN with spectral normalization applied to all discriminator layers, using ResBlock architecture and hinge loss.

| Component | Details |
|-----------|---------|
| **Generator** | ResBlock-based with upsampling, self-attention at 16x16, BatchNorm |
| **Discriminator** | ResBlock-based with downsampling, spectral normalization on all layers, self-attention at 16x16 |
| **Loss** | Hinge loss |
| **Training** | 150 epochs, batch size 64, lr_g=0.0002, lr_d=0.0002, n_critic=1 |

**Results**: IS = 1.131, FID = 41.50 (best FID ~35.1 at epoch 120)

---

### Task 3: SAGAN without Spectral Normalization and TTUR

A Self-Attention GAN variant with spectral normalization **disabled** in both generator and discriminator, to study its impact on training stability.

| Component | Details |
|-----------|---------|
| **Generator** | ConvTranspose2d-based, self-attention at 16x16, BatchNorm, NO spectral norm |
| **Discriminator** | Conv2d-based, self-attention at 8x8, BatchNorm, NO spectral norm |
| **Loss** | Hinge loss |
| **Training** | 150 epochs, batch size 64, lr_g=0.0001, lr_d=0.0004 |

**Results**: IS = 1.004, FID = 512.36 (best FID ~92.1 at epoch 130, then catastrophic mode collapse)

> **Key Finding**: Without spectral normalization, the model suffered catastrophic mode collapse after epoch 130. The discriminator became too powerful (D loss -> 0), providing insufficient gradients to the generator.

---

### Task 4: Complete SAGAN

A full Self-Attention GAN with spectral normalization on both generator and discriminator, TTUR (Two Time-scale Update Rule), and gradient penalty.

| Component | Details |
|-----------|---------|
| **Generator** | ConvTranspose2d-based, self-attention at 16x16, BatchNorm, spectral norm on all layers |
| **Discriminator** | Conv2d-based, self-attention at 8x8, BatchNorm, spectral norm on all layers |
| **Loss** | Hinge loss + gradient penalty (lambda=10) |
| **Training** | 250 epochs, batch size 64, lr_g=0.0001, lr_d=0.0004, early stopping (patience=15) |
| **TTUR** | D learning rate 4x higher than G learning rate |

**Results**: IS = 1.105, FID = 86.28 (best FID ~85.3 at epoch 220)

---

### Task 5: Comparative Observations

Detailed comparison and analysis of all four GAN variants is provided in `Task5_Comparison_GAN.pdf`.

## Results Summary

| Model | Best FID | Final FID | Final IS | Epochs | Training Stability |
|-------|----------|-----------|----------|--------|-------------------|
| **SNGAN** (Task 2) | **35.1** | 41.5 | 1.131 | 150 | Stable |
| **WGAN-GP** (Task 1) | 63.2 | 64.2 | 1.167 | 250 | Stable |
| **Complete SAGAN** (Task 4) | 85.3 | 86.3 | 1.105 | 250 | Mostly stable |
| **SAGAN w/o SN** (Task 3) | 92.1 | 512.4 | 1.004 | 150 | Collapsed at epoch ~130 |

**Key Takeaways**:
- SNGAN achieved the best FID score (35.1), demonstrating the effectiveness of spectral normalization with a simple architecture
- Removing spectral normalization (Task 3) led to catastrophic mode collapse, confirming its critical role in stabilizing GAN training
- The conditional WGAN-GP maintained stable training throughout 250 epochs thanks to gradient penalty
- IS scores are relatively low (~1.1) across all models because evaluation targets a single class (automobile), limiting inter-class diversity

## Environment

- **Runtime**: Google Colab (CUDA GPU)
- **Framework**: PyTorch
- **Dataset**: CIFAR-10 (automobile class, ~5,000 training images)
- **Evaluation**: InceptionV3 (pretrained on ImageNet) for IS and FID

## Repository Contents

```
Final Submission/
    Task1_Conditional_WGAN.html        # Notebook export - Conditional WGAN-GP
    Task2_SNGAN.html                   # Notebook export - SNGAN
    Task3_SAGAN_NoSN.html              # Notebook export - SAGAN without SN
    Task4_CompleteSAGAN.html           # Notebook export - Complete SAGAN
    Task5_Comparison_GAN.pdf           # Observations and comparative analysis
    Task1_WGAN_Images.zip              # Generated automobile samples (WGAN)
    Task2_SNGAN_Images.zip             # Generated automobile samples (SNGAN)
    Task3_SAGAN_NoSN_Images.zip        # Generated samples (SAGAN w/o SN)
    Task4_CompleteSAGAN_Images.zip     # Generated samples (Complete SAGAN)
Assignment_Problem_Statement.md        # Problem statement
```
