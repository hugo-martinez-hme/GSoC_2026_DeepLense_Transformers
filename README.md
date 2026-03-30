# DeepLense: Foundation Model for Gravitational Lensing

This repository contains the evaluation tests for the Google Summer of Code 2026 project: **Foundation Model for Gravitational Lensing** under the **ML4SCI** organization.

## Project Overview
The goal is to develop a self-supervised foundation model using Vision Transformers (ViT) to learn robust representations of dark matter substructures. This model serves as a backbone for downstream tasks such as classification, regression, and super-resolution.

## Completed Tasks

### [Specific Test VII: Physics-Informed Neural Networks (PINNs)](./Specific_Test_VII_PINN.ipynb)
- Implementation of a PINN framework that integrates gravitational lensing physical laws into the loss function.
- Focus on improving interpretability and physical consistency in mass distribution estimates.

### [Specific Test IX: Foundation Model (MAE & Super-Resolution)](./Specific_Test_IX_Foundation.ipynb)
- **Task IX.A:** Pre-training a Masked Autoencoder (MAE) on unlabeled lensing data and fine-tuning for multi-class classification (no_sub, cdm, axion). Includes ROC curves and AUC evaluation.
- **Task IX.B:** Fine-tuning the pre-trained encoder for a Super-Resolution task. Performance evaluated using MSE, SSIM, and PSNR metrics.

## Hardware Note
Due to local CUDA driver constraints on Arch Linux, models were validated architecturally using CPU execution. Full-scale training is planned for GPU/HPC environments during the GSoC coding period.
