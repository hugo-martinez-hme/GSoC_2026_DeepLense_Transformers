# ML4SCI DeepLense - Foundation Models & Transformers (GSoC 2026)
**Candidate:** Hugo Martínez Estévez
**Project:** Foundation Model for Gravitational Lensing / Exploring Transformers

## Overview
This repository contains the evaluation tasks for the DeepLense "Foundation Model" project. Due to local hardware constraints (CUDA driver mismatch on Arch Linux forcing CPU execution), models were trained on a restricted subset of the dataset (800 images/class) for a limited number of epochs. The focus is strictly on architectural correctness and PyTorch pipeline design.

## 1. Common Test I (`Common_Test_I_Baseline.ipynb`)
- **Objective:** Multi-class classification baseline.
- **Implementation:** A baseline CNN with Log-Scale preprocessing to handle the high dynamic range of astronomical images.

## 2. Specific Test V (`Specific_Test_V_Transformers.ipynb`)
- **Objective:** Implement a Transformer-based model for the same classification task, strictly avoiding Convolutional Neural Networks (CNNs).
- **Implementation:** A custom PyTorch `MicroViT` (Vision Transformer). The image is mathematically split into patches via `unfold` and projected using a pure `nn.Linear` embedding layer to adhere to the "No CNN" rule. It includes a learnable class token, positional embeddings, and a multi-head Self-Attention encoder.
- **Results:** Successfully demonstrated the forward pass, loss optimization, and ROC/AUC generation of a pure ViT architecture on CPU.
