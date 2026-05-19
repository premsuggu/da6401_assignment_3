# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview
This repository contains a complete implementation of the Transformer architecture for Neural Machine Translation (NMT), specifically for German-to-English translation using the Multi30k dataset. The implementation is based on the landmark paper "Attention Is All You Need" and is built from scratch using PyTorch.

## Core Features
- **Transformer Architecture:** Custom implementation of Multi-Head Attention, Positional Encoding, and the Encoder/Decoder stacks.
- **Training Pipeline:** Support for Label Smoothing and the Noam Learning Rate Scheduler.
- **Inference:** A robust greedy decoding mechanism for generating translations.
- **Ablation Studies:** Comprehensive analysis of the Noam Scheduler, Scaling Factors, and Positional Encodings.

## Project Structure
- `model.py`: Contains the core Transformer model and sub-modules.
- `dataset.py`: Handles data loading, tokenization (SpaCy), and vocabulary building.
- `train.py`: Implementation of the training loop, validation, and inference methods.
- `lr_scheduler.py`: Custom Noam Scheduler implementation.
- `tasks/`: Individual notebooks for Part 2 experiments and visualizations.

## Links
- **Git Repository:** [https://github.com/premsuggu/da6401_assignment_3](https://github.com/premsuggu/da6401_assignment_3)
- **Weights & Biases Report:** [https://api.wandb.ai/links/prem11suggu/zxrqj6ew](https://api.wandb.ai/links/prem11suggu/zxrqj6ew)
