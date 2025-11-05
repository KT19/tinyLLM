# Train LLM on a Single Consumer GPU

This repository provides a complete training pipeline for training a (tiny) Large Language Model on a single GPU (~16 GB VRAM), supporting 4K context length.
The goal is to demonstrate an efficient, end-to-end workflow — from tokenizer training to post-training — that fits within consumer hardware constraints.

## Environment

Framework: Pytorch

Hardware: Single GPU (≈ 16 GB VRAM)

Package Manager: UV
 (for environment and dependency management)

## Components
- Dataset preparation
`uv run create_shard.py`

- Tokenizer training (BPE)
`uv run train_tokenizer.py`

- Pretraining
  - Three-stage curriculum to support progressively longer sequences
`uv run pretrain.py`

- Post-training
`uv run sft.py`

- Chat conversion
  - 4K context length
`uv run chat_inference.py`

## Model Architecture

Total parameters (146M)

To support long-context and memory-efficient training, several custom attention mechanisms and optimizations are integrated:

- Sliding-Window Attention

- Grouped-Query Attention

- Flash-Attention

- Tied input/output embeddings

All of these are combined to maximize memory efficiency without sacrificing performance.

Additionally, a customized Hierarchical Reasoning Model (HRM) is attached on top of the dense transformer.
Experimentally, full backpropagation (instead of 1-step gradient approximation) yields better reasoning performance, so it is used here.

Deep supervision is omitted because integrating it with a dense transformer + HRM setup is non-trivial.

## Training Pipeline Overview

0. Dataset Preparation

Pretraining uses FineWeb-Edu and Open-Web-Math datasets.
To fit within limited RAM (~ 5 GB), each dataset is sharded into smaller subsets.
Total dataset size: ≈ 10 GB (5 GB × 2).

1. Tokenizer Training

A custom 32 K-vocabulary BPE tokenizer is trained from scratch using the pretraining corpus.
While public tokenizers (e.g., GPT-2) exist, their larger vocabularies are inefficient for small-scale training, so a custom tokenizer is preferred.

2. Pretraining

Following standard causal language modeling (next-token prediction), pretraining uses curriculum learning — gradually increasing input lengths as training progresses to stabilize convergence on long sequences. (1k -> 2k -> 4k)

Note: Can be extended to 8K by reducing batch size, but 4K is chosen as the best trade-off between compute efficiency and context capacity.

Training time in my environment: ⏱ ~ 1600 minutes (in current configs)

3. Post-Training

The post-training phase adds instruction following and reasoning capabilities.

**Stage 1 — Supervised Fine-Tuning (SFT)**

Datasets: Alpaca + Dolly

Context length < 2048

⏱ ~ 155 minutes

**Stage 2 — Extended Fine-Tuning**

Datasets: UltraChat + Open-R1-Math

Context length < 4096

Includes manual reasoning-format augmentation for Open-R1-Math to boost reasoning ability

⏱ ~ 50 minutes / per 10000 steps

## Example of generated texts (Snapshot)
### After pretraining
- Input
```
What is
```
- Response
```
What is a critical piece of information that supports the decision-making process. For example, you can use a computer model to predict how many hours it takes you to run a car. You can also use these models to determine the time for an event to occur.
How do you predict your outcome?
The simplest and easiest way to predict your outcome is to have a plan. You can try to predict how your outcome will take place by using the model. Some of the key things you can do to make sure that your outcome will be different in each instance.
```

### After SFT stage1
- User input
```
What is an apple?
```
- AI response
```
An apple is a small, round oval-shaped fruit with a creamy texture that resembles the flavor of freshly baked apples or oranges. It has a sweet and tangy taste with its sweetness and texture. The flesh is firm and smooth, with a slightly tangy flesh that makes it a great addition to any dish. Its sweetness and crunch make it a popular ingredient in many dishes such as salads, fried fruit, and other snacks. The flesh is often used as a sweetener for desserts, sweets, and baked goods, as well as for a variety of dishes like grilled chicken breasts or mashed potatoes.
```

### After SFT stage2
Check it yourself!


## References
- GQA
- MQA
- Flash-attention
- Tie embedding
- Roadless scheduler
- Alpaca dataset
- Open assistant dataset
- Ultrachat dataset
- universal transformer
- HRM
