# Monolithic Joint-Embedding Revolution (Classifier-Free Guidance)

[← Back to Main README](../README.md)

## Overview
**Classifier-Free Guidance (CFG)**, introduced by Jonathan Ho and Tim Salimans in 2021, bypassed the need for an external classifier network by training a single monolithic model to predict both conditional and unconditional outputs.

## Mechanism
During training, the conditioning signal $c$ is randomly replaced with a null token $\emptyset$ (dropout rate of 10-20%). During inference, the guided output is calculated as:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \left( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \right)$$

where $s \ge 1$ is the guidance scale.

## Architectural Flow
```mermaid
flowchart TD
    Prompt["Conditioning Prompt (c)"] --> DropBlock{"10-20% Dropout?"}
    DropBlock -- Yes --> NullToken["Null Condition (∅)"]
    DropBlock -- No --> EmbedToken["Prompt Embedding (c)"]
    NullToken --> JointModel["Monolithic Diffusion Model"]
    EmbedToken --> JointModel
    Latent["Noisy Input xₜ"] --> JointModel
    JointModel --> UnifiedOutput["Unified Score Predictor"]
```

## Advantages
- Bypasses adversarial texture corruption.
- Eliminates secondary classifier pipeline overhead during inference.
