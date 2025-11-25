# High Frequency Market Data Microstructure Classification (2024)

## CapitalFund Management - ENS Data Challenge 

(**Deep Learning for Stock Identity Recognition from Order Book Sequences**)

![High-Frequency Trading Visualization](https://empirica.io/wp-content/uploads/2020/05/high-frequency-trading.jpg)


*Financial microstructure is full of subtle signatures.
Some equities trade with tight spreads and constant flow.
Others show bursts of activity, asymmetric order imbalance, or venue-specific behavior.* 
*These patterns are rarely visible with the naked eye,
but they are deeply encoded in the flow of bid/ask updates and transactions.*

And then comes the challenge:

**Given one of more than 1,000 unique sequences of 100 consecutive order-book events (all anonymous, all different, and with unknown timing (milliseconds or seconds)) :** **Can we identify which of 24 equities produced it ?**

This repository contains my full solution.

## Understanding the target

Each training example corresponds to:

- one stock (identity hidden),
- observed over 1000+ : 100 consecutive microstructure events,
- forming a short "signature" of its trading behavior.

The goal is to classify each 100-event sequence into one of:

24 equity classes {0, 1, …, 23} each one representing a unique stock id

Microstructure patterns differ across equities due to:

- volatility regimes
- liquidity and spread behavior
- order imbalances
- trade size distributions
- venue-specific activity

The model must learn these statistical fingerprints.

## A mathematically guided deep-learning approach

My method is based on the complementarity between:

- financial microstructure intuition,
- embedding layers for categorical market events,
- a bidirectional GRU backbone,
- strong regularization,
- and a stable training strategy.

Although the notebook includes full implementation detail, the intuition rests on four pillars: sequence reconstruction, embedding learning, temporal modeling, and robust regularization.

## The problem to solve

This is a multiclass sequence classification problem:

- 24 classes
- 1,048 sequences
- each = 100 time steps, mixing numerical and categorical microstructure features

Events are noisy, order-dependent, and partially stochastic — justifying the use of recurrent deep learning architectures.

## Modeling pipeline

### I - Preprocessing

**Sequence reconstruction** (Raw table → 3D tensor: (1048, 100, d))
**Feature handling** (Numerical features scaled, Categorical channels to embeddings, 80/20 train-validation split)

### II - Model architecture

**Embeddings** (For venue, action, side, trade), Learned representations: $\text{Embed}(c) \in \mathbb{R}^k$

**Bidirectional GRU**
Processes temporal structure forward and backward: $$h_t^{bi} = [h_t^{\rightarrow}, h_t^{\leftarrow}]$$
Mathematically, gates allow: $$h_t = (1 - z_t)h_{t-1} + z_t \tilde{h}_t$$

**Regularization** (Dropout, recurrent dropout, L2 penalties, layer normalization, label smoothing ($\varepsilon = 0.1$))

**Softmax classifier**
Outputs: $p(y = k | x)$, Prediction via: $\hat{y} = \arg\max_k p(y = k | x)$

### III - Training strategy
Early stopping, ReduceLROnPlateau, Best-epoch checkpointing : Ensures stable learning and good generalization.

### IV - Prediction

Reload the best GRU model, apply identical preprocessing, predict class probabilities, select argmax, export y_prediction.csv

## How performance is evaluated

The challenge platform receives your prediction file and compares it with the real stock identities of the test sequences (identities that participants never have access to).
It computes: $$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$
This accuracy becomes the public score.

## My results

After reconstruction, embedding learning, temporal modeling, and strong regularization:

**Final Accuracy**: 0.42755

**Rank**: 40 / 187 participants

*https://challengedata.ens.fr/challenges/146*

## What's in the notebook

- Data preprocessing
- Sequence reconstruction
- Embedding + Bi-GRU architecture
- Mathematical intuition
- Regularization strategies
- Training loop
- Prediction & submission file

Everything is reproducible and aligned with microstructure principles.

## Further ideas

- Transformer architectures
- Dilated CNNs for pattern extraction
- OFI features, spread dynamics, volatility metrics
- Ensembling of GRU + LSTM + CNN
- Microstructure data augmentation
