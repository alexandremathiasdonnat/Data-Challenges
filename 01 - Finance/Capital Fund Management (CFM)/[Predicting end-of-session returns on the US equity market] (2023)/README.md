# Predicting End-of-session Returns on the US Equity Market (2023)

## Capital Fund Management – ENS Data Challenge

**(Predicting Afternoon Price Direction from Morning Intraday Trajectories)**

![Financial Markets Banner](https://depts.washington.edu/compfin/web/wp-content/uploads/2015/09/forex-blue-banner.jpeg)



*Stocks prices intraday trajectories carry structure.
Some equities show smooth morning trends.
Others oscillate violently, with bursts of volatility.
Many stay flat for hours before suddenly moving.*

*These patterns are invisible to the naked eye,
but they can be encoded in the sequence of five-minute returns.*

And that leads to the core question:

**For each anonymized stock ID : given a morning return path (09:30–14:00), can we predict the direction of the 14:00–16:00 move?**

This repository contains my full contribution.

## Understanding the target

The training and prediction files each contain ≈ 850,000 rows.

Every row corresponds to:

- one equity on one trading day (both anonymized),
- a sequence of 53 intraday returns r₀…r₅₂ in basis points,
- describing the price evolution from 09:30 to 14:00,
- and a target label reod ∈ {−1, 0, 1} describing the subsequent 14:00–16:00 return:

| Label | Meaning | Threshold |
|-------|---------|-----------|
| −1 | strong down move | < −25 bps |
| 0 | flat / small move | between −25 and +25 bps |
| +1 | strong up move | > +25 bps |

The challenge is to infer the future two-hour move using only the five-minute path of the morning.

## The problem to solve

Several factors make this prediction problem fundamentally difficult:

- Afternoon returns are extremely noisy and often small.
- Many drivers of price moves—news, macro releases, order-book shocks—are unobserved.
- Train and test contain different days and different stocks, so the model must generalize patterns, not memorize IDs.
- The predictive signal is weak compared to randomness.

Yet intraday paths still contain statistical fingerprints:

- cumulative direction of the morning (ret_sum),
- imbalance between positive and negative 5-min returns,
- intraday volatility / dispersion,
- presence of large spikes,
- shape of the cumulative trajectory.

The goal is to extract these fingerprints with a clean, memory-efficient gradient-boosting pipeline.

## A mathematically guided gradient-boosting approach

Rather than brute-forcing thousands of engineered features or training neural nets requiring large compute,
this notebook implements a stable, interpretable baseline built on three pillars:

### (1) Full use of the raw return path

All 53 returns r₀…r₅₂ are kept—no information loss.

### (2) Global statistics capturing the path shape

Including:

- sum of returns (trend)
- standard deviation (volatility)
- counts of positive and negative moves
- counts of large spikes (>25 bps, >50 bps)

These form a compact but expressive representation.

### (3) A regularized LightGBM multiclass classifier

- small num_leaves
- large min_data_in_leaf
- feature/bagging subsampling
- early stopping
- careful validation

This is sufficient to approximate the provided benchmark and extract real signal by surpassing randomness.

## Modeling pipeline

### I — Preprocessing

- Read x_train.csv, y_train.csv, x_test.csv
- Align IDs
- Build features from raw returns
- No use of day or equity → prevents leakage & overfitting
- Ensure train/test have identical feature order

### II — Feature Engineering

Simple but strong:

- 53 raw intraday returns
- ret_sum
- ret_std
- pos_count
- neg_count
- big_move_count_25
- big_move_count_50

All numeric, memory-efficient, and interpretable.

### III — Train/Val split

- 80/20 stratified split
- Multiclass mapping {−1,0,1} → {0,1,2}

### IV — LightGBM model

Main settings:

- 3 classes
- learning_rate = 0.05
- num_leaves = 31
- min_data_in_leaf = 200 (prevents overfitting)
- bagging + feature subsampling
- 2000 boosting rounds with early stopping

This produces a stable and reproducible accuracy around 0.41–0.49, depending on randomness.

### V — Full retraining & predictions

Refit using all training data with best_iteration.
Export y_prediction.csv.

## Evaluation metric

The challenge platform computes:

$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$

A purely random classifier (three balanced classes) would get:

- ≈ 33% accuracy

The provided benchmark achieves:

- 41.74%

My pipeline matches and slightly surpasses this range depending on the split,
demonstrating that real predictive signal was captured from intraday paths.

## My results

After engineering trajectory statistics and applying a strongly regularized LightGBM:

**Final internal accuracy:** ~0.49

**Internal macro-F1:** ~0.48

**Public leaderboard accuracy :** ~0.412

**Rank**: 130 / 260 participants

*https://challengedata.ens.fr/challenges/84*

*Any score significatively  > 0.33 proves that the model is not random and that morning trajectories indeed contain weak but exploitable predictive signal.*



## What's in the notebook

- Problem description & data understanding
- Exploratory comments
- Feature engineering
- LightGBM training + early stopping
- Full retraining
- Test predictions & export

All fully reproducible.

## Further Ideas
- Heavy feature engineering (PCA, rolling patterns, interaction terms, symbolic transformations) was intentionally avoided - it can uncover dataset artefacts but rarely generalizes to real markets.

- Neural models (1D-CNN, GRU/LSTM) were not used due to compute cost, tuning difficulty, and high overfitting risk on such weak signals.

- Massive feature search / FE at scale would require distributed compute (multi-VM cluster, batch pipelines), incompatible with my 8 GB RAM setup.

- Even with heavier methods, gains would be marginal: the afternoon return depends strongly on unobserved noise (news, order-book shocks, macro events).

