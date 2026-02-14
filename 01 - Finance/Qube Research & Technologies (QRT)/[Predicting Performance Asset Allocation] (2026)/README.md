# Learning Directional Signals from Systematic Portfolio Allocations (2026)

## QRT – ENS Data Challenge
*Meta-Decision Modelling for Allocation Trust/Fade*
![figure](figure.jpg)

*Can we determine whether a systematic portfolio allocation should be trusted or faded using only its recent behaviour?*

Each row of the dataset represents a systematic allocation over multiple assets. We observe its recent performance and liquidity dynamics, and must decide whether its next return will be positive or negative.

This is a meta-decision problem:
- We do not predict asset returns
- We do not construct portfolios
- We decide whether to follow or invert an already constructed allocation

The signal-to-noise ratio is extremely weak. Directional accuracy barely exceeds 50%, making the challenge fundamentally difficult.

## Mathematical Formulation

### Context

$$r_{S,t+1} = \sum_{i=1}^{M} w_{S,t,i} \cdot r_{i,t+1}$$

Where:
- $w_{S,t,i}$ = allocation weights
- $r_{i,t+1}$ = next-day asset returns
- $r_{S,t+1}$ = allocation's realised next-day return

### Provided Data

- Last 20 daily returns: $(r_{S,t}, r_{S,t-1}, \ldots, r_{S,t-19})$
- Signed volume measures
- Turnover statistics
- Allocation group
- Target return $r_{S,t+1}$

### Learning Objective

$$f(x_{S,t}) \in \{0, 1\}$$

Where: $1$ = trust allocation, $0$ = fade allocation

## Evaluation Metric

**Binary Accuracy:**

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{sign}(\hat{r}_i) = \text{sign}(r_i)]$$

**Implications:**
- Magnitude does not matter
- Calibration does not matter
- Only directional correctness matters

**Baseline Performance:**
- Always long: ≈ 0.507
- Winning scores: ≈ 0.52

## Feature Engineering

### Rolling Return Geometry

For windows $k \in \{3, 5, 10, 20\}$:
- Mean return
- Volatility
- Min / Max
- Sign fraction
- Entropy of sign distribution
- Linear trend slope

### Scale-Free Normalisation

$${ret\_norm} = \frac{r_{S,t}}{\sigma_{S,t}}$$

Where $\sigma_{S,t}$ is the rolling volatility over the recent window.

Normalises returns by recent volatility to enable fair comparison across allocations with different risk profiles.

Removes volatility scaling differences between allocations.

### Liquidity Structure

For signed volumes:
- Rolling mean
- Volatility
- Extremes

### Limited Interactions

Only economically interpretable interactions:
- Momentum × Turnover
- Normalised return × Entropy

## Model Architecture

**Algorithm:** LightGBM with binary objective

**Hyperparameters:**
- `learning_rate = 0.03`
- `num_leaves = 128`
- `min_data_in_leaf = 200`
- `feature_fraction = 0.8`
- `bagging_fraction = 0.8`
- L2 regularisation

**Validation Strategy:** Strict time-based split
- Train: first 80% of timestamps
- Validation: last 20%
- Final model: refit on full training data using optimal boosting rounds

## My Results

- **Public leaderboard score:** 0.5154
- **Validation accuracy:** ≈ 0.526
- **Final rank** : 79/360 on https://challengedata.ens.fr/participants/challenges/167/


Extensive attempts at feature inflation, cross-sectional leakage tricks, regime priors, and tail-risk modelling did not materially improve performance, indicating proximity to the signal frontier.

## Insights

Central principle: When signal is weak, stability beats complexity.

- Directional persistence and volatility scaling matter more than extreme statistical sophistication
- Added complexity increases variance rather than predictive power
- Small structural improvements (0.002–0.004 in accuracy) require strict leakage control, robust validation, and controlled feature engineering

## What's Inside the notebook

- Data loading and preprocessing
- Rolling statistical feature construction
- Time-series safe split
- LightGBM training with early stopping
- Full refit and submission generation
- Experimental variants

## Further Ideas and extensions

- Rolling 5-fold temporal cross-validation
- Multi-seed ensembling
- Probability calibration via meta-model
- Threshold optimisation under regime segmentation
- Hybrid linear + boosting ensemble
- Bayesian posterior smoothing

---

***Alexandre Mathias DONNAT, Sr***