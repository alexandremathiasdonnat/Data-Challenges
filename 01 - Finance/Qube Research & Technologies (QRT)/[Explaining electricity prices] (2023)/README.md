# Electricity Futures Contracts Spread Forecasting (2023)

## QRT - ENS Data Challenge 

![Electric Grid](https://biztechmagazine.com/sites/biztechmagazine.com/files/styles/cdw_hero/public/articles/202110/electric%20grid%20hero.jpg?itok=4nD1-GFZ)

*Can we forecast tomorrow’s FR–DE electricity spread (the day-to-day variation of a futures contract on electricity price) using only today’s physical, meteorological, and fuel indicators?*

*Our avaible data provides, for each country and each day, dozens of system drivers: wind and solar production, residual load, cross-border flows, gas/coal/CO₂ prices, temperature, wind speed, rainfall, and more. The target is the spread variation, a tiny and extremely noisy signal extracted from electricity futures.*

**This makes the problem uniquely difficult:**
- Very low signal-to-noise ratio :  the true predictive signal is tiny compared to randomness.
- Different structural regimes in France vs Germany.
- Missing key variables (nuclear availability, outages, intraday shocks).
- Spearman correlation as evaluation, which ignores exact values and evaluates only the ranking : weakens traditional ML precision, amplifies noise, and strongly favors overfitting when data is limited.

**Despite this, careful feature engineering and robust validation can extract part of the underlying structure.**

This repository contains my full modelling framework.

## Understanding the target

Unlike classical regression, the goal is not to minimize RMSE.
The evaluation of predictions uses a Spearman rank correlation, meaning:
- Only the ordering of predicted values matters.
- Any monotonic transformation is irrelevant.
- Capturing the relative dynamics of the spread is the essence of the problem.

This has two implications:

- The true signal is extremely noisy.
- Classical ML struggles unless carefully steered toward rank behaviour.

## A mathematically-guided approach

Although the notebook contains the full technical pipeline, the intuition is built on three pillars.

**1) Physical feature construction**
Raw variables (wind, renewables, residual load, gas/coal/CO₂, imports…) are transformed into: spreads (FR – DE), tensions after renewables, rolling volatilities (3–7–14 days), regime indicators, local z-scores, lagged variations, cross-interactions ("tightness × fuels", "renewables × wind")
This reconstructs the physical mechanics of short-term electricity formation.

**2) Robust target treatment**
Because the target contains sharp regime flips and micro-structure noise: a rolling median smoothing stabilizes extremes, country-wise winsorisation limits outliers, a local rolling standard deviation normalisation adjusts for heteroskedasticity, optional tanh compression keeps the supervised target numerically stable.
This converts the chaotic target into a usable signal.

**3) Machine-learning prediction and blending**

Two complementary models are trained:  **LightGBM** (non-linear, strong on physical interactions) & **Ridge regression** (linear, smooth, robust to noise)
Both are evaluated out-of-fold using Spearman, then blended using an optimal weight:

$$\hat{y}_{blend} = w \cdot \hat{y}_{LGBM} + (1-w) \cdot \hat{y}_{Ridge}$$

This hybrid approach gives the best stability across folds.

Final models are refit on the full training set and applied to the test contracts.
The predicted rank-preserving values are exported for submission.

## Evaluation

Predictions are sent to the platform, which computes: the Spearman correlation between predicted ranking and hidden true ranking (a higher Spearman indicates better ordering and better forecasting performance) :  *https://challengedata.ens.fr*

## My results

Using a hybrid pipeline combining physical feature engineering, time-series structure, LightGBM modelling and rank-based blending, the model reached:

- **OOF Spearman (internal CV): ≈ 0.37**
- **Public leaderboard Spearman: 0.2284**
- **Rank: 352 / 1 070 participants**

This gap between internal and external performance is expected: the challenge exhibits very high inherent noise, making purely predictive ML approaches reach a natural ceiling at moderate Spearman values.

The model remains fully consistent, stable across folds, and competitive, but further gains would likely require non-predictive approaches (reverse engineering of the public score, heuristic rank reshaping, brute-force order search).
These techniques can improve leaderboard position but are less realistic actuarially, which is why they were not used here.

## What’s inside the notebook?

Data loading & inspection, Physical feature engineering (load, renewables, fuels, spreads…), Temporal structure (lags, deltas, rolling windows), Target smoothing + winsorization + normalization, Rank-preserving ML training, Blending and calibration, Final prediction and submission
All steps are reproducible.

## Limits & further ideas

Although the method captures meaningful physical structure, the target remains highly noise-dominated.
Beyond moderate correlations, traditional supervised ML reaches a natural ceiling.
Further improvements would likely require: richer domain-specific signals, regime-switching models, direct rank-optimization techniques, hybrid stochastic/ML approaches
The current notebook focuses on predictive, interpretable, and realistic modelling, within the constraints of the dataset.

**Alexandre Mathias DONNAT, Sr**