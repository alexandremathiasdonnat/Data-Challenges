# ENS Data Challenges

**Hi there! 👋**

This repository gathers my work on ENS Data Challenges, a series of competitive data science problems based on real-world datasets provided by companies, public institutions, and research labs.

All challenges are hosted on the official ENS platform:  
https://challengedata.ens.fr

They emphasize realistic constraints: signal research, noisy data, hidden test labels, strict evaluation protocols, and leaderboard-based benchmarking.

## Focus & Interests

My initial focus has been on financial and market-related challenges, reflecting my primary interest in quantitative finance and applied modeling under uncertainty.

Going forward, I am particularly interested in expanding toward challenges in:
- **Energy & Environmental systems**
- **Healthcare & Neuroscience**
- **Industrial systems**
- **Oceanography**

These domains share similar difficulties: weak signals, complex dynamics, and the need for robust, interpretable models.

## Completed Challenges

### Automatic Title Extraction from Financial Documents (2023)  
**AMF - ENS Data Challenge**  
Reconstructing the full hierarchical structure of French-listed companies' financial reports (titles and levels 1–8) using only noisy OCR text blocks and layout metadata.  
**Result:** Rank **6 / 32**  
https://challengedata.ens.fr/challenges/86

---

### High-Frequency Market Microstructure Classification (2024)  
**Capital Fund Management - ENS Data Challenge**  
Identifying the originating stock of short anonymous order-book event sequences using a bidirectional GRU architecture.  
**Result:** Rank **40 / 187**  
https://challengedata.ens.fr/challenges/146

---

### Predicting End-of-Session US Equity Returns (2023)  
**Capital Fund Management - ENS Data Challenge**  
Predicting afternoon price direction from morning intraday return trajectories using a regularized gradient-boosting pipeline.  
**Result:** Rank **130 / 260**  
https://challengedata.ens.fr/challenges/84

---

### Fire Pure Premium Prediction (2025)  
**Crédit Agricole Assurances - ENS Data Challenge**  
Hybrid actuarial + machine learning modeling of rare but severe agricultural fire risk to predict fair insurance premiums.  
**Result:** Rank **28 / 253**  
https://challengedata.ens.fr/challenges/161

---

### Electricity Futures Spread Forecasting (2023)  
**QRT - ENS Data Challenge**  
Forecasting the FR–DE electricity futures spread using physical, meteorological, and fuel indicators under a rank-based (Spearman) evaluation.  
**Result:** Rank **352 / 1,070**   
https://challengedata.ens.fr/challenges/97

---

### Learning Linear Factors for Equity Returns (2022)  
**QRT - ENS Data Challenge**  
Learning orthonormal linear factors on the Stiefel manifold to predict cross-sectional equity returns from historical price data.  
**Result:** Rank **140 / 360**  
https://challengedata.ens.fr/challenges/72

---

## My Methodological Philosophy

Across challenges, my dominant effort lies in representation building rather than model tuning.

Most of the work is spent on:
- understanding the structure of the problem and the evaluation metric,
- studying domain-specific feature engineering methods and existing approaches,
- exploring, testing, and validating feature constructions tailored to each dataset,
- iterating extensively on inputs before selecting appropriate modeling techniques.

Feature engineering typically represents the majority of the modeling effort, as it defines the information available to downstream machine learning predictive models. If too noisy, even the best ML models would not be able to extract any consistent predictive signal.

Model choices and learning strategies are then adapted to the resulting representation and to the specific constraints of each challenge. Depending on the problem, this may range from lightweight regularized baselines to more compute-intensive architectures when justified.

In general, my personnal objective is not leaderboard optimization through computing brute force, but building inputs and models that reflect research-realistic underlying structure of the data and generalize beyond the benchmark split.

---
**Alexandre Mathias Donnat**

