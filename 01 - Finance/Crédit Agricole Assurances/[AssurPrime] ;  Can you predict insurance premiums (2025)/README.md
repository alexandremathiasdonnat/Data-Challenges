# Fire Pure Premium Prediction

## Crédit Agricole Assurances - ENS Data Challenge

Predicting fire risk in the agricultural world is a fascinating paradox.

Most farms will never experience a fire. But when one does, the financial cost can be catastrophic: destroyed buildings, lost equipment, halted production, months of recovery.

This rare-but-severe nature makes fire insurance one of the most challenging risks to model. That is precisely the mission Crédit Agricole Assurances entrusted to participants in this ENS data challenge: 

**Can we predict, for each agricultural contract, the fair "pure premium" reflecting both the rarity and the severity of fire losses?**

This repository is my contribution to that challenge.

##  Understanding the target

The "pure premium" for fire risk combines two core actuarial dimensions:

- **How often fire events occur** (frequency)
- **How costly they are when they occur** (severity)

Together, they form the expected financial impact of fire for each contract, the foundation of actuarial pricing.

##  A mathematically guided approach

Although the notebook gives the full technical details, the intuition rests on two complementary modelling paths:

### 1) The actuarial decomposition model

This approach mirrors standard insurance pricing logic:

- **Fire frequency** is treated as a rare event count, naturally modelled using the Poisson distribution, a classical tool for modelling the number of occurrences of an event over an exposure period.
- **Fire severity** is stabilised using logarithmic transformations, because claim amounts are extremely skewed and sometimes extreme.

This decomposed structure reflects how insurers traditionally think about risk.

### 2) A machine-learning track

A second model directly predicts the expected premium from raw features, without forcing any manual decomposition. This allows the algorithm to learn complex interactions that the actuarial structure cannot explicitly express.

##  When both worlds collaborate

The core idea is simple but powerful:

Instead of choosing between structure or flexibility, we make both models collaborate.

The two predictions are blended, and a final calibration step adjusts any global bias so that the output aligns with historical patterns.

This hybrid approach proved extremely effective: mathematics, probability modelling, and machine learning working together, not separately.

##  How is the final model performance evaluated ?

The true fire premiums for the test set are hidden by the platform : *https://challengedata.ens.fr*

We submit our predictions, and the system computes:

- A **Root Mean Squared Error (RMSE)** between our predicted premiums and the real premiums (kept secret)
- The lower the RMSE, the better the ranking

## My results

After refinement, blending and calibration, the model reaches:

-  **RMSE :** 5600.57
-  **Rank :** 28 / 253 participants



##  What's in the notebook

The notebook walks through the full solution:

- Data treatment and encoding
- Rare-event modelling via Poisson logic
- Severity stabilisation
- Hybrid actuarial + ML modelling
- Blending and calibration
- Final submission for the ENS platform

Accessible, structured, and faithful to insurance modelling philosophy.

##  Want to explore further?

The notebook closes with improvement ideas, including:

- Extending distributional assumptions
- Regularisation strategies
- Testing alternative blending schemas
- Richer feature engineering
- Deep models (and their risks)
