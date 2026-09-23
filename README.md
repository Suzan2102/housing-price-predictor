# 🏠 Housing Price Predictor — Linear Regression

A Streamlit web app that prices apartments using a **linear regression** model trained on `Project housing data .csv` (2,999 sold apartments). Machine-learning exercise, part A.

## Method
- **Split:** 80% training (2,399 rows) / 20% test (600 rows), `random_state=42`
- **Encoding:**
  - `waterfront` is already a 0/1 dummy and is used as-is
  - `view` (0–4) and `condition` (1–5) are categorical indexes, one-hot encoded with `drop_first` (baselines: view=0, condition=1)
  - `sqft_above` is dropped because `sqft_living = sqft_above + sqft_basement` exactly (perfect multicollinearity)
- **Model:** `sklearn.linear_model.LinearRegression`

## Model evaluation — R-Squared (R²)
The model is evaluated with **R²** on the 20% test set (apartments not seen during training).

| | Train (80%) | **Test (20%)** |
|---|---|---|
| **R²** | 0.649 | **0.584** |
| Adjusted R² | 0.647 | 0.573 |

- The model explains **~58%** of the variance in apartment prices on unseen data.
- The small drop from train (0.649) to test (0.584) means there is no significant overfitting.
- Supplementary test metrics: MAE $148,055 · RMSE $204,654 · MAPE 32.3%.

## Prediction example
A new apartment: 3 bedrooms, 2 bathrooms, 1,800 sqft living (1,400 above + 400 basement), 6,000 sqft lot, 1 floor, no waterfront, view 0, condition 3, built 1985.
→ **Predicted price ≈ $395,000.** This is reasonable: the 484 apartments in the data with 3 bedrooms and similar living area (±15%) have a median price of $405,000 (IQR $304K–$517K).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `model.py` — data loading, encoding, training, evaluation, prediction
- `app.py` — Streamlit UI (pricing form, metrics, charts, coefficients, data)
- `Project housing data .csv` — dataset
