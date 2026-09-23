# 🏠 Housing Price Predictor — Linear Regression

**🚀 Live app: https://suzan-housing-price.streamlit.app/**

A Streamlit web app that prices apartments using a **linear regression** model trained on `Project housing data .csv` (2,999 sold apartments). Machine-learning exercise, part A.

## Method
- **Split:** 80% training (2,399 rows) / 20% test (600 rows), `random_state=42`
- **Encoding:**
  - `waterfront` is already a 0/1 dummy and is used as-is
  - `view` (0–4) and `condition` (1–5) are one-hot encoded with fixed levels (baselines: view=0, condition=1)
  - `sqft_above` is dropped because `sqft_living = sqft_above + sqft_basement` exactly (perfect multicollinearity)
- **Model:** ordinary least squares `LinearRegression`, in two versions:
  - **Baseline:** `price ~ raw features`
  - **Improved (used by the app):** `sqrt(price) ~ engineered features`
    - log of the areas, house age and age², bedrooms², bathrooms per bedroom, a has-basement flag
    - interactions: area×view, area×condition, area×waterfront, age×condition
    - the prediction is transformed back to dollars (`pred² + σ²`) before scoring

## Model evaluation — R-Squared (R²)
R² is computed in dollars on the 20% test set.

| Model | Train R² | **Test R²** | Adj. test R² | Mean test R² over 30 random 80/20 splits |
|---|---|---|---|---|
| Baseline | 0.649 | 0.584 | 0.573 | 0.611 |
| **Improved** | 0.713 | **0.623** | 0.607 | **0.667** (better in 30/30 splits) |

**Why R² can't go much higher:** the dataset has no location column (zip code or neighbourhood). Location is the strongest driver of house prices, and no model built on these columns can explain it.

## Prediction example
A new apartment: 3 bedrooms, 2 bathrooms, 1,800 sqft living (1,400 above + 400 basement), 6,000 sqft lot, 1 floor, no waterfront, view 0, condition 3, built 1985.
→ **Predicted price ≈ $430,000.** This is reasonable: the 484 apartments in the data with 3 bedrooms and similar living area (±15%) have a median price of $405,000 (IQR $304K–$517K).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `model.py` — data loading, encoding, training, evaluation, prediction
- `app.py` — Streamlit UI (pricing form, metrics, charts, coefficients, data)
- `Project housing data .csv` — dataset
