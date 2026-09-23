"""Linear regression model for apartment pricing (80% train / 20% test).

Two models are trained on the same split:
- baseline: price ~ raw features (+ one-hot view/condition)
- improved: sqrt(price) ~ engineered features (log areas, age, interactions)
Both are ordinary least-squares LinearRegression and are scored by R² on the
test set in *dollars*.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "Project housing data .csv"
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
REF_YEAR = 2015  # latest yr_built in the data, used to compute house age

# view (0-4) and condition (1-5) are categorical indexes -> one-hot encoded.
CATEGORICAL = ["view", "condition"]
# sqft_above is dropped: sqft_living = sqft_above + sqft_basement exactly,
# so keeping all three would make the regression perfectly collinear.
NUMERIC = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
           "waterfront", "sqft_basement", "yr_built"]

ENGINEERED = {
    "log_living": "log(sqft_living) — price grows in % with area, not in fixed $",
    "log_living2": "log(sqft_living)² — curvature of the area effect",
    "log_lot": "log(sqft_lot) — tames the very skewed lot sizes",
    "bedrooms2": "bedrooms² — diminishing value of extra bedrooms",
    "bath_per_bed": "bathrooms / bedrooms",
    "has_basement": "1 if sqft_basement > 0",
    "age, age2": "age = 2015 − yr_built, plus age² (old houses in prime areas)",
    "log_living_x_view": "area × view — a good view is worth more in a big house",
    "log_living_x_cond": "area × condition",
    "log_living_x_wf": "area × waterfront",
    "age_x_cond": "age × condition — a renovated old house",
}


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


CATEGORY_LEVELS = {"view": [0, 1, 2, 3, 4], "condition": [1, 2, 3, 4, 5]}


def _dummies(df):
    """One-hot with fixed levels; the first level (view=0, condition=1) is the baseline.

    Fixed levels matter: get_dummies(drop_first=True) on a single new row or on
    a split missing a category would otherwise drop the wrong column.
    """
    out = {}
    for col, levels in CATEGORY_LEVELS.items():
        for level in levels[1:]:
            out[f"{col}_{level}"] = (df[col].astype(int) == level).astype(int).values
    return pd.DataFrame(out)


def encode(df, columns=None):
    """Baseline encoding: raw numeric features + one-hot view/condition."""
    X = pd.concat([df[NUMERIC].reset_index(drop=True),
                   _dummies(df).reset_index(drop=True)], axis=1)
    return X if columns is None else X.reindex(columns=columns, fill_value=0)


def engineer(df, columns=None):
    """Improved encoding: log areas, age, squared terms and interactions."""
    d = df.reset_index(drop=True)
    L = np.log(d.sqft_living)
    age = REF_YEAR - d.yr_built
    X = pd.DataFrame({
        "bedrooms": d.bedrooms, "bedrooms2": d.bedrooms ** 2,
        "bathrooms": d.bathrooms, "bath_per_bed": d.bathrooms / d.bedrooms.clip(lower=1),
        "log_living": L, "log_living2": L ** 2, "log_lot": np.log(d.sqft_lot),
        "floors": d.floors, "waterfront": d.waterfront,
        "sqft_basement": d.sqft_basement, "has_basement": (d.sqft_basement > 0).astype(int),
        "age": age, "age2": age ** 2,
        "log_living_x_view": L * d.view, "log_living_x_cond": L * d.condition,
        "log_living_x_wf": L * d.waterfront, "age_x_cond": age * d.condition,
    })
    X = pd.concat([X, _dummies(d)], axis=1)
    return X if columns is None else X.reindex(columns=columns, fill_value=0)


def metrics(y_true, y_pred, n_features):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return {
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - n_features - 1),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


class SqrtPriceModel:
    """LinearRegression on sqrt(price); predictions returned in dollars.

    sqrt shrinks the long right tail of prices less aggressively than log,
    which keeps expensive houses from being under-predicted. Squaring a
    prediction of sqrt(price) underestimates the mean price by Var(residual),
    so that variance is added back: E[y] = pred² + σ².
    """

    def fit(self, X, y):
        self.lr = LinearRegression().fit(X, np.sqrt(y))
        self.sigma2 = float(np.var(np.sqrt(y) - self.lr.predict(X)))
        return self

    def predict(self, X):
        return self.lr.predict(X) ** 2 + self.sigma2


def _fit(df_train, df_test, kind):
    make = engineer if kind == "improved" else encode
    X_tr = make(df_train)
    X_te = make(df_test, columns=X_tr.columns)
    model = (SqrtPriceModel() if kind == "improved" else LinearRegression()).fit(X_tr, df_train[TARGET])
    return model, X_tr, X_te


def train(df, kind="improved", random_state=RANDOM_STATE):
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=random_state)
    model, X_tr, X_te = _fit(df_train, df_test, kind)
    pred_train, pred_test = model.predict(X_tr), model.predict(X_te)
    lr = model.lr if kind == "improved" else model
    k = X_tr.shape[1]
    return {
        "kind": kind,
        "model": model,
        "columns": list(X_tr.columns),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "train_metrics": metrics(df_train[TARGET], pred_train, k),
        "test_metrics": metrics(df_test[TARGET], pred_test, k),
        "test_results": pd.DataFrame({"actual": df_test[TARGET].values, "predicted": pred_test}),
        "coefficients": pd.Series(lr.coef_, index=X_tr.columns),
        "intercept": lr.intercept_,
    }


def r2_over_splits(df, n_splits=30):
    """Test R² of both models over many random 80/20 splits (robustness check)."""
    rows = []
    for rs in range(n_splits):
        df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=rs)
        row = {"split": rs}
        for kind in ("baseline", "improved"):
            model, _, X_te = _fit(df_train, df_test, kind)
            row[kind] = r2_score(df_test[TARGET], model.predict(X_te))
        rows.append(row)
    return pd.DataFrame(rows)


def predict(result, features: dict):
    make = engineer if result["kind"] == "improved" else encode
    X = make(pd.DataFrame([features]), columns=result["columns"])
    return float(result["model"].predict(X)[0])


if __name__ == "__main__":
    df = load_data()
    example = dict(bedrooms=3, bathrooms=2, sqft_living=1800, sqft_lot=6000,
                   floors=1, waterfront=0, view=0, condition=3,
                   sqft_above=1400, sqft_basement=400, yr_built=1985)
    for kind in ("baseline", "improved"):
        res = train(df, kind)
        print(f"{kind:9s} train R2={res['train_metrics']['R2']:.3f} "
              f"test R2={res['test_metrics']['R2']:.3f} adj={res['test_metrics']['Adj_R2']:.3f} "
              f"example=${predict(res, example):,.0f}")
    s = r2_over_splits(df)
    print(f"30 splits: baseline {s.baseline.mean():.3f}  improved {s.improved.mean():.3f}  "
          f"improved wins {(s.improved > s.baseline).sum()}/30")
