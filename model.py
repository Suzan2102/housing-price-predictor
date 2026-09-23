"""Linear regression model for apartment pricing (80% train / 20% test)."""
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

# view (0-4) and condition (1-5) are categorical indexes -> one-hot encoded.
CATEGORICAL = ["view", "condition"]
# sqft_above is dropped: sqft_living = sqft_above + sqft_basement exactly,
# so keeping all three would make the regression perfectly collinear.
NUMERIC = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
           "waterfront", "sqft_basement", "yr_built"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def encode(df, columns=None):
    """One-hot encode categorical columns (drop_first avoids the dummy trap)."""
    X = df[NUMERIC + CATEGORICAL].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(int)
    X = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def train(df):
    X = encode(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = LinearRegression().fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }

    return {
        "model": model,
        "columns": list(X.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_metrics": metrics(y_train, pred_train),
        "test_metrics": metrics(y_test, pred_test),
        "test_results": pd.DataFrame({"actual": y_test.values, "predicted": pred_test}),
        "coefficients": pd.Series(model.coef_, index=X.columns),
        "intercept": model.intercept_,
    }


def predict(result, features: dict):
    X = encode(pd.DataFrame([features]), columns=result["columns"])
    return float(result["model"].predict(X)[0])


if __name__ == "__main__":
    res = train(load_data())
    print(f"train={res['n_train']} test={res['n_test']}")
    print("train:", res["train_metrics"])
    print("test: ", res["test_metrics"])
    print(res["coefficients"].round(1))
    example = dict(bedrooms=3, bathrooms=2, sqft_living=1800, sqft_lot=6000,
                   floors=1, waterfront=0, view=0, condition=3,
                   sqft_above=1400, sqft_basement=400, yr_built=1985)
    print("example prediction:", round(predict(res, example)))
