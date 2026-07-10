from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import get_feature_columns, load_config, project_path
from src.preprocessing import TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()
FEATURE_COLUMNS = get_feature_columns(CONFIG)
MODEL_PATH = project_path(CONFIG["paths"]["trained_model"])


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train_model(df: pd.DataFrame, model_path: str | Path | None = MODEL_PATH) -> dict:
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(CONFIG["model"]["test_size"]),
        random_state=int(CONFIG["model"]["random_state"]),
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    baseline_column = CONFIG["model"]["baseline_group_column"]
    hour_means = y_train.groupby(x_train[baseline_column]).mean()
    fallback = float(y_train.mean())
    baseline_predictions = x_test[baseline_column].map(hour_means).fillna(fallback)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
    return {
        "model": model,
        "metrics": {
            "Linear Regression": _metrics(y_test, predictions),
            "Hour-of-day baseline": _metrics(y_test, baseline_predictions),
        },
        "test_frame": x_test.assign(actual=y_test, predicted=predictions).reset_index(drop=True),
    }


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Model has not been trained yet.")
    return joblib.load(model_path)


def predict_energy(features_dict: dict, model=None) -> float:
    model = model or load_model()
    features = pd.DataFrame([{column: features_dict[column] for column in FEATURE_COLUMNS}])
    return round(float(model.predict(features)[0]), 3)
