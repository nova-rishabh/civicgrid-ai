from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_feature_columns, get_target_column, load_config

CONFIG = load_config()
FEATURE_COLUMNS = get_feature_columns(CONFIG)
TARGET_COLUMN = get_target_column(CONFIG)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_training_data(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    checks = {}
    for feature in CONFIG["data_generation"]["features"]:
        column = feature["name"]
        if feature["type"] == "binary":
            checks[column] = df[column].isin([0, 1])
        else:
            checks[column] = df[column].between(feature["min"], feature["max"])
    checks[TARGET_COLUMN] = df[TARGET_COLUMN].ge(CONFIG["data_generation"]["clip_target_lower"])
    for column, valid in checks.items():
        if not bool(valid.all()):
            errors.append(f"{column} contains values outside the expected range.")
    return errors


def validate_features(df: pd.DataFrame) -> list[str]:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        return [f"Missing feature columns: {', '.join(missing)}"]
    errors: list[str] = []
    for feature in CONFIG["data_generation"]["features"]:
        column = feature["name"]
        if feature["type"] == "binary":
            valid = df[column].isin([0, 1])
        else:
            valid = df[column].between(feature["min"], feature["max"])
        if not bool(valid.all()):
            errors.append(f"{column} contains values outside the expected range.")
    return errors
