from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import get_feature_columns, get_loads, load_config, project_path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()
DATA_DIR = project_path(CONFIG["paths"]["data_dir"])
MODELS_DIR = project_path(CONFIG["paths"]["models_dir"])

FEATURE_COLUMNS = get_feature_columns(CONFIG)
LOAD_DEMANDS = get_loads(CONFIG)


def _generate_feature(rng: np.random.Generator, feature: dict, rows: int):
    minimum = feature["min"]
    maximum = feature["max"]
    feature_type = feature["type"]
    if feature_type == "binary":
        return rng.integers(0, 2, rows)
    if feature_type == "int":
        return rng.integers(int(minimum), int(maximum) + 1, rows)
    if feature_type == "float":
        return rng.uniform(float(minimum), float(maximum), rows).round(2)
    raise ValueError(f"Unsupported feature type: {feature_type}")


def _generate_civic_microgrid_features(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    hour = rng.integers(0, 24, rows)
    is_weekend = rng.choice([0, 1], size=rows, p=[5 / 7, 2 / 7])

    daylight = np.sin((hour - 6) / 12 * np.pi)
    daylight = np.clip(daylight, 0, None)
    cloud_factor = rng.beta(5, 2, rows)
    solar_irradiance = daylight * cloud_factor * rng.normal(900, 70, rows)
    solar_irradiance = np.clip(solar_irradiance, 0, 1000).round(2)

    morning_peak = np.exp(-0.5 * ((hour - 8) / 2.2) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 18) / 2.6) ** 2)
    leisure_peak = np.exp(-0.5 * ((hour - 16) / 4.0) ** 2)
    weekday_pattern = 45 + 185 * morning_peak + 220 * evening_peak
    weekend_pattern = 65 + 260 * leisure_peak
    footfall_mean = np.where(is_weekend == 1, weekend_pattern, weekday_pattern)
    footfall_count = rng.poisson(np.clip(footfall_mean, 0, 500))
    footfall_count = np.clip(footfall_count, 0, 500)

    temperature = 22 + 9 * daylight + rng.normal(0, 2.8, rows) - 1.2 * is_weekend
    temperature = np.clip(temperature, 15, 45).round(2)

    return pd.DataFrame(
        {
            "hour_of_day": hour,
            "solar_irradiance": solar_irradiance,
            "footfall_count": footfall_count,
            "temperature": temperature,
            "is_weekend": is_weekend,
        }
    )


def generate_training_data(
    seed: int | None = None,
    rows: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or CONFIG
    generation = config["data_generation"]
    seed = generation["seed"] if seed is None else seed
    rows = generation["rows"] if rows is None else rows
    rng = np.random.default_rng(seed)
    feature_columns = get_feature_columns(config)
    if set(feature_columns) == {
        "hour_of_day",
        "solar_irradiance",
        "footfall_count",
        "temperature",
        "is_weekend",
    }:
        data = _generate_civic_microgrid_features(rng, rows)
    else:
        data = pd.DataFrame(
            {
                feature["name"]: _generate_feature(rng, feature, rows)
                for feature in generation["features"]
            }
        )
    target = float(generation["intercept"])
    for column, coefficient in generation["coefficients"].items():
        target = target + float(coefficient) * data[column]
    noise = rng.normal(0, float(generation["noise_std"]), rows)
    data[generation["target"]] = (
        target + noise
    ).clip(lower=float(generation["clip_target_lower"])).round(int(generation["round_decimals"]))
    return data


def write_default_data(config: dict | None = None) -> None:
    config = config or CONFIG
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    generate_training_data(config=config).to_csv(DATA_DIR / "training_data.csv", index=False)
    for scenario in config["scenarios"].values():
        row = dict(scenario["features"])
        for optional_field in ["demand_multiplier", "forced_predicted_energy"]:
            if optional_field in scenario:
                row[optional_field] = scenario[optional_field]
        pd.DataFrame([row]).to_csv(DATA_DIR / scenario["filename"], index=False)
    (MODELS_DIR / ".gitkeep").touch()


if __name__ == "__main__":
    write_default_data()
