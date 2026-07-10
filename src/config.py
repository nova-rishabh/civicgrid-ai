from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default_config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "app",
        "theme",
        "paths",
        "data_generation",
        "model",
        "optimization",
        "loads",
        "scenarios",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    feature_names = get_feature_columns(config)
    if len(config["app"].get("pages", [])) < 5:
        raise ValueError("The app config must define at least five page labels.")

    coefficient_names = set(config["data_generation"].get("coefficients", {}))
    unknown_coefficients = coefficient_names.difference(feature_names)
    if unknown_coefficients:
        raise ValueError(f"Coefficient columns are not configured as features: {unknown_coefficients}")

    critical_load = config["optimization"]["critical_load"]
    if critical_load not in config["loads"]:
        raise ValueError(f"Critical load '{critical_load}' is not present in loads.")

    baseline_column = config["model"]["baseline_group_column"]
    if baseline_column not in feature_names:
        raise ValueError(f"Baseline group column '{baseline_column}' is not configured as a feature.")

    if config["model"]["type"] != "LinearRegression":
        raise ValueError("Only LinearRegression is currently supported by the offline prototype.")

    if config["optimization"]["solver"] != "PULP_CBC_CMD":
        raise ValueError("Only PULP_CBC_CMD is currently supported by the offline prototype.")

    for name, load in config["loads"].items():
        for field in ["demand", "priority", "min_required"]:
            if field not in load:
                raise ValueError(f"Load '{name}' is missing '{field}'.")
            if float(load[field]) < 0:
                raise ValueError(f"Load '{name}' has negative '{field}'.")
        if float(load["min_required"]) > float(load["demand"]):
            raise ValueError(f"Load '{name}' minimum cannot exceed demand.")

    for scenario_name, scenario in config["scenarios"].items():
        missing_features = [column for column in feature_names if column not in scenario["features"]]
        if missing_features:
            raise ValueError(f"Scenario '{scenario_name}' is missing: {', '.join(missing_features)}")


def get_feature_columns(config: dict[str, Any]) -> list[str]:
    return [feature["name"] for feature in config["data_generation"]["features"]]


def get_target_column(config: dict[str, Any]) -> str:
    return config["data_generation"]["target"]


def get_loads(config: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    config = config or load_config()
    loads = deepcopy(config["loads"])
    for load in loads.values():
        load["demand"] = float(load["demand"])
        load["priority"] = float(load["priority"])
        load["min_required"] = float(load["min_required"])
    return loads


def project_path(relative_path: str | Path) -> Path:
    return ROOT / relative_path
