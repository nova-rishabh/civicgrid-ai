from __future__ import annotations

from src.config import load_config

CONFIG = load_config()


def proportional_equal_cut(predicted_energy: float, loads: dict) -> dict[str, float]:
    total_demand = sum(float(load["demand"]) for load in loads.values())
    if total_demand <= 0:
        return {name: 0.0 for name in loads}
    scale = min(max(predicted_energy / total_demand, 0.0), 1.0)
    return {name: float(load["demand"]) * scale for name, load in loads.items()}


def priority_score(allocation: dict[str, float], loads: dict) -> float:
    return sum(float(loads[name]["priority"]) * float(value) for name, value in allocation.items())


def comparison_metrics(
    optimized: dict[str, float],
    baseline: dict[str, float],
    loads: dict,
    critical_load: str | None = None,
) -> dict:
    critical_load = critical_load or CONFIG["optimization"]["critical_load"]

    def emergency_pct(allocation: dict[str, float]) -> float:
        required = float(loads[critical_load]["min_required"])
        if required == 0:
            return 100.0
        return min(100.0, allocation.get(critical_load, 0.0) / required * 100)

    def fully_satisfied(allocation: dict[str, float]) -> int:
        return sum(
            1
            for name, load in loads.items()
            if allocation.get(name, 0.0) >= float(load["demand"]) - 1e-6
        )

    return {
        f"{loads[critical_load].get('label', critical_load).title()} minimum met (%)": {
            "optimized": emergency_pct(optimized),
            "baseline": emergency_pct(baseline),
        },
        "Priority-weighted score": {
            "optimized": priority_score(optimized, loads),
            "baseline": priority_score(baseline, loads),
        },
        "Loads fully satisfied": {
            "optimized": fully_satisfied(optimized),
            "baseline": fully_satisfied(baseline),
        },
    }
