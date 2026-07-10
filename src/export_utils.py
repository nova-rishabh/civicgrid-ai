from __future__ import annotations

import pandas as pd


def allocation_to_csv(predicted_energy: float, loads: dict, optimized: dict, baseline: dict) -> bytes:
    rows = []
    for name, load in loads.items():
        demand = float(load["demand"])
        rows.append(
            {
                "load": name,
                "demand_kwh": demand,
                "priority": load["priority"],
                "min_required_kwh": load["min_required"],
                "optimized_kwh": optimized.get(name, 0.0),
                "baseline_kwh": baseline.get(name, 0.0),
                "optimized_pct_of_demand": round(optimized.get(name, 0.0) / demand * 100, 2)
                if demand
                else 0,
                "predicted_energy_kwh": predicted_energy,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

