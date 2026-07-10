from __future__ import annotations

from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value

from src.config import load_config

CONFIG = load_config()


def optimize_allocation(predicted_energy: float, loads: dict) -> dict:
    allocation = {name: 0.0 for name in loads}
    if predicted_energy < 0:
        return {"status": "Infeasible", "allocation": allocation, "objective_value": 0.0}

    prob = LpProblem("CivicGrid_Allocation", LpMaximize)
    x = {
        name: LpVariable(name, lowBound=0, upBound=float(load["demand"]))
        for name, load in loads.items()
    }
    prob += lpSum(float(loads[name]["priority"]) * x[name] for name in loads)
    prob += lpSum(x[name] for name in loads) <= float(predicted_energy), "available_energy"
    for name, load in loads.items():
        minimum = float(load.get("min_required", 0))
        if minimum > 0:
            prob += x[name] >= minimum, f"minimum_{name}"

    prob.solve(PULP_CBC_CMD(msg=0))
    status = LpStatus.get(prob.status, "Unknown")
    if status != "Optimal":
        return {"status": status, "allocation": allocation, "objective_value": 0.0}
    decimals = int(CONFIG["optimization"]["round_decimals"])
    allocation = {name: round(float(x[name].varValue or 0.0), decimals) for name in loads}
    return {
        "status": status,
        "allocation": allocation,
        "objective_value": round(float(value(prob.objective) or 0.0), decimals),
    }
