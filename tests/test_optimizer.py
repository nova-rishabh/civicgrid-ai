from src.config import load_config
from src.data_generator import LOAD_DEMANDS
from src.optimizer import optimize_allocation


def test_optimizer_optimal_demo():
    predicted_energy = sum(load["demand"] for load in LOAD_DEMANDS.values())
    result = optimize_allocation(predicted_energy, LOAD_DEMANDS)
    critical_load = load_config()["optimization"]["critical_load"]
    assert result["status"] == "Optimal"
    assert sum(result["allocation"].values()) <= predicted_energy + 1e-6
    assert result["allocation"][critical_load] >= LOAD_DEMANDS[critical_load]["min_required"]


def test_optimizer_infeasible_is_handled():
    critical_load = load_config()["optimization"]["critical_load"]
    infeasible_energy = LOAD_DEMANDS[critical_load]["min_required"] - 1
    result = optimize_allocation(infeasible_energy, LOAD_DEMANDS)
    assert result["status"] == "Infeasible"
    assert sum(result["allocation"].values()) == 0
