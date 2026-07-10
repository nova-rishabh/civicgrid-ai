from src.baseline import comparison_metrics, proportional_equal_cut
from src.config import load_config
from src.data_generator import LOAD_DEMANDS, generate_training_data
from src.ml_model import predict_energy, train_model
from src.optimizer import optimize_allocation


def test_full_workflow_runs_without_exceptions():
    config = load_config()
    scenario = config["scenarios"][config["app"]["default_scenario"]]["features"]
    df = generate_training_data()
    model_result = train_model(df, None)
    predicted = predict_energy(scenario, model_result["model"])
    optimized = optimize_allocation(predicted, LOAD_DEMANDS)
    baseline = proportional_equal_cut(predicted, LOAD_DEMANDS)
    metrics = comparison_metrics(
        optimized["allocation"],
        baseline,
        LOAD_DEMANDS,
        config["optimization"]["critical_load"],
    )
    assert optimized["status"] == "Optimal"
    assert len(metrics) >= 3


def test_prediction_value_is_passed_to_optimizer():
    total_minimum = sum(load["min_required"] for load in LOAD_DEMANDS.values())
    total_demand = sum(load["demand"] for load in LOAD_DEMANDS.values())
    low = optimize_allocation(total_minimum + 1, LOAD_DEMANDS)["allocation"]
    high = optimize_allocation(total_demand, LOAD_DEMANDS)["allocation"]
    assert sum(high.values()) > sum(low.values())
    assert high != low
