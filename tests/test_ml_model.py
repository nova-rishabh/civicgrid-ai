from src.config import load_config
from src.data_generator import generate_training_data
from src.ml_model import predict_energy, train_model


def test_model_prediction_sane_range():
    config = load_config()
    scenario = config["scenarios"][config["app"]["default_scenario"]]["features"]
    df = generate_training_data()
    result = train_model(df, None)
    prediction = predict_energy(scenario, result["model"])
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 100
