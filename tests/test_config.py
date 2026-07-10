from src.config import get_feature_columns, get_loads, load_config


def test_config_has_valid_dynamic_contracts():
    config = load_config()
    features = get_feature_columns(config)
    loads = get_loads(config)

    assert len(config["app"]["pages"]) >= 5
    assert config["optimization"]["critical_load"] in loads
    assert config["model"]["baseline_group_column"] in features
    assert config["data_generation"]["target"] not in features

    for scenario in config["scenarios"].values():
        assert set(features).issubset(scenario["features"])
