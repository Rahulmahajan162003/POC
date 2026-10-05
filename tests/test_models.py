import os
import pytest
import pandas as pd
import numpy as np
from models.next_cell_model import NextCellPredictor
from models.weakness_model import WeaknessPredictor


def test_next_cell_predictor_load_and_predict():
    model_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    assert os.path.exists(model_path), "Model artifact should exist"

    model = NextCellPredictor.load(model_path)
    assert model.is_trained
    assert len(model.cell_classes) == 6

    # Create dummy single row dataframe matching feature names
    dummy_data = {feat: [0.0] for feat in model.feature_names}
    dummy_data["current_rsrp_dbm"] = [-75.0]
    dummy_data["current_sinr_db"] = [15.0]
    dummy_data["neighbor_Cell_B_rsrp"] = [-65.0]
    df_sample = pd.DataFrame(dummy_data)

    probs = model.predict_probabilities(df_sample)
    assert isinstance(probs, dict)
    assert len(probs) == 6
    assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)


def test_weakness_predictor_load_and_predict():
    model_path = os.path.join("artifacts", "models", "weakness_model.joblib")
    assert os.path.exists(model_path), "Model artifact should exist"

    model = WeaknessPredictor.load(model_path)
    assert model.is_trained

    dummy_data = {feat: [0.0] for feat in model.feature_names}
    dummy_data["current_rsrp_dbm"] = [-115.0]  # Very weak
    df_sample = pd.DataFrame(dummy_data)

    risk = model.predict_weakness_risk(df_sample)
    assert 0.0 <= risk <= 1.0
