import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import FeaturePipeline
from models.next_cell_model import NextCellPredictor


def test_no_target_leakage_in_feature_columns():
    pipeline = FeaturePipeline()
    cols = pipeline.feature_columns
    
    # Must never contain ground truth targets
    assert "target_next_cell" not in cols
    assert "current_weakening_label" not in cols
    assert "future_rsrp" not in cols
    assert "future_cell" not in cols


def test_causal_rolling_features():
    pipeline = FeaturePipeline(window_size=3)
    data = {
        "trajectory_id": ["t1", "t1", "t1", "t1"],
        "step_index": [0, 1, 2, 3],
        "current_cell": ["Cell_A", "Cell_A", "Cell_A", "Cell_A"],
        "previous_cell_1": ["NONE", "Cell_A", "Cell_A", "Cell_A"],
        "current_rsrp_dbm": [-70.0, -80.0, -90.0, -100.0],
        "current_sinr_db": [20.0, 15.0, 10.0, 5.0],
        "speed_mps": [25.0, 25.0, 25.0, 25.0],
        "heading_deg": [0.0, 0.0, 0.0, 0.0],
        "heading_change_deg": [0.0, 0.0, 0.0, 0.0],
        "current_dwell_s": [0.0, 1.0, 2.0, 3.0],
    }
    df = pd.DataFrame(data)
    feat_df = pipeline.extract_batch_features(df)
    
    # Step 0 should have delta 0.0 (no backward look)
    assert feat_df["delta_rsrp"].iloc[0] == 0.0
    # Step 1 delta: -80 - (-70) = -10.0
    assert feat_df["delta_rsrp"].iloc[1] == -10.0
    # Step 3 rolling slope over window 3: (-100 - -70) / 3 = -10.0
    assert feat_df["rolling_rsrp_slope"].iloc[3] == pytest.approx(-10.0)


def test_model_feature_alignment():
    model = NextCellPredictor.load("artifacts/models/next_cell_model.joblib")
    pipeline = FeaturePipeline(cell_list=model.cell_classes)
    
    assert set(model.feature_names) == set(pipeline.feature_columns)
