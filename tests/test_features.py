import pytest
import numpy as np
import pandas as pd
from features.feature_engineering import FeaturePipeline
from features.validation import TelemetryValidator


def test_feature_pipeline_batch():
    pipeline = FeaturePipeline()
    sample_data = {
        "trajectory_id": ["traj_1", "traj_1", "traj_1"],
        "step_index": [0, 1, 2],
        "current_cell": ["Cell_A", "Cell_A", "Cell_A"],
        "previous_cell_1": ["NONE", "Cell_A", "Cell_A"],
        "current_rsrp_dbm": [-75.0, -78.0, -82.0],
        "current_sinr_db": [18.0, 15.0, 12.0],
        "speed_mps": [28.0, 28.5, 29.0],
        "heading_deg": [0.0, 5.0, 10.0],
        "heading_change_deg": [0.0, 5.0, 5.0],
        "current_dwell_s": [0.0, 1.0, 2.0],
        "neighbor_Cell_A_rsrp": [-75.0, -78.0, -82.0],
        "neighbor_Cell_B_rsrp": [-95.0, -90.0, -84.0],
        "neighbor_Cell_C_rsrp": [-110.0, -108.0, -105.0],
        "neighbor_Cell_D_rsrp": [-115.0, -115.0, -115.0],
        "neighbor_Cell_E_rsrp": [-120.0, -120.0, -120.0],
        "neighbor_Cell_F_rsrp": [-120.0, -120.0, -120.0],
        "neighbor_Cell_A_sinr": [18.0, 15.0, 12.0],
        "neighbor_Cell_B_sinr": [2.0, 5.0, 8.0],
        "neighbor_Cell_C_sinr": [-5.0, -3.0, -1.0],
        "neighbor_Cell_D_sinr": [-10.0, -10.0, -10.0],
        "neighbor_Cell_E_sinr": [-10.0, -10.0, -10.0],
        "neighbor_Cell_F_sinr": [-10.0, -10.0, -10.0],
    }
    df = pd.DataFrame(sample_data)
    feat_df = pipeline.extract_batch_features(df)
    
    for col in pipeline.feature_columns:
        assert col in feat_df.columns, f"Missing feature {col}"
    
    assert feat_df["delta_rsrp"].iloc[1] == pytest.approx(-3.0)


def test_telemetry_validator():
    validator = TelemetryValidator()
    
    # Healthy record
    good_rec = {
        "current_rsrp_dbm": -80.0,
        "current_sinr_db": 15.0,
        "speed_mps": 25.0,
        "timestamp": 10.0,
    }
    rep1 = validator.validate_record(good_rec, last_timestamp=9.0)
    assert rep1.is_valid
    assert rep1.status == "HEALTHY"

    # Out of order timestamp
    bad_ts_rec = {
        "current_rsrp_dbm": -80.0,
        "current_sinr_db": 15.0,
        "speed_mps": 25.0,
        "timestamp": 8.0,
    }
    rep2 = validator.validate_record(bad_ts_rec, last_timestamp=10.0)
    assert len(rep2.issues) > 0
