import os
import sys
import json
import yaml
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.feature_engineering import FeaturePipeline
from models.next_cell_model import NextCellPredictor
from models.weakness_model import WeaknessPredictor


def main():
    print("=" * 60)
    print("YELLOWSENSE ML TRAINING & CALIBRATION PIPELINE")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = os.path.join("data", "synthetic", "synthetic_mobility_dataset.parquet")
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Running generator first...")
        from scripts.generate_data import main as gen_main
        gen_main()

    print("\n[1/5] Loading synthetic dataset & extracting features...")
    df = pd.read_parquet(data_path)
    
    # Ensure targets have no NaNs
    df["target_next_cell"] = df["target_next_cell"].ffill().bfill()
    df["current_weakening_label"] = df["current_weakening_label"].fillna(0).astype(int)

    cell_classes = sorted(list(config.get("cells", {}).keys()))

    feature_pipeline = FeaturePipeline(cell_list=cell_classes)
    feat_df = feature_pipeline.extract_batch_features(df)
    
    # Drop any remaining row with NaN in target or features
    feat_df = feat_df.dropna(subset=["target_next_cell", "current_weakening_label"] + feature_pipeline.feature_columns).reset_index(drop=True)

    # Save processed features
    processed_path = os.path.join("data", "processed", "processed_features.parquet")
    os.makedirs("data/processed", exist_ok=True)
    feat_df.to_parquet(processed_path, index=False)
    print(f"      Processed {len(feat_df):,} rows with {len(feature_pipeline.feature_columns)} features.")

    # 2. Trajectory-Level Partitioning
    print("\n[2/5] Partitioning data at Trajectory Level (70% Train / 15% Val / 15% Test)...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=config.get("seed", 42))
    train_idx, temp_idx = next(gss1.split(feat_df, groups=feat_df["trajectory_id"]))
    
    train_df = feat_df.iloc[train_idx].reset_index(drop=True)
    temp_df = feat_df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=config.get("seed", 42))
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["trajectory_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"      Train Set: {len(train_df):,} rows ({train_df['trajectory_id'].nunique()} trajectories)")
    print(f"      Val Set:   {len(val_df):,} rows ({val_df['trajectory_id'].nunique()} trajectories)")
    print(f"      Test Set:  {len(test_df):,} rows ({test_df['trajectory_id'].nunique()} trajectories)")

    # Save test set for evaluation script
    test_path = os.path.join("data", "processed", "test_split.parquet")
    test_df.to_parquet(test_path, index=False)

    feature_cols = feature_pipeline.feature_columns
    X_train, y_train_next = train_df[feature_cols], train_df["target_next_cell"]
    X_val, y_val_next = val_df[feature_cols], val_df["target_next_cell"]
    y_train_weak, y_val_weak = train_df["current_weakening_label"], val_df["current_weakening_label"]

    # 3. Train Next-Cell Model
    print("\n[3/5] Training & Calibrating Next-Cell Multi-Class XGBoost Model...")
    next_cell_predictor = NextCellPredictor(
        cell_classes=cell_classes,
        xgb_params=config.get("model", {}).get("next_cell_params", {}),
        calibration_method=config.get("model", {}).get("calibration_method", "sigmoid"),
    )
    next_cell_predictor.fit(X_train, y_train_next, X_val, y_val_next)
    print("      -> Next-Cell model calibrated successfully.")

    # 4. Train Weakness Model
    print("\n[4/5] Training & Calibrating Current-Cell Weakness Binary XGBoost Model...")
    weakness_predictor = WeaknessPredictor(
        xgb_params=config.get("model", {}).get("weakness_params", {}),
        calibration_method=config.get("model", {}).get("calibration_method", "sigmoid"),
    )
    weakness_predictor.fit(X_train, y_train_weak, X_val, y_val_weak)
    print("      -> Serving-cell weakness model calibrated successfully.")

    # 5. Save Artifacts & Metadata
    print("\n[5/5] Saving model artifacts and metadata...")
    os.makedirs("artifacts/models", exist_ok=True)
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")

    next_cell_predictor.save(next_cell_path)
    weakness_predictor.save(weakness_path)

    metadata = {
        "model_version": config.get("model", {}).get("version", "v1.0.0-xgb-calibrated"),
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "cell_classes": cell_classes,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "prediction_horizon_steps": config.get("simulator", {}).get("prediction_horizon_steps", 3),
        "calibration_method": config.get("model", {}).get("calibration_method", "sigmoid"),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }
    meta_path = os.path.join("artifacts", "models", "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"      Saved: {next_cell_path}")
    print(f"      Saved: {weakness_path}")
    print(f"      Saved: {meta_path}")
    print("\nTraining & Calibration completed successfully.")


if __name__ == "__main__":
    main()
