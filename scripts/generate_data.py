import os
import sys
import yaml
import pandas as pd

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.generator import SyntheticDatasetGenerator
from features.validation import TelemetryValidator


def main():
    print("=" * 60)
    print("YELLOWSENSE MOBILITY SIMULATOR — DATA GENERATION")
    print("=" * 60)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("\n[1/3] Initializing 6-cell network & route generators...")
    generator = SyntheticDatasetGenerator(config)

    print("[2/3] Generating synthetic multi-scenario trajectories...")
    df = generator.generate_full_dataset()
    print(f"      Generated {len(df):,} telemetry records across {df['trajectory_id'].nunique()} trajectories.")

    print("[3/3] Validating data quality...")
    validator = TelemetryValidator()
    validated_df = validator.validate_dataframe(df)

    out_raw = os.path.join("data", "synthetic", "synthetic_mobility_dataset.parquet")
    validated_df.to_parquet(out_raw, index=False)
    print(f"\n-> Saved raw synthetic dataset: {out_raw}")

    # Display scenario summary
    summary = df.groupby("scenario").agg(
        trajectories=("trajectory_id", "nunique"),
        records=("step_index", "count"),
        mean_speed=("speed_mps", "mean"),
    ).reset_index()
    print("\nDataset Summary by Scenario:")
    print(summary.to_string(index=False))
    print("\nData generation completed successfully.")


if __name__ == "__main__":
    main()
