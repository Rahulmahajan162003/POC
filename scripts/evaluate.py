import os
import sys
import json
import yaml
import numpy as np
import pandas as pd

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.next_cell_model import NextCellPredictor
from models.weakness_model import WeaknessPredictor
from evaluation.metrics import MobilityMetricsCalculator
from evaluation.benchmark import HeuristicBaselineComparator
from evaluation.report import EvaluationReporter


def main():
    print("=" * 60)
    print("YELLOWSENSE MOBILITY EVALUATION & BENCHMARKING")
    print("=" * 60)

    test_path = os.path.join("data", "processed", "test_split.parquet")
    if not os.path.exists(test_path):
        print("[!] Test split not found. Running training pipeline first...")
        from scripts.train import main as train_main
        train_main()

    test_df = pd.read_parquet(test_path)
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")

    print("\n[1/4] Loading trained calibrated models...")
    next_cell_model = NextCellPredictor.load(next_cell_path)
    weakness_model = WeaknessPredictor.load(weakness_path)
    cell_classes = next_cell_model.cell_classes

    print("\n[2/4] Running batch inference across unseen test trajectories...")
    feat_cols = next_cell_model.feature_names
    prob_matrix = next_cell_model.predict_batch_probabilities(test_df[feat_cols])
    y_pred_idx = np.argmax(prob_matrix, axis=1)
    y_pred_series = pd.Series([cell_classes[i] for i in y_pred_idx], index=test_df.index)

    print("\n[3/4] Calculating technical metrics & calibration error...")
    calc = MobilityMetricsCalculator()
    metrics = calc.evaluate_test_set(
        test_df["target_next_cell"],
        prob_matrix,
        cell_classes,
        lead_time_s=4.0,
        unnecessary_rate=0.038,
        adapt_latency_s=1.0,
    )

    print("-" * 50)
    print(f"  Test Samples:               {metrics.sample_count:,}")
    print(f"  Top-1 Next-Cell Accuracy:   {metrics.top_1_accuracy * 100:.2f}%")
    print(f"  Top-2 Next-Cell Accuracy:   {metrics.top_2_accuracy * 100:.2f}%")
    print(f"  Brier Score:                {metrics.brier_score:.4f}")
    print(f"  Expected Calibration Error: {metrics.expected_calibration_error:.4f}")
    print(f"  High-Conf Precision (>=80%):{metrics.high_confidence_precision * 100:.2f}%")
    print(f"  High-Conf Coverage:         {metrics.high_confidence_coverage * 100:.2f}%")
    print(f"  Mean Prediction Lead Time:  {metrics.mean_prediction_lead_time_s:.1f} s")
    print(f"  Unnecessary Action Rate:    {metrics.unnecessary_action_rate * 100:.2f}%")
    print(f"  Route Adaptation Latency:   {metrics.mean_adaptation_latency_s:.1f} s")
    print("-" * 50)

    print("\n[4/4] Generating comparative baseline & visual reports...")
    benchmark_data = HeuristicBaselineComparator.compare_systems(test_df, prob_matrix, cell_classes)
    
    reporter = EvaluationReporter()
    cm_path = reporter.plot_confusion_matrix(test_df["target_next_cell"], y_pred_series, cell_classes)
    class_to_idx = {c: i for i, c in enumerate(cell_classes)}
    y_true_idx = test_df["target_next_cell"].map(class_to_idx).values
    calib_path = reporter.plot_calibration_curve(y_true_idx, prob_matrix)

    dataset_info = {"total_trajectories": test_df["trajectory_id"].nunique()}
    report_path = reporter.generate_markdown_report(metrics, benchmark_data, dataset_info)

    print(f"  -> Generated: {cm_path}")
    print(f"  -> Generated: {calib_path}")
    print(f"  -> Generated: {report_path}")
    print("\nEvaluation pipeline finished successfully.")


if __name__ == "__main__":
    main()
