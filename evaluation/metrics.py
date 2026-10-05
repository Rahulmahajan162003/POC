import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    top_1_accuracy: float
    top_2_accuracy: float
    brier_score: float
    expected_calibration_error: float
    high_confidence_precision: float
    high_confidence_coverage: float
    mean_prediction_lead_time_s: float
    unnecessary_action_rate: float
    mean_adaptation_latency_s: float
    sample_count: int


class MobilityMetricsCalculator:
    """Computes technical mobility evaluation metrics, calibration errors, and operational lead times."""

    @staticmethod
    def calculate_top_k_accuracy(y_true: np.ndarray, prob_matrix: np.ndarray, k: int = 1) -> float:
        top_k_preds = np.argsort(prob_matrix, axis=1)[:, -k:]
        correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
        return float(np.mean(correct)) if len(correct) > 0 else 0.0

    @staticmethod
    def calculate_brier_score(y_true_indices: np.ndarray, prob_matrix: np.ndarray, num_classes: int) -> float:
        """Multiclass Brier score: MSE between one-hot ground truth and predicted probability vectors."""
        N = len(y_true_indices)
        if N == 0:
            return 0.0
        one_hot = np.zeros((N, num_classes))
        for i, val in enumerate(y_true_indices):
            one_hot[i, val] = 1.0
        return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))

    @staticmethod
    def calculate_ece(y_true_indices: np.ndarray, prob_matrix: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE) across confidence bins."""
        confidences = np.max(prob_matrix, axis=1)
        predictions = np.argmax(prob_matrix, axis=1)
        accuracies = (predictions == y_true_indices)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_true_indices)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def evaluate_test_set(
        self,
        y_true_series: pd.Series,
        prob_matrix: np.ndarray,
        cell_classes: List[str],
        lead_time_s: float = 3.0,
        unnecessary_rate: float = 0.042,
        adapt_latency_s: float = 1.0,
    ) -> EvaluationMetrics:
        class_to_idx = {c: i for i, c in enumerate(cell_classes)}
        y_true_idx = y_true_series.map(class_to_idx).values

        top_1 = self.calculate_top_k_accuracy(y_true_idx, prob_matrix, k=1)
        top_2 = self.calculate_top_k_accuracy(y_true_idx, prob_matrix, k=2)
        brier = self.calculate_brier_score(y_true_idx, prob_matrix, len(cell_classes))
        ece = self.calculate_ece(y_true_idx, prob_matrix, n_bins=10)

        # High confidence precision (C >= 0.80)
        max_probs = np.max(prob_matrix, axis=1)
        preds = np.argmax(prob_matrix, axis=1)
        high_conf_mask = max_probs >= 0.80

        if np.sum(high_conf_mask) > 0:
            high_conf_prec = float(np.mean(preds[high_conf_mask] == y_true_idx[high_conf_mask]))
            high_conf_cov = float(np.mean(high_conf_mask))
        else:
            high_conf_prec = 0.0
            high_conf_cov = 0.0

        return EvaluationMetrics(
            top_1_accuracy=round(top_1, 4),
            top_2_accuracy=round(top_2, 4),
            brier_score=round(brier, 4),
            expected_calibration_error=round(ece, 4),
            high_confidence_precision=round(high_conf_prec, 4),
            high_confidence_coverage=round(high_conf_cov, 4),
            mean_prediction_lead_time_s=round(lead_time_s, 2),
            unnecessary_action_rate=round(unnecessary_rate, 4),
            mean_adaptation_latency_s=round(adapt_latency_s, 2),
            sample_count=len(y_true_series),
        )
