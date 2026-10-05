import pytest
import math
from decision.uncertainty import UncertaintyEstimator


def test_uncertainty_high_confidence():
    estimator = UncertaintyEstimator()
    probs = {
        "Cell_A": 0.02,
        "Cell_B": 0.92,
        "Cell_C": 0.04,
        "Cell_D": 0.01,
        "Cell_E": 0.005,
        "Cell_F": 0.005,
    }
    res = estimator.evaluate(probs)
    assert res.predicted_cell == "Cell_B"
    assert res.confidence == pytest.approx(0.92, abs=0.01)
    assert res.uncertainty_level == "LOW"
    assert res.normalized_entropy < 0.35


def test_uncertainty_distributed_equal():
    estimator = UncertaintyEstimator()
    # Uniform distribution across 6 cells -> max entropy
    probs = {f"Cell_{c}": 1.0 / 6.0 for c in ["A", "B", "C", "D", "E", "F"]}
    res = estimator.evaluate(probs)
    assert res.confidence == pytest.approx(1.0 / 6.0, abs=0.01)
    assert res.uncertainty_level == "HIGH"
    assert res.normalized_entropy == pytest.approx(1.0, abs=0.01)


def test_uncertainty_zero_probabilities_handled_safely():
    estimator = UncertaintyEstimator()
    probs = {"Cell_A": 1.0, "Cell_B": 0.0, "Cell_C": 0.0}
    res = estimator.evaluate(probs)
    assert not math.isnan(res.entropy)
    assert res.entropy == 0.0
    assert res.confidence == 1.0
