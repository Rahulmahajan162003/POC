import pytest
from simulator.cells import CellNetwork
from decision.divergence import RouteDivergenceDetector
from decision.uncertainty import UncertaintyResult
from decision.policy import MobilityPolicyEngine, MobilityState


@pytest.fixture
def policy_engine():
    net = CellNetwork.default_6_cell_network()
    detector = RouteDivergenceDetector(net)
    return MobilityPolicyEngine(detector, conf_high_th=0.80, conf_med_th=0.60, weakness_risk_th=0.55)


def test_policy_high_confidence_proactive(policy_engine):
    telem = {"current_cell": "Cell_A", "x": 800.0, "y": 0.0, "heading_deg": 0.0, "timestamp": 10.0}
    inf_res = {"weakness_risk": 0.85, "telemetry_quality": {"is_valid": True}}
    unc_res = UncertaintyResult(
        predicted_cell="Cell_B",
        confidence=0.92,
        confidence_percentage=92.0,
        entropy=0.3,
        normalized_entropy=0.15,
        uncertainty_level="LOW",
        margin=0.85,
        top_candidates=[],
    )
    decision = policy_engine.evaluate(telem, inf_res, unc_res, step_index=1)
    assert decision.action == "PROACTIVE_CANDIDATE"
    assert decision.candidate_cell == "Cell_B"
    assert decision.state == MobilityState.PROACTIVE_CANDIDATE


def test_policy_moderate_confidence_monitor(policy_engine):
    telem = {"current_cell": "Cell_A", "x": 400.0, "y": 0.0, "heading_deg": 0.0, "timestamp": 5.0}
    inf_res = {"weakness_risk": 0.30, "telemetry_quality": {"is_valid": True}}
    unc_res = UncertaintyResult(
        predicted_cell="Cell_B",
        confidence=0.68,
        confidence_percentage=68.0,
        entropy=1.2,
        normalized_entropy=0.45,
        uncertainty_level="MEDIUM",
        margin=0.30,
        top_candidates=[],
    )
    decision = policy_engine.evaluate(telem, inf_res, unc_res, step_index=1)
    assert decision.action == "MONITOR_RECHECK"


def test_policy_high_uncertainty_hold(policy_engine):
    telem = {"current_cell": "Cell_A", "x": 0.0, "y": 0.0, "heading_deg": 0.0, "timestamp": 1.0}
    inf_res = {"weakness_risk": 0.20, "telemetry_quality": {"is_valid": True}}
    unc_res = UncertaintyResult(
        predicted_cell="Cell_B",
        confidence=0.45,
        confidence_percentage=45.0,
        entropy=2.1,
        normalized_entropy=0.80,
        uncertainty_level="HIGH",
        margin=0.05,
        top_candidates=[],
    )
    decision = policy_engine.evaluate(telem, inf_res, unc_res, step_index=1)
    assert decision.action == "HOLD_RECHECK"
    assert decision.candidate_cell is None
