import pytest
from simulator.cells import CellNetwork
from decision.divergence import RouteDivergenceDetector


def test_divergence_normal_trajectory():
    net = CellNetwork.default_6_cell_network()
    detector = RouteDivergenceDetector(net)

    # UE at (500, 0), moving east (0 deg) directly towards Cell B (2000, 0)
    current_telem = {
        "x": 500.0,
        "y": 0.0,
        "heading_deg": 0.0,
        "current_cell": "Cell_A",
        "neighbor_Cell_B_rsrp": -85.0,
        "neighbor_Cell_C_rsrp": -110.0,
    }
    prior_telem = {
        "neighbor_Cell_B_rsrp": -88.0,
        "neighbor_Cell_C_rsrp": -112.0,
    }
    rep = detector.detect(current_telem, active_candidate_cell="Cell_B", prior_telemetry=prior_telem)
    assert not rep.divergence_detected
    assert rep.heading_discrepancy_deg < 5.0


def test_divergence_sudden_exit_ramp():
    net = CellNetwork.default_6_cell_network()
    detector = RouteDivergenceDetector(net)

    # UE at (1100, 300), suddenly turning north-east (55 deg) towards Cell C (1600, 1100), away from Cell B (2000, 0)
    current_telem = {
        "x": 1100.0,
        "y": 300.0,
        "heading_deg": 65.0,
        "current_cell": "Cell_A",
        "neighbor_Cell_B_rsrp": -92.0,  # Dropping
        "neighbor_Cell_C_rsrp": -78.0,  # Rising fast!
    }
    prior_telem = {
        "neighbor_Cell_B_rsrp": -86.0,
        "neighbor_Cell_C_rsrp": -88.0,
    }
    rep = detector.detect(current_telem, active_candidate_cell="Cell_B", prior_telemetry=prior_telem)
    assert rep.divergence_detected
    assert rep.invalidated_candidate == "Cell_B"
    assert rep.emerging_candidate == "Cell_C"
