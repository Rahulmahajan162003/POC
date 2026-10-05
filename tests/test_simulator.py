import pytest
import yaml
import pandas as pd
from simulator.cells import CellNetwork
from simulator.routes import RouteGenerator
from simulator.mobility import MobilitySimulator
from simulator.radio import RadioEnvironment
from simulator.generator import SyntheticDatasetGenerator


@pytest.fixture
def sample_config():
    return {
        "seed": 42,
        "simulator": {
            "dt_seconds": 1.0,
            "prediction_horizon_steps": 3,
            "noise_std_db": 1.5,
            "temporal_correlation_alpha": 0.75,
            "hysteresis_db": 3.0,
            "thermal_noise_dbm": -104.0,
            "path_loss_exponent": 3.4,
            "reference_loss_d0": 38.5,
        },
        "cells": {
            "Cell_A": {"x": 0.0, "y": 0.0},
            "Cell_B": {"x": 2000.0, "y": 0.0},
            "Cell_C": {"x": 1600.0, "y": 1100.0},
            "Cell_D": {"x": 0.0, "y": 1800.0},
            "Cell_E": {"x": 3600.0, "y": 0.0},
            "Cell_F": {"x": 2800.0, "y": 1600.0},
        },
        "scenarios": {
            "highway_straight": {"num_trajectories": 2},
            "highway_exit": {"num_trajectories": 2},
        },
        "features": {
            "weakening_rsrp_threshold_dbm": -102.0,
            "weakening_delta_drop_db": 6.0,
        },
    }


def test_cell_network_distance():
    net = CellNetwork.default_6_cell_network()
    assert len(net.get_cell_ids()) == 6
    dists = net.get_distances(0.0, 0.0)
    assert dists["Cell_A"] == 5.0  # Min clamped distance
    assert dists["Cell_B"] == pytest.approx(2000.0, abs=1.0)


def test_mobility_kinematics():
    wps = [(0.0, 0.0, 20.0), (100.0, 0.0, 20.0)]
    sim = MobilitySimulator(wps, dt=1.0)
    step1 = sim.step()
    assert step1["speed_mps"] > 0
    assert step1["heading_deg"] == pytest.approx(0.0, abs=1.0)


def test_synthetic_generator(sample_config):
    generator = SyntheticDatasetGenerator(sample_config)
    df = generator.generate_single_trajectory(
        scenario="highway_exit",
        trajectory_id="test_001",
        ue_id="UE_001",
        seed=42,
    )
    assert not df.empty
    assert "target_next_cell" in df.columns
    assert "current_weakening_label" in df.columns
    assert "neighbor_Cell_B_rsrp" in df.columns
    assert len(df) > 10
