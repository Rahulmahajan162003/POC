import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "yellowsense-mobility-intelligence"


def test_api_scenario_endpoints():
    # Scenario info
    resp1 = client.get("/scenario")
    assert resp1.status_code == 200
    assert "highway_exit" in resp1.json()["available_scenarios"]

    # Scenario reset
    resp2 = client.post("/scenario/reset", json={"scenario": "highway_exit", "seed": 42})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "reset_successful"

    # Scenario step
    resp3 = client.post("/scenario/step")
    assert resp3.status_code == 200
    step_data = resp3.json()
    assert "prediction" in step_data
    assert "decision" in step_data
    assert "telemetry" in step_data


def test_api_predict():
    payload = {
        "current_cell": "Cell_A",
        "current_rsrp_dbm": -78.0,
        "current_sinr_db": 16.0,
        "speed_mps": 28.0,
        "heading_deg": 0.0,
        "neighbor_Cell_B_rsrp": -82.0,
        "neighbor_Cell_C_rsrp": -105.0,
        "neighbor_Cell_D_rsrp": -115.0,
        "neighbor_Cell_E_rsrp": -120.0,
        "neighbor_Cell_F_rsrp": -120.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_next_cell" in data
    assert "confidence" in data
    assert "uncertainty_level" in data
    assert "recommended_action" in data


def test_api_invalid_payload_error():
    # Missing required field `current_cell` and `current_rsrp_dbm`
    bad_payload = {"speed_mps": 30.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_api_unknown_scenario_error():
    response = client.post("/scenario/reset", json={"scenario": "invalid_scenario_name"})
    assert response.status_code == 400
