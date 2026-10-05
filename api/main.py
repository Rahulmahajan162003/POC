import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.cells import CellNetwork
from simulator.routes import RouteGenerator
from simulator.mobility import MobilitySimulator
from simulator.radio import RadioEnvironment
from features.feature_engineering import FeaturePipeline
from models.next_cell_model import NextCellPredictor
from models.weakness_model import WeaknessPredictor
from models.inference import MobilityInferenceEngine
from decision.uncertainty import UncertaintyEstimator
from decision.divergence import RouteDivergenceDetector
from decision.policy import MobilityPolicyEngine, MobilityState
from storage.logger import EventLogger, MobilityEvent
from evaluation.metrics import MobilityMetricsCalculator


# Global runtime state
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

NETWORK = CellNetwork(CONFIG.get("cells", {}))
NEXT_CELL_MODEL: Optional[NextCellPredictor] = None
WEAKNESS_MODEL: Optional[WeaknessPredictor] = None
INFERENCE_ENGINE: Optional[MobilityInferenceEngine] = None
UNCERTAINTY_ESTIMATOR = UncertaintyEstimator()
DIVERGENCE_DETECTOR = RouteDivergenceDetector(NETWORK)
POLICY_ENGINE = MobilityPolicyEngine(
    DIVERGENCE_DETECTOR,
    conf_high_th=CONFIG.get("policy", {}).get("confidence_high_threshold", 0.80),
    conf_med_th=CONFIG.get("policy", {}).get("confidence_medium_threshold", 0.60),
    weakness_risk_th=CONFIG.get("policy", {}).get("weakening_risk_threshold", 0.55),
)
LOGGER = EventLogger()

# Live simulation session state
SIM_STATE: Dict[str, Any] = {
    "scenario": "highway_exit",
    "mobility": None,
    "radio": None,
    "history": [],
    "step_index": 0,
    "time": 0.0,
    "is_active": False,
}


def load_models():
    global NEXT_CELL_MODEL, WEAKNESS_MODEL, INFERENCE_ENGINE
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")

    if os.path.exists(next_cell_path) and os.path.exists(weakness_path):
        NEXT_CELL_MODEL = NextCellPredictor.load(next_cell_path)
        WEAKNESS_MODEL = WeaknessPredictor.load(weakness_path)
        feat_pipe = FeaturePipeline(cell_list=NEXT_CELL_MODEL.cell_classes)
        INFERENCE_ENGINE = MobilityInferenceEngine(NEXT_CELL_MODEL, WEAKNESS_MODEL, feat_pipe)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="YellowSense AI-Native Mobility Intelligence API",
    description="Uncertainty-Aware 5G Mobility Intelligence Engine with Route Divergence Adaptation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class TelemetryPayload(BaseModel):
    current_cell: str
    current_rsrp_dbm: float
    current_sinr_db: float
    speed_mps: float
    heading_deg: float
    heading_change_deg: Optional[float] = 0.0
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    timestamp: Optional[float] = 0.0
    previous_cell_1: Optional[str] = "NONE"
    previous_cell_2: Optional[str] = "NONE"
    current_dwell_s: Optional[float] = 0.0
    neighbor_Cell_A_rsrp: Optional[float] = -120.0
    neighbor_Cell_B_rsrp: Optional[float] = -120.0
    neighbor_Cell_C_rsrp: Optional[float] = -120.0
    neighbor_Cell_D_rsrp: Optional[float] = -120.0
    neighbor_Cell_E_rsrp: Optional[float] = -120.0
    neighbor_Cell_F_rsrp: Optional[float] = -120.0
    neighbor_Cell_A_sinr: Optional[float] = -10.0
    neighbor_Cell_B_sinr: Optional[float] = -10.0
    neighbor_Cell_C_sinr: Optional[float] = -10.0
    neighbor_Cell_D_sinr: Optional[float] = -10.0
    neighbor_Cell_E_sinr: Optional[float] = -10.0
    neighbor_Cell_F_sinr: Optional[float] = -10.0


class PredictResponse(BaseModel):
    predicted_next_cell: str
    probabilities: Dict[str, float]
    confidence: float
    confidence_percentage: float
    entropy: float
    normalized_entropy: float
    uncertainty_level: str
    weakness_risk: float
    recommended_action: str
    state: str
    reason: str
    divergence_detected: bool
    latency_ms: float


class ResetScenarioRequest(BaseModel):
    scenario: str = "highway_exit"
    seed: int = 42


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "yellowsense-mobility-intelligence",
        "model_loaded": INFERENCE_ENGINE is not None,
        "active_scenario": SIM_STATE["scenario"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict_mobility(payload: TelemetryPayload):
    if INFERENCE_ENGINE is None:
        load_models()
        if INFERENCE_ENGINE is None:
            raise HTTPException(status_code=503, detail="Models are not trained yet. Run train.py first.")

    telem_dict = payload.model_dump()
    inf_res = INFERENCE_ENGINE.run_inference(telem_dict, recent_history=[])
    unc_res = UNCERTAINTY_ESTIMATOR.evaluate(inf_res["probabilities"])
    decision = POLICY_ENGINE.evaluate(telem_dict, inf_res, unc_res, step_index=0)

    return PredictResponse(
        predicted_next_cell=unc_res.predicted_cell,
        probabilities=inf_res["probabilities"],
        confidence=unc_res.confidence,
        confidence_percentage=unc_res.confidence_percentage,
        entropy=unc_res.entropy,
        normalized_entropy=unc_res.normalized_entropy,
        uncertainty_level=unc_res.uncertainty_level,
        weakness_risk=inf_res["weakness_risk"],
        recommended_action=decision.action,
        state=decision.state.value,
        reason=decision.reason,
        divergence_detected=decision.divergence_detected,
        latency_ms=inf_res["latency_ms"],
    )


@app.get("/scenario")
def get_scenario_info():
    return {
        "available_scenarios": RouteGenerator.get_available_scenarios(),
        "active_scenario": SIM_STATE["scenario"],
        "step_index": SIM_STATE["step_index"],
        "sim_time": SIM_STATE["time"],
        "is_active": SIM_STATE["is_active"],
    }


@app.post("/scenario/reset")
def reset_scenario(req: ResetScenarioRequest):
    if INFERENCE_ENGINE is None:
        load_models()

    scenario = req.scenario
    if scenario not in RouteGenerator.get_available_scenarios():
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")

    waypoints = RouteGenerator.generate_waypoints(scenario, seed=req.seed)
    mobility = MobilitySimulator(waypoints, dt=1.0)
    radio = RadioEnvironment(NETWORK, seed=req.seed)
    POLICY_ENGINE.reset()

    SIM_STATE["scenario"] = scenario
    SIM_STATE["mobility"] = mobility
    SIM_STATE["radio"] = radio
    SIM_STATE["history"] = []
    SIM_STATE["step_index"] = 0
    SIM_STATE["time"] = 0.0
    SIM_STATE["is_active"] = True

    return {"status": "reset_successful", "scenario": scenario, "initial_x": mobility.x, "initial_y": mobility.y}


@app.post("/scenario/step")
def step_scenario():
    if not SIM_STATE["is_active"] or SIM_STATE["mobility"] is None:
        reset_scenario(ResetScenarioRequest(scenario="highway_exit"))

    if INFERENCE_ENGINE is None:
        load_models()

    mobility = SIM_STATE["mobility"]
    radio = SIM_STATE["radio"]
    scenario = SIM_STATE["scenario"]

    kin = mobility.step()
    rad = radio.compute_telemetry(kin["x"], kin["y"], dt=1.0)

    telemetry = {
        "trajectory_id": f"sim_{scenario}",
        "ue_id": "UE_API_USER",
        "scenario": scenario,
        "step_index": SIM_STATE["step_index"],
        "timestamp": round(SIM_STATE["time"], 1),
        "x": kin["x"],
        "y": kin["y"],
        "speed_mps": kin["speed_mps"],
        "heading_deg": kin["heading_deg"],
        "heading_change_deg": kin["heading_change_deg"],
        **rad,
    }

    inf_res = INFERENCE_ENGINE.run_inference(telemetry, recent_history=SIM_STATE["history"])
    unc_res = UNCERTAINTY_ESTIMATOR.evaluate(inf_res["probabilities"])
    decision = POLICY_ENGINE.evaluate(telemetry, inf_res, unc_res, step_index=SIM_STATE["step_index"])

    # Log to SQLite
    event = MobilityEvent(
        prediction_id=f"api_{scenario}_{SIM_STATE['step_index']:04d}",
        ue_id="UE_API_USER",
        trajectory_id=f"sim_{scenario}",
        scenario=scenario,
        step_index=SIM_STATE["step_index"],
        timestamp=round(SIM_STATE["time"], 1),
        x=kin["x"],
        y=kin["y"],
        speed_mps=kin["speed_mps"],
        heading_deg=kin["heading_deg"],
        current_cell=rad["current_cell"],
        predicted_cell=unc_res.predicted_cell,
        probabilities=inf_res["probabilities"],
        confidence=unc_res.confidence,
        uncertainty_level=unc_res.uncertainty_level,
        normalized_entropy=unc_res.normalized_entropy,
        weakening_risk=inf_res["weakness_risk"],
        action=decision.action,
        state=decision.state.value,
        reason=decision.reason,
        divergence_detected=decision.divergence_detected,
        observed_cell=rad["current_cell"],
        outcome="ADAPTED" if decision.divergence_detected else "CORRECT",
        latency_ms=inf_res["latency_ms"],
    )
    LOGGER.log_event(event)

    SIM_STATE["history"].append(telemetry)
    if len(SIM_STATE["history"]) > 10:
        SIM_STATE["history"].pop(0)

    SIM_STATE["step_index"] += 1
    SIM_STATE["time"] += 1.0

    return {
        "step_index": SIM_STATE["step_index"] - 1,
        "telemetry": telemetry,
        "prediction": {
            "predicted_next_cell": unc_res.predicted_cell,
            "probabilities": inf_res["probabilities"],
            "confidence": unc_res.confidence,
            "confidence_percentage": unc_res.confidence_percentage,
            "entropy": unc_res.entropy,
            "normalized_entropy": unc_res.normalized_entropy,
            "uncertainty_level": unc_res.uncertainty_level,
            "weakness_risk": inf_res["weakness_risk"],
        },
        "decision": {
            "action": decision.action,
            "candidate_cell": decision.candidate_cell,
            "state": decision.state.value,
            "reason": decision.reason,
            "divergence_detected": decision.divergence_detected,
        },
        "is_finished": kin["is_finished"],
    }


@app.get("/events")
def get_recent_events(limit: int = 50):
    df = LOGGER.get_events(limit=limit)
    return {"events": df.to_dict(orient="records")}


@app.get("/metrics")
def get_metrics_summary():
    report_path = os.path.join("artifacts", "reports", "evaluation_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"report_markdown": content}
    return {"status": "Run evaluate.py to generate full report."}
