import os
import sys
import time
import yaml
import pandas as pd

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


def run_live_demo(scenario: str = "highway_exit", delay_s: float = 0.3):
    print("=" * 75)
    print("YELLOWSENSE AI-NATIVE MOBILITY INTELLIGENCE — LIVE CLI DEMO")
    print(f"SCENARIO: {scenario.upper()}")
    print("=" * 75)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Initialize Pipeline & Models
    next_cell_path = os.path.join("artifacts", "models", "next_cell_model.joblib")
    weakness_path = os.path.join("artifacts", "models", "weakness_model.joblib")

    if not os.path.exists(next_cell_path):
        print("[!] Models not trained. Running training pipeline...")
        from scripts.train import main as train_main
        train_main()

    next_cell_model = NextCellPredictor.load(next_cell_path)
    weakness_model = WeaknessPredictor.load(weakness_path)
    network = CellNetwork(config.get("cells", {}))
    
    feature_pipeline = FeaturePipeline(cell_list=next_cell_model.cell_classes)
    inference_engine = MobilityInferenceEngine(next_cell_model, weakness_model, feature_pipeline)
    uncertainty_estimator = UncertaintyEstimator()
    divergence_detector = RouteDivergenceDetector(network)
    policy_engine = MobilityPolicyEngine(
        divergence_detector,
        conf_high_th=config.get("policy", {}).get("confidence_high_threshold", 0.80),
        conf_med_th=config.get("policy", {}).get("confidence_medium_threshold", 0.60),
        weakness_risk_th=config.get("policy", {}).get("weakening_risk_threshold", 0.55),
    )
    logger = EventLogger()

    # 2. Setup Kinematics & Radio
    waypoints = RouteGenerator.generate_waypoints(scenario, seed=42, traj_index=0)
    mobility = MobilitySimulator(waypoints, dt=1.0)
    radio = RadioEnvironment(network, seed=42)

    history = []
    step = 0
    t = 0.0

    print(f"\n{'STEP':<5} | {'POS (X, Y)':<16} | {'SERV':<7} | {'RSRP':<7} | {'PRED':<7} | {'CONF':<7} | {'UNCERT':<8} | {'WEAK%':<7} | {'ACTION / STATE':<22}")
    print("-" * 105)

    while True:
        kin = mobility.step()
        rad = radio.compute_telemetry(kin["x"], kin["y"], dt=1.0)
        
        telemetry = {
            "trajectory_id": f"demo_{scenario}",
            "ue_id": "UE_DEMO_01",
            "scenario": scenario,
            "step_index": step,
            "timestamp": round(t, 1),
            "x": kin["x"],
            "y": kin["y"],
            "speed_mps": kin["speed_mps"],
            "heading_deg": kin["heading_deg"],
            "heading_change_deg": kin["heading_change_deg"],
            **rad,
        }

        # Run AI Inference
        inf_res = inference_engine.run_inference(telemetry, recent_history=history)
        unc_res = uncertainty_estimator.evaluate(inf_res["probabilities"])

        # Run Policy Decision
        decision = policy_engine.evaluate(telemetry, inf_res, unc_res, step_index=step)

        # Print Live Row
        pos_str = f"({kin['x']:.0f}, {kin['y']:.0f})"
        conf_str = f"{unc_res.confidence_percentage:.1f}%"
        weak_str = f"{inf_res['weakness_risk'] * 100:.1f}%"
        
        status_display = decision.action
        if decision.divergence_detected:
            status_display = ">>> DIVERGENCE! <<<"

        print(f"{step:<5} | {pos_str:<16} | {rad['current_cell']:<7} | {rad['current_rsrp_dbm']:<7.1f} | {unc_res.predicted_cell:<7} | {conf_str:<7} | {unc_res.uncertainty_level:<8} | {weak_str:<7} | {status_display:<22}")

        if decision.divergence_detected:
            print(f"      [!] {decision.reason}")

        # Log event
        event = MobilityEvent(
            prediction_id=f"evt_{scenario}_{step:04d}",
            ue_id="UE_DEMO_01",
            trajectory_id=f"demo_{scenario}",
            scenario=scenario,
            step_index=step,
            timestamp=round(t, 1),
            x=kin["x"],
            y=kin["y"],
            speed_mps=kin["speed_mps"],
            heading_deg=kin["heading_deg"],
            current_cell=rad["current_cell"],
            predicted_cell=unc_res.predicted_cell,
            probabilities=unc_res.probabilities if hasattr(unc_res, "probabilities") else inf_res["probabilities"],
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
        logger.log_event(event)

        history.append(telemetry)
        if len(history) > 10:
            history.pop(0)

        step += 1
        t += 1.0

        if kin["is_finished"]:
            break
        time.sleep(delay_s)

    print("-" * 105)
    print("\nDemonstration trajectory completed. All events logged to SQLite & Parquet.")


if __name__ == "__main__":
    run_live_demo()
