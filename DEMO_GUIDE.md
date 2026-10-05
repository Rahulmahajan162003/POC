# YellowSense — Live Demonstration Guide (2–3 Minutes)
**COMET 5G Spring Board 6 | Problem Statement #6: AI-Native Network Intelligence**

---

## 🎯 Demonstration Objective
Showcase how **YellowSense** predicts future cell demand, uses **entropy uncertainty** to prevent overconfident mistakes, detects real-time **route divergence**, and dynamically adapts without hardcoded heuristics.

---

## ⏱️ Step-by-Step 3-Minute Presentation Flow

### Phase 1: Launch & Orientation (0:00 – 0:45)
1. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
2. **Open Browser** at `http://localhost:8501`.
3. **Set the Stage**:
   > *"Judges, today 5G handover is purely reactive: a phone only searches for a new cell after its connection has already degraded. YellowSense is an AI-native mobility intelligence layer that predicts future cell demand 3 seconds in advance, quantifies uncertainty, and dynamically adapts when user behavior changes."*
4. **Point to the Layout**:
   - **Topology Map**: 6 Cells (`Cell_A` through `Cell_F`) with coverage contours.
   - **Top KPI Cards**: Serving Cell, Predicted Cell, Calibrated Confidence, Entropy Uncertainty, Weakening Risk.
   - **Policy Engine Banner**: Action and live human-readable rationale.

---

### Phase 2: The Normal Corridor & Proactive Preparation (0:45 – 1:30)
1. Ensure the scenario dropdown is set to: **⭐ Highway Route Deviation (Primary Demo)**.
2. Click **`⏭ STEP`** or **`▶ START`**.
3. **Highlight Key States**:
   - At Step 10–25: UE is in `Cell_A`, traveling east at ~28 m/s towards `Cell_B`.
   - **Model Prediction**: `Cell_B` with **~92% Calibrated Confidence**.
   - **Weakening Risk**: **~84%** (imminent serving cell attenuation).
   - **Entropy Uncertainty**: **LOW** ($H_{norm} < 0.20$).
   - **Decision Policy**: **`PROACTIVE_CANDIDATE`** allocated for `Cell_B`.
   - **YellowSense Advantage**: Target resources and beamforming are pre-warmed 3.0 seconds before physical threshold crossing.

---

### Phase 3: The Unexpected Turn & Route Divergence (1:30 – 2:15)
1. Continue simulation past Step 40.
2. **The Event**:
   - The vehicle suddenly takes the exit ramp, steering north-east ($+55^\circ$) towards `Cell_C` instead of continuing along the highway to `Cell_B`.
3. **Point to the Red Alert Banner**:
   - **State**: **`ROUTE DIVERGENCE DETECTED`** $\to$ **`RE-EVALUATING`**.
   - **Explanation**: *"Trajectory heading deviated by 52° from Cell_B corridor; alternative Cell_C signal is emerging (+6.8 dBm/s)."*
   - **Safety Action**: `Cell_B` proactive candidate is immediately **invalidated** to prevent wrong target handover.
   - **Re-evaluation**: AI updates distribution; `Cell_C` emerges with high confidence.
   - **State**: Transitions to **`UPDATED_CANDIDATE`** $\to$ `Cell_C`.

---

### Phase 4: Observation & Ground Truth Confirmation (2:15 – 2:45)
1. Allow simulation to reach Step 75–100.
2. **Observe Outcome**:
   - UE enters `Cell_C` coverage at $(1600, 1100)$.
   - Handover completes seamlessly.
   - Event is marked as **`ADAPTED`** in the bottom replayable audit log.

---

### Phase 5: Technical Validation & Heuristic Benchmark (2:45 – 3:00)
1. Click the **📊 EVALUATION & BENCHMARKS** tab in the dashboard.
2. **Highlight Technical Proof**:
   - **Top-1 Accuracy**: **97.11%** (vs 78.9% for reactive A3 heuristic).
   - **Expected Calibration Error (ECE)**: **0.0135** (exceeds < 0.05 standard).
   - **Unnecessary Handover Reduction**: **79.3% reduction in ping-pong handovers**.
   - Show the **Confusion Matrix** and **Reliability Diagram**.
3. **Concluding Statement**:
   > *"By combining calibrated prediction, entropy uncertainty, and divergence detection, YellowSense eliminates blind reactive handovers and delivers genuine AI-native network intelligence."*

---

## 🎮 Secondary Demonstrations to Try

1. **Uncertainty in Action — `urban_grid`**:
   - Select `urban_grid` from the dropdown.
   - At complex 4-way intersections, show how probabilities split across multiple cells, confidence drops to ~45%, entropy rises to **HIGH**, and policy safely chooses **`HOLD_RECHECK`** instead of guessing.
2. **Degraded Telemetry Gate — `sparse_telemetry` or `noisy_measurement`**:
   - Show how the data quality gate blocks aggressive proactive switching when packet jitter is excessive.
