import os
import json
import sqlite3
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class MobilityEvent:
    prediction_id: str
    ue_id: str
    trajectory_id: str
    scenario: str
    step_index: int
    timestamp: float
    x: float
    y: float
    speed_mps: float
    heading_deg: float
    current_cell: str
    predicted_cell: str
    probabilities: Dict[str, float]
    confidence: float
    uncertainty_level: str
    normalized_entropy: float
    weakening_risk: float
    action: str
    state: str
    reason: str
    divergence_detected: bool
    observed_cell: Optional[str] = None
    outcome: Optional[str] = None  # "CORRECT", "INCORRECT", "ADAPTED", "PENDING"
    latency_ms: float = 0.0
    model_version: str = "v1.0.0"


class EventLogger:
    """Audit logger persisting all mobility decisions to SQLite and Parquet for replay & analysis."""

    def __init__(self, db_path: str = "data/mobility_events.db", parquet_dir: str = "data/processed"):
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mobility_events (
                    prediction_id TEXT PRIMARY KEY,
                    ue_id TEXT,
                    trajectory_id TEXT,
                    scenario TEXT,
                    step_index INTEGER,
                    timestamp REAL,
                    x REAL,
                    y REAL,
                    speed_mps REAL,
                    heading_deg REAL,
                    current_cell TEXT,
                    predicted_cell TEXT,
                    probabilities_json TEXT,
                    confidence REAL,
                    uncertainty_level TEXT,
                    normalized_entropy REAL,
                    weakening_risk REAL,
                    action TEXT,
                    state TEXT,
                    reason TEXT,
                    divergence_detected INTEGER,
                    observed_cell TEXT,
                    outcome TEXT,
                    latency_ms REAL,
                    model_version TEXT,
                    logged_at TEXT
                )
            """)
            conn.commit()

    def log_event(self, event: MobilityEvent):
        logged_at = datetime.datetime.utcnow().isoformat()
        probs_json = json.dumps(event.probabilities)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO mobility_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                event.prediction_id,
                event.ue_id,
                event.trajectory_id,
                event.scenario,
                event.step_index,
                event.timestamp,
                event.x,
                event.y,
                event.speed_mps,
                event.heading_deg,
                event.current_cell,
                event.predicted_cell,
                probs_json,
                event.confidence,
                event.uncertainty_level,
                event.normalized_entropy,
                event.weakening_risk,
                event.action,
                event.state,
                event.reason,
                1 if event.divergence_detected else 0,
                event.observed_cell or "",
                event.outcome or "PENDING",
                event.latency_ms,
                event.model_version,
                logged_at,
            ))
            conn.commit()

    def get_events(self, scenario: Optional[str] = None, trajectory_id: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
        query = "SELECT * FROM mobility_events WHERE 1=1"
        params = []
        if scenario:
            query += " AND scenario = ?"
            params.append(scenario)
        if trajectory_id:
            query += " AND trajectory_id = ?"
            params.append(trajectory_id)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def export_to_parquet(self, filename: str = "events_dump.parquet") -> str:
        df = self.get_events(limit=100000)
        out_path = os.path.join(self.parquet_dir, filename)
        df.to_parquet(out_path, index=False)
        return out_path

    def clear_events(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM mobility_events")
            conn.commit()
