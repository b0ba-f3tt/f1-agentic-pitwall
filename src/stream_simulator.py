import time
from src.telemetry_engine import PitWallTelemetryEngine


def replay_race_stream(year: int, gp: str, session_type: str, driver: str, rival: str, delay_sec: float = 0.05, dataset_id: str | None = None) -> list[dict]:
    engine = PitWallTelemetryEngine(year, gp, session_type, dataset_id=dataset_id)
    max_lap, triggered_alerts = int(engine.laps["LapNumber"].max()), []
    seen_anomaly_laps, seen_cliff_stints = set(), set()

    for current_lap in range(5, max_lap + 1):
        anom, cliff = engine.detect_lap_anomalies(driver, max_lap=current_lap), engine.analyze_tire_cliff(driver, max_lap=current_lap)
        new_anomalies = [a for a in anom["anomalies"] if a["lap"] not in seen_anomaly_laps]
        new_cliffs = [s for s in cliff["stints"] if s["cliff_detected"] and s["stint"] not in seen_cliff_stints]

        if new_anomalies or new_cliffs:
            seen_anomaly_laps.update(a["lap"] for a in new_anomalies)
            seen_cliff_stints.update(s["stint"] for s in new_cliffs)
            pit_state = engine.evaluate_pit_window_shifts(driver, rival, max_lap=current_lap)
            triggered_alerts.append({"triggered_at_lap": current_lap, "new_anomalies": new_anomalies,
                                     "new_tire_cliffs": new_cliffs, "latest_pit_events": pit_state["pit_window_events"][-2:]})
            print(f"[LAP {current_lap:02d}] ALERT TRIGGERED -> Anomalies: {len(new_anomalies)} | Cliffs: {len(new_cliffs)}")

        if delay_sec > 0:
            time.sleep(delay_sec)

    return triggered_alerts