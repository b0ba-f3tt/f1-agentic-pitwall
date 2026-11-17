import pandas as pd
from src.telemetry_engine import PitWallTelemetryEngine


def _build_mock_engine() -> PitWallTelemetryEngine:
    engine = PitWallTelemetryEngine.__new__(PitWallTelemetryEngine)
    laps_data = [{"Driver": "LEC", "LapNumber": lap, "Stint": 1, "Compound": "MEDIUM", "TyreLife": lap,
                  "LapTimeSec": (base := 90.0 + (lap * 0.14) + (2.6 if lap == 10 else 0.0)),
                  "FuelCorrectedSec": base - ((15 - lap) * 0.055), "IsPitLap": False, "IsGreenFlag": True,
                  "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "Time": pd.to_timedelta(lap * 90.5, unit="s")} for lap in range(1, 16)]
    engine.laps = pd.DataFrame(laps_data)
    engine.weather_data = pd.DataFrame()
    return engine


def test_detect_lap_anomalies_flags_spike():
    """Lap 10 has +2.6s anomaly (after standing-start Lap 1 excluded)."""
    engine = _build_mock_engine()
    res = engine.detect_lap_anomalies("LEC", z_threshold=2.5)
    assert 10 in [a["lap"] for a in res["anomalies"]]


def test_analyze_tire_cliff_detects_high_degradation():
    engine = _build_mock_engine()
    res = engine.analyze_tire_cliff("LEC", deg_threshold_sec_per_lap=0.085)
    assert len(res["stints"]) == 1 and res["stints"][0]["cliff_detected"] is True