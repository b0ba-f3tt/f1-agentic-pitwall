import os
import pytest
from src.telemetry_engine import PitWallTelemetryEngine


@pytest.mark.skipif(
    not os.path.exists("data/real_sessions/2024_Silverstone_R_laps.parquet"),
    reason="Real Parquet datasets not generated yet",
)
def test_real_silverstone_2024_telemetry():
    engine = PitWallTelemetryEngine(dataset_id="2024_Silverstone_R")
    drivers = engine.get_driver_list()
    assert "HAM" in drivers and "VER" in drivers

    weather = engine.get_weather_summary()
    assert weather["rainfall_detected"] is True

    cliff = engine.analyze_tire_cliff("LEC")
    assert len(cliff["stints"]) >= 1

    pit = engine.evaluate_pit_window_shifts("HAM", "VER")
    assert len(pit["pit_window_events"]) >= 1