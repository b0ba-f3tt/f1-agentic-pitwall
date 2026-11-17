import json
import os
from mcp.server.mcpserver import MCPServer as FastMCP
from src.telemetry_engine import PitWallTelemetryEngine

mcp = FastMCP("F1-PitWall-Telemetry")
_ENGINE: PitWallTelemetryEngine | None = None


def get_engine() -> PitWallTelemetryEngine:
    global _ENGINE
    if _ENGINE is None:
        year = int(os.getenv("F1_YEAR", "2024"))
        gp = os.getenv("F1_GP", "Silverstone")
        session = os.getenv("F1_SESSION", "R")
        dataset_id = os.getenv("F1_DATASET_ID") or None
        _ENGINE = PitWallTelemetryEngine(year, gp, session, dataset_id=dataset_id)
    return _ENGINE


@mcp.tool()
def list_active_drivers() -> str:
    eng = get_engine()
    return json.dumps({"dataset_id": eng.dataset_id or f"{eng.year}_{eng.gp}_{eng.session_type}", "drivers": eng.get_driver_list()})


@mcp.tool()
def get_track_weather_telemetry() -> str:
    return json.dumps(get_engine().get_weather_summary())


@mcp.tool()
def detect_anomalous_lap_drops(driver: str, z_threshold: float = 2.8) -> str:
    return json.dumps(get_engine().detect_lap_anomalies(driver.upper(), z_threshold=z_threshold))


@mcp.tool()
def detect_tire_wear_cliff(driver: str, deg_threshold_sec_per_lap: float = 0.085) -> str:
    return json.dumps(get_engine().analyze_tire_cliff(driver.upper(), deg_threshold_sec_per_lap=deg_threshold_sec_per_lap))


@mcp.tool()
def analyze_pit_window_shift(driver: str, rival_driver: str) -> str:
    return json.dumps(get_engine().evaluate_pit_window_shifts(driver.upper(), rival_driver.upper()))


if __name__ == "__main__":
    mcp.run(transport="stdio")