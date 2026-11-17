# F1 Pit-Wall Telemetry & Agentic MCP Strategy Triage

End-to-end Formula 1 race telemetry analytics and agentic strategy briefing pipeline built with `FastF1`, `SciPy`, `PyArrow`, and the `Model Context Protocol (MCP)`.

## Architecture
- **Deterministic Telemetry Engine (`src/telemetry_engine.py`)**: Fuel-load normalization (`0.055 s/lap`), modified Z-score lap anomaly detection, OLS regression tire-cliff modeling, weather telemetry analysis, and undercut/overcut net gap swing calculation.
- **Real Race Parquet Datasets (`data/real_sessions/`)**: Pre-ingested FIA/F1 race timing and weather snapshots (`2024_Silverstone_R`, `2024_Monza_R`, `2023_Singapore_R`) extracted via `FastF1`.
- **MCP Tool Server (`src/mcp_server.py`)**: Exposes telemetry models over `stdio` via `FastMCP`.
- **Agentic Triage Orchestrator (`src/strategy_agent.py`)**: Autonomous LLM tool-calling loop that synthesizes telemetry anomalies into pit-wall strategy briefings, with a deterministic offline fallback mode.
- **Lap-by-Lap Streaming Replay (`src/stream_simulator.py`)**: Sequential lap ingestion that triggers strategy alerts the exact lap a tire cliff or pace anomaly breaches threshold.

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/ingest_real_datasets.py
pytest -v
python main.py --dataset 2024_Silverstone_R --driver HAM --rival VER
python main.py --dataset 2024_Silverstone_R --driver LEC --rival VER --stream

```