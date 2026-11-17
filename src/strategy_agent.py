import json
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import APIError, OpenAI, RateLimitError
from src.telemetry_engine import PitWallTelemetryEngine


SYSTEM_PROMPT = """You are a Senior Race Strategy Engineer on an F1 Pit Wall.
You have access to real-time telemetry analysis tools over MCP.
Execute a strict triage workflow:
1. Check track and rainfall weather telemetry.
2. Inspect lap-time anomalies for the target driver and rival.
3. Evaluate tire-wear degradation slopes and thermal/wear cliffs across stints.
4. Analyze pit-window shifts and undercut/overcut net time swings.
5. Produce a structured Pit-Wall Strategy Briefing with:
   - EXECUTIVE TRIAGE ALERT (Severity: CRITICAL / HIGH / NOMINAL)
   - ANOMALY & DEGRADATION TELEMETRY SUMMARY (exact lap numbers, compounds, slopes, Z-scores)
   - PIT-WINDOW & UNDERCUT EXPOSURE (net gap swings in seconds)
   - RECOMMENDED PIT-WALL ACTION (concrete lap call, target compound, contingency)."""


def _clean_tool_schema(raw_schema: dict | None) -> dict:
    schema = dict(raw_schema or {"type": "object", "properties": {}})
    clean_props = {}
    for prop_name, prop_def in schema.get("properties", {}).items():
        clean_props[prop_name] = {
            k: v for k, v in prop_def.items() if k in ("type", "description", "enum", "default")
        }
    return {
        "type": "object",
        "properties": clean_props,
        "required": schema.get("required", []),
    }


async def run_agentic_triage(
    driver: str,
    rival: str,
    year: int,
    gp: str,
    session_type: str,
    dataset_id: str | None = None,
) -> str:
    project_root = os.getcwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["F1_YEAR"] = str(year)
    env["F1_GP"] = gp
    env["F1_SESSION"] = session_type
    env["F1_DATASET_ID"] = dataset_id or ""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=env,
        cwd=project_root,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                tools_resp = await mcp_session.list_tools()

                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": _clean_tool_schema(
                                getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                            ),
                        },
                    }
                    for t in tools_resp.tools
                ]

                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL") or None,
                )
                target_label = dataset_id or f"{year} {gp} Grand Prix ({session_type})"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Run full pit-wall telemetry triage for {driver} against rival {rival} "
                            f"on session {target_label}."
                        ),
                    },
                ]

                while True:
                    response = client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        temperature=0.2,
                    )
                    msg = response.choices[0].message
                    messages.append(msg)

                    if not msg.tool_calls:
                        return msg.content or ""

                    for call in msg.tool_calls:
                        args = json.loads(call.function.arguments or "{}")
                        result = await mcp_session.call_tool(call.function.name, arguments=args)
                        tool_output = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": tool_output,
                        })
    except (RateLimitError, APIError) as exc:
        fallback = generate_offline_briefing(driver, rival, year, gp, session_type, dataset_id=dataset_id)
        return f"[WARN] LLM provider unavailable ({exc.__class__.__name__}); falling back to deterministic pit-wall engine.\n\n{fallback}"


def generate_offline_briefing(
    driver: str,
    rival: str,
    year: int,
    gp: str,
    session_type: str,
    dataset_id: str | None = None,
) -> str:
    eng = PitWallTelemetryEngine(year, gp, session_type, dataset_id=dataset_id)
    weather = eng.get_weather_summary()
    anom = eng.detect_lap_anomalies(driver)
    cliff = eng.analyze_tire_cliff(driver)
    pit = eng.evaluate_pit_window_shifts(driver, rival)

    cliffs = [s for s in cliff["stints"] if s["cliff_detected"]]
    severity = (
        "CRITICAL"
        if (cliffs and anom["anomaly_count"] > 0)
        else ("HIGH" if cliffs or anom["anomaly_count"] > 0 else "NOMINAL")
    )

    label = dataset_id or f"{year} {gp} {session_type}"
    lines = [
        f"=== PIT-WALL STRATEGY BRIEFING: {driver} vs {rival} ({label}) ===",
        f"TRIAGE SEVERITY: {severity}",
        f"WEATHER: Track {weather.get('track_temp_min_c')}C-{weather.get('track_temp_max_c')}C | Rain: {weather.get('rainfall_detected')}",
        f"\n[1] LAP-TIME ANOMALIES DETECTED: {anom['anomaly_count']}",
    ]
    for a in anom["anomalies"]:
        lines.append(
            f"  - Lap {a['lap']} ({a['compound']}, Age {a['tyre_life']}): "
            f"{a['lap_time_sec']}s (+{a['delta_sec']}s vs rolling median, Z={a['z_score']})"
        )

    lines.append("\n[2] TIRE DEGRADATION & CLIFF ANALYSIS:")
    for s in cliff["stints"]:
        flag = "CLIFF DETECTED" if s["cliff_detected"] else "STABLE"
        lines.append(
            f"  - Stint {s['stint']} ({s['compound']}, Laps {s['tyre_life_range'][0]}-{s['tyre_life_range'][1]}): "
            f"Slope {s['deg_slope_sec_per_lap']} s/lap | Drop {s['stint_pace_drop_sec']}s [{flag}]"
        )

    lines.append("\n[3] PIT-WINDOW & UNDERCUT SHIFTS:")
    for p in pit["pit_window_events"]:
        lines.append(
            f"  - Lap {p['pit_lap']}: Gap {p['gap_before_sec']}s -> {p['gap_after_sec']}s "
            f"(Swing {p['net_gap_swing_sec']}s | {p['window_shift']})"
        )

    return "\n".join(lines)