import json
import os
import fastf1

os.makedirs("cache", exist_ok=True)
os.makedirs("data/real_sessions", exist_ok=True)
fastf1.Cache.configure(cache_dir="cache")

BENCHMARK_RACES = [
    {"year": 2024, "gp": "Silverstone", "session": "R", "drivers": ["HAM", "VER", "NOR", "LEC", "SAI"]},
    {"year": 2024, "gp": "Monza", "session": "R", "drivers": ["LEC", "PIA", "NOR", "SAI", "VER"]},
    {"year": 2023, "gp": "Singapore", "session": "R", "drivers": ["SAI", "NOR", "HAM", "RUS", "LEC"]},
]


def export_real_sessions():
    manifest = []
    for cfg in BENCHMARK_RACES:
        tag = f"{cfg['year']}_{cfg['gp']}_{cfg['session']}"
        print(f"[INGEST] Loading real FastF1 telemetry for {tag}...")
        session = fastf1.get_session(cfg["year"], cfg["gp"], cfg["session"])
        session.load(telemetry=False, weather=True)
        laps = session.laps.copy()
        laps_path = f"data/real_sessions/{tag}_laps.parquet"
        weather_path = f"data/real_sessions/{tag}_weather.parquet"
        laps.to_parquet(laps_path, index=False)
        if session.weather_data is not None and not session.weather_data.empty:
            session.weather_data.to_parquet(weather_path, index=False)
        manifest.append({"dataset_id": tag, "year": cfg["year"], "gp": cfg["gp"], "session": cfg["session"],
                         "total_laps": int(laps["LapNumber"].max()), "total_rows": int(len(laps)),
                         "featured_drivers": cfg["drivers"], "laps_file": laps_path, "weather_file": weather_path})
        print(f"[SAVED] {laps_path} ({len(laps)} real lap records)")

    with open("data/real_sessions/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[DONE] Real F1 dataset manifest written to data/real_sessions/manifest.json")


if __name__ == "__main__":
    export_real_sessions()