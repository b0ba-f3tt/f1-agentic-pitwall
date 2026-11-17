import os
import fastf1
import numpy as np
import pandas as pd
from scipy.stats import linregress

os.makedirs("cache", exist_ok=True)
fastf1.Cache.configure(cache_dir="cache")


class PitWallTelemetryEngine:
    def __init__(self, year: int = 2024, gp: str = "Silverstone", session_type: str = "R", dataset_id: str | None = None):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.dataset_id = dataset_id

        if dataset_id:
            laps_file = f"data/real_sessions/{dataset_id}_laps.parquet"
            weather_file = f"data/real_sessions/{dataset_id}_weather.parquet"
            if not os.path.exists(laps_file):
                raise FileNotFoundError(f"Real dataset snapshot not found: {laps_file}")
            raw_laps = pd.read_parquet(laps_file)
            self.weather_data = pd.read_parquet(weather_file) if os.path.exists(weather_file) else pd.DataFrame()
        else:
            session = fastf1.get_session(year, gp, session_type)
            session.load(telemetry=False, weather=True)
            raw_laps = session.laps.copy()
            self.weather_data = session.weather_data.copy() if session.weather_data is not None else pd.DataFrame()

        self.laps = self._prepare_clean_laps(raw_laps)

    def _prepare_clean_laps(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df[raw_df["LapTime"].notna()].copy()
        df["LapTime"] = pd.to_timedelta(df["LapTime"]) if not pd.api.types.is_timedelta64_dtype(df["LapTime"]) else df["LapTime"]
        df["Time"] = pd.to_timedelta(df["Time"]) if not pd.api.types.is_timedelta64_dtype(df["Time"]) else df["Time"]
        total_laps = int(df["LapNumber"].max())
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
        df["IsPitLap"] = df["PitInTime"].notna() | df["PitOutTime"].notna()
        df["IsGreenFlag"] = df["TrackStatus"].astype(str) == "1"
        df["FuelCorrectedSec"] = df["LapTimeSec"] - ((total_laps - df["LapNumber"]) * 0.055)
        return df

    def get_driver_list(self) -> list[str]:
        return sorted(self.laps["Driver"].dropna().unique().tolist())

    def get_weather_summary(self) -> dict:
        w = self.weather_data
        if w is None or w.empty:
            return {"status": "NO_WEATHER_DATA"}
        return {
            "air_temp_min_c": round(float(w["AirTemp"].min()), 1),
            "air_temp_max_c": round(float(w["AirTemp"].max()), 1),
            "track_temp_min_c": round(float(w["TrackTemp"].min()), 1),
            "track_temp_max_c": round(float(w["TrackTemp"].max()), 1),
            "track_temp_delta_c": round(float(w["TrackTemp"].iloc[-1] - w["TrackTemp"].iloc[0]), 1),
            "rainfall_detected": bool(w["Rainfall"].any()),
        }

    def detect_lap_anomalies(self, driver: str, max_lap: int | None = None, z_threshold: float = 2.8) -> dict:
        drv_laps = self.laps[(self.laps["Driver"] == driver) & (self.laps["LapNumber"] > 1) & (~self.laps["IsPitLap"]) & (self.laps["IsGreenFlag"])].copy()
        if max_lap is not None:
            drv_laps = drv_laps[drv_laps["LapNumber"] <= max_lap]
        if len(drv_laps) < 5:
            return {"driver": driver, "anomaly_count": 0, "anomalies": [], "status": "INSUFFICIENT_DATA"}

        rolling_med = drv_laps["FuelCorrectedSec"].rolling(window=5, min_periods=3, center=True).median()
        residual = drv_laps["FuelCorrectedSec"] - rolling_med
        mad = max(float(np.median(np.abs(residual.dropna()))), 0.25)
        modified_z = 0.6745 * residual / mad
        drv_laps["DeltaToRollingSec"], drv_laps["ModifiedZ"] = residual.round(3), modified_z.round(2)
        flagged = drv_laps[drv_laps["ModifiedZ"] > z_threshold]
        anomalies = [{"lap": int(r["LapNumber"]), "stint": int(r["Stint"]), "compound": str(r["Compound"]),
                      "tyre_life": int(r["TyreLife"]), "lap_time_sec": round(float(r["LapTimeSec"]), 3),
                      "delta_sec": float(r["DeltaToRollingSec"]), "z_score": float(r["ModifiedZ"])} for _, r in flagged.iterrows()]

        return {"driver": driver, "anomaly_count": len(anomalies), "anomalies": anomalies}

    def analyze_tire_cliff(self, driver: str, max_lap: int | None = None, deg_threshold_sec_per_lap: float = 0.085) -> dict:
        drv_laps = self.laps[(self.laps["Driver"] == driver) & (~self.laps["IsPitLap"]) & (self.laps["IsGreenFlag"])].copy()
        if max_lap is not None:
            drv_laps = drv_laps[drv_laps["LapNumber"] <= max_lap]

        stint_reports = []
        for stint_id, group in drv_laps.groupby("Stint"):
            if len(group) < 4:
                continue
            x, y = group["TyreLife"].astype(float).values, group["FuelCorrectedSec"].astype(float).values
            slope, _, r_val, _, _ = linregress(x, y)
            tail_drop = float(group.tail(3)["FuelCorrectedSec"].mean() - group.head(3)["FuelCorrectedSec"].mean())
            cliff_detected = bool(slope >= deg_threshold_sec_per_lap or tail_drop >= 0.95)
            stint_reports.append({"stint": int(stint_id), "compound": str(group["Compound"].iloc[0]),
                                  "laps_in_stint": int(len(group)), "tyre_life_range": [int(x.min()), int(x.max())],
                                  "deg_slope_sec_per_lap": round(float(slope), 4), "r_squared": round(float(r_val ** 2), 3),
                                  "stint_pace_drop_sec": round(tail_drop, 3), "cliff_detected": cliff_detected})

        return {"driver": driver, "stints": stint_reports}

    def evaluate_pit_window_shifts(self, driver: str, rival_driver: str, max_lap: int | None = None) -> dict:
        d1 = self.laps[self.laps["Driver"] == driver].set_index("LapNumber")
        d2 = self.laps[self.laps["Driver"] == rival_driver].set_index("LapNumber")
        common_laps = d1.index.intersection(d2.index)
        if max_lap is not None:
            common_laps = common_laps[common_laps <= max_lap]

        pit_events = []
        for lap in common_laps:
            d1_pit, d2_pit = pd.notna(d1.loc[lap, "PitInTime"]), pd.notna(d2.loc[lap, "PitInTime"])
            if not (d1_pit or d2_pit):
                continue
            pre_lap, post_lap = max(int(common_laps.min()), int(lap) - 2), min(int(common_laps.max()), int(lap) + 3)
            if pre_lap not in common_laps or post_lap not in common_laps:
                continue
            t1_pre, t2_pre, t1_post, t2_post = d1.loc[pre_lap, "Time"], d2.loc[pre_lap, "Time"], d1.loc[post_lap, "Time"], d2.loc[post_lap, "Time"]
            if any(pd.isna(t) for t in (t1_pre, t2_pre, t1_post, t2_post)):
                continue

            gap_before, gap_after = float((t1_pre - t2_pre).total_seconds()), float((t1_post - t2_post).total_seconds())
            raw_swing = round(gap_after - gap_before, 3)
            both_pitted = any(pd.notna(d1.loc[l, "PitInTime"]) for l in range(pre_lap, post_lap + 1) if l in d1.index) and \
                          any(pd.notna(d2.loc[l, "PitInTime"]) for l in range(pre_lap, post_lap + 1) if l in d2.index)

            if both_pitted:
                shift_type = "UNDERCUT_GAIN" if raw_swing < -0.5 else ("UNDERCUT_LOSS" if raw_swing > 0.5 else "NEUTRAL_CYCLE")
            else:
                shift_type = "DRIVER_PIT_EXPOSURE" if d1_pit else "RIVAL_UNDERCUT_THREAT"

            pit_events.append({"pit_lap": int(lap), "driver_pitted": bool(d1_pit), "rival_pitted": bool(d2_pit),
                               "cycle_completed": bool(both_pitted), "gap_before_sec": round(gap_before, 3),
                               "gap_after_sec": round(gap_after, 3), "net_gap_swing_sec": raw_swing, "window_shift": shift_type})

        return {"driver": driver, "rival": rival_driver, "pit_window_events": pit_events}