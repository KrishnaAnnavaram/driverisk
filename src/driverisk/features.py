"""Trip table: one row for each trip with exposure, features and the event target.

Features come from speed, time, position and context. The target comes from the
acceleration signals. `assert_no_label_inputs` checks that no feature reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driverisk.events import LABEL_INPUTS, EventRules, trip_events

ROAD_SHARE_COLUMNS = ("share_motorway", "share_primary", "share_secondary", "share_residential", "share_unknown")
FEATURES = (
    "duration_min",
    "mean_speed_kmh",
    "p50_speed_kmh",
    "p95_speed_kmh",
    "max_speed_kmh",
    "speed_std_kmh",
    "idle_share",
    "night_share",
    "weekend",
    "start_hour",
    "rain_share",
    "mean_precip_mm",
    "mean_wind_ms",
    "mean_temp_c",
    *ROAD_SHARE_COLUMNS,
    "speeding_share",
    "p95_rpm",
    "collision_density",
)
TARGET = "harsh_events"
EXPOSURE = "distance_km"
GROUP = "device"


class LabelInFeatures(AssertionError):
    """A feature name points to a signal that defines the target."""


@dataclass(frozen=True)
class TripRules:
    min_distance_km: float = 0.5
    min_duration_min: float = 2.0
    speeding_margin_kmh: float = 10.0
    rain_mm: float = 0.1
    night_hours: tuple[int, ...] = (22, 23, 0, 1, 2, 3, 4, 5)


def assert_no_label_inputs(columns) -> None:
    bad = [c for c in columns if any(sig in str(c) for sig in LABEL_INPUTS) or str(c).startswith("harsh_")]
    if bad:
        raise LabelInFeatures(f"features read target signals: {bad}")


def _trip_row(g: pd.DataFrame, rules: TripRules, events: EventRules) -> dict:
    speed = g["speed_kmh"].to_numpy(dtype=float)
    secs = g["dt_s"].to_numpy(dtype=float)
    hours = g["time"].dt.hour.to_numpy()
    w = np.where(secs > 0, secs, 1.0)
    duration_min = (g["time"].iloc[-1] - g["time"].iloc[0]).total_seconds() / 60
    sp = speed[~np.isnan(speed)]
    row = {
        "trip_id": g["trip_id"].iloc[0],
        "device": g["device"].iloc[0],
        "start": g["time"].iloc[0],
        "distance_km": float(g["dist_m"].sum() / 1000),
        "duration_min": float(duration_min),
        "mean_speed_kmh": float(np.mean(sp)) if sp.size else np.nan,
        "p50_speed_kmh": float(np.percentile(sp, 50)) if sp.size else np.nan,
        "p95_speed_kmh": float(np.percentile(sp, 95)) if sp.size else np.nan,
        "max_speed_kmh": float(np.max(sp)) if sp.size else np.nan,
        "speed_std_kmh": float(np.std(sp)) if sp.size else np.nan,
        "idle_share": float(np.mean(sp < 2)) if sp.size else np.nan,
        "night_share": float(np.average(np.isin(hours, rules.night_hours), weights=w)),
        "weekend": int(g["time"].iloc[0].dayofweek >= 5),
        "start_hour": int(g["time"].iloc[0].hour),
    }
    precip = g["precip_mm"].to_numpy(dtype=float) if "precip_mm" in g else np.full(len(g), np.nan)
    known = ~np.isnan(precip)
    row["rain_share"] = float(np.mean(precip[known] >= rules.rain_mm)) if known.any() else np.nan
    row["mean_precip_mm"] = float(np.nanmean(precip)) if known.any() else np.nan
    for col, name in (("wind_ms", "mean_wind_ms"), ("temp_c", "mean_temp_c")):
        vals = g[col].to_numpy(dtype=float) if col in g else np.full(len(g), np.nan)
        row[name] = float(np.nanmean(vals)) if (~np.isnan(vals)).any() else np.nan
    road = g["road_class"].astype(str) if "road_class" in g else pd.Series(["unknown"] * len(g))
    for share in ROAD_SHARE_COLUMNS:
        row[share] = float((road == share.removeprefix("share_")).mean())
    limit = g["speed_limit_kmh"].to_numpy(dtype=float) if "speed_limit_kmh" in g else np.full(len(g), np.nan)
    has_limit = ~np.isnan(limit) & ~np.isnan(speed)
    row["speeding_share"] = (
        float(np.mean(speed[has_limit] > limit[has_limit] + rules.speeding_margin_kmh)) if has_limit.any() else np.nan
    )
    rpm = g["rpm"].to_numpy(dtype=float)
    row["p95_rpm"] = float(np.nanpercentile(rpm, 95)) if (~np.isnan(rpm)).any() else np.nan
    dens = g["collision_density"].to_numpy(dtype=float) if "collision_density" in g else np.full(len(g), np.nan)
    row["collision_density"] = float(np.nanmean(dens)) if (~np.isnan(dens)).any() else np.nan
    row.update(trip_events(g, events))
    return row


def build_trip_table(points: pd.DataFrame, rules: TripRules = TripRules(), events: EventRules = EventRules()) -> pd.DataFrame:
    """Aggregate per-second readings (with kinematics and context) into the trip table."""
    rows = [_trip_row(g, rules, events) for _, g in points.groupby("trip_id", sort=False) if len(g) >= 2]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    keep = (table["distance_km"] >= rules.min_distance_km) & (table["duration_min"] >= rules.min_duration_min)
    table = table[keep].reset_index(drop=True)
    assert_no_label_inputs(FEATURES)
    return table
