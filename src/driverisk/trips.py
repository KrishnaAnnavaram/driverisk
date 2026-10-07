"""Trip segmentation and per-second kinematics.

A new trip starts when the ignition changes from off to on, or when the time gap between two
readings of a device is longer than `max_gap_s`. Rows with the ignition off are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def segment_trips(wide: pd.DataFrame, max_gap_s: float = 600.0) -> pd.DataFrame:
    """Add `trip_id` to each row with the ignition on (or unknown). Returns the kept rows."""
    out = []
    for device, g in wide.sort_values(["device", "time"]).groupby("device", sort=False):
        g = g.copy()
        gap = g["time"].diff().dt.total_seconds()
        ign = g["ignition"]
        prev_ign = ign.shift()
        new_trip = gap.isna() | (gap > max_gap_s) | ((prev_ign == 0) & (ign == 1))
        g["trip_no"] = new_trip.cumsum()
        g = g[ign.fillna(1) != 0]
        if g.empty:
            continue
        # renumber so that trips with only ignition-off rows leave no hole
        codes = pd.factorize(g["trip_no"])[0]
        g["trip_id"] = [f"{device}-{c:04d}" for c in codes]
        out.append(g.drop(columns="trip_no"))
    if not out:
        return wide.iloc[0:0].assign(trip_id=pd.Series(dtype=str))
    return pd.concat(out, ignore_index=True)


def add_kinematics(trips: pd.DataFrame, max_dt_s: float = 3.0) -> pd.DataFrame:
    """Add `dt_s`, `dist_m` and `acc_long` (m/s^2) to each row, inside each trip.

    * `acc_long` is `acc_x` when the device sends it. Else it is the speed change over time,
      only where two readings are at most `max_dt_s` apart.
    * `dist_m` is the GPS distance when both positions are known. Else it is speed x time.
    """
    parts = []
    for _, g in trips.groupby("trip_id", sort=False):
        g = g.sort_values("time").copy()
        dt = g["time"].diff().dt.total_seconds().to_numpy()
        v = g["speed_kmh"].to_numpy(dtype=float) / 3.6
        dv = np.diff(v, prepend=np.nan)
        derived = np.where((dt > 0) & (dt <= max_dt_s), dv / np.where(dt > 0, dt, 1), np.nan)
        acc_x = g["acc_x"].to_numpy(dtype=float)
        g["acc_long"] = np.where(np.isnan(acc_x), derived, acc_x)
        gps = haversine_m(g["lat"].shift(), g["lon"].shift(), g["lat"], g["lon"])
        v_mean = np.nanmean(np.vstack([v, np.roll(v, 1)]), axis=0) if len(v) else v
        by_speed = np.where((dt > 0) & (dt <= 60), v_mean * dt, 0.0)
        dist = np.where(np.isnan(gps) | (dt > 60), by_speed, gps)
        g["dt_s"] = np.nan_to_num(dt, nan=0.0)
        g["dist_m"] = np.nan_to_num(dist, nan=0.0)
        parts.append(g)
    if not parts:
        return trips.assign(acc_long=np.nan, dt_s=0.0, dist_m=0.0)
    return pd.concat(parts, ignore_index=True)
