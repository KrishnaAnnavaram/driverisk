"""Synthetic fleet: long-format telematics for many drivers, with no download and no network.

Each driver has a hidden style value. A higher value gives higher speeds, more speeding and
more harsh events. Rain and night add harsh events. The weather and the road map come from
the same offline providers that the pipeline uses, so the context in the features is the
context that the simulator used. The traces are fake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driverisk.context.roads import OfflineRoads
from driverisk.context.weather import OfflineWeather

BASE_LAT, BASE_LON = 52.20, 0.12


def _trip(rng, t0, lat, lon, style, roads, weather, duration_s):
    heading = rng.uniform(0, 2 * np.pi)
    hour = t0.floor("h")
    w = weather.hourly(lat, lon, hour, hour)
    rainy = float(w["precip_mm"].iloc[0]) >= 0.1
    night = t0.hour >= 22 or t0.hour <= 5
    p_event = 0.0025 * style**1.6 * (1.8 if rainy else 1.0) * (1.4 if night else 1.0)
    v = 0.0
    forced, forced_left = 0.0, 0
    rows = np.zeros((duration_s, 6))
    for s in range(duration_s):
        cls, limit = roads.lookup([lat], [lon])
        limit = float(limit[0]) if not np.isnan(limit[0]) else 50.0
        target = limit * (0.82 + 0.13 * style) * (0.92 if rainy else 1.0) / 3.6
        if s > duration_s - 40:
            target = 0.0
        if forced_left > 0:
            a = forced
            forced_left -= 1
        else:
            a = float(np.clip(0.12 * (target - v), -2.2, 1.2 + 0.5 * style) + rng.normal(0, 0.18 * style))
            if v > 5 and rng.random() < p_event:
                forced = -rng.uniform(3.3, 5.5) if rng.random() < 0.6 else rng.uniform(2.7, 3.6)
                forced_left = 1
                a = forced
        v = max(0.0, v + a)
        lateral = rng.normal(0, 0.35)
        if v > 5 and rng.random() < 0.3 * p_event:
            lateral = rng.choice([-1, 1]) * rng.uniform(3.2, 4.4)
        heading += rng.normal(0, 0.02)
        step = v / 6_371_000.0
        lat += np.degrees(step * np.cos(heading))
        lon += np.degrees(step * np.sin(heading) / np.cos(np.radians(lat)))
        rpm = 800 + v * 3.6 * 28 + rng.normal(0, 60)
        rows[s] = (v * 3.6, a, lateral, lat, lon, rpm)
    return rows, lat, lon


def simulate_fleet(n_drivers: int = 30, trips_per_driver: int = 10, seed: int = 0,
                   start: str = "2024-03-04", min_trip_min: int = 4, max_trip_min: int = 16) -> pd.DataFrame:
    """Return long-format readings: `deviceId, timestamp, variable, value` (all text)."""
    if n_drivers < 2 or trips_per_driver < 1:
        raise ValueError("need at least 2 drivers and 1 trip for each driver")
    rng = np.random.default_rng(seed)
    roads, weather = OfflineRoads(seed), OfflineWeather(seed)
    frames = []
    t_start = pd.Timestamp(start, tz="UTC")
    for d in range(n_drivers):
        device = f"DEV{d:03d}"
        style = float(rng.lognormal(0.0, 0.4))
        night_lover = rng.random() < 0.2
        lat = BASE_LAT + rng.normal(0, 0.12)
        lon = BASE_LON + rng.normal(0, 0.2)
        t = t_start + pd.Timedelta(hours=float(rng.uniform(6, 20)))
        for _ in range(trips_per_driver):
            duration = int(rng.integers(min_trip_min, max_trip_min + 1) * 60)
            rows, lat, lon = _trip(rng, t, lat, lon, style, roads, weather, duration)
            times = t + pd.to_timedelta(np.arange(duration), unit="s")
            stamp = times.strftime("%Y-%m-%dT%H:%M:%SZ")
            wide = pd.DataFrame(
                {
                    "Vehicle speed": np.round(rows[:, 0], 2).astype(str),
                    "ACCELERATION X": np.round(rows[:, 1], 3).astype(str),
                    "ACCELERATION Y": np.round(rows[:, 2], 3).astype(str),
                    "ENGINE RPM": np.round(rows[:, 5], 0).astype(str),
                    "POSITION": [f"{a:.6f},{b:.6f},{30.0:.1f}" for a, b in zip(rows[:, 3], rows[:, 4])],
                },
                index=stamp,
            )
            long = wide.rename_axis("timestamp").reset_index().melt(
                id_vars="timestamp", var_name="variable", value_name="value"
            )
            ign = pd.DataFrame(
                {"timestamp": [stamp[0], (times[-1] + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")],
                 "variable": "IGNITION_STATUS", "value": ["1", "0"]}
            )
            frames.append(pd.concat([ign.iloc[:1], long, ign.iloc[1:]], ignore_index=True).assign(deviceId=device))
            gap_h = rng.uniform(1.5, 30)
            t = times[-1] + pd.Timedelta(hours=float(gap_h))
            if night_lover and rng.random() < 0.5:
                t = t.floor("D") + pd.Timedelta(hours=float(rng.uniform(22, 26)))
            elif t.hour < 6:
                t = t + pd.Timedelta(hours=7)
    out = pd.concat(frames, ignore_index=True)
    return out.loc[:, ["deviceId", "timestamp", "variable", "value"]]
