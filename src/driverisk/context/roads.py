"""Road class and speed limit for each reading, from an offline source. No per-row API calls.

* `RoadSegmentsCsv`: points along road segments that you prepare once from an OpenStreetMap
  extract (`lat, lon, road_class, speed_limit_kmh`). A BallTree (haversine) finds the nearest
  point. A reading farther than `max_m` metres gets the class `unknown`.
* `OfflineRoads`: a synthetic road map for the demo and the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from driverisk.context.weather import _stable_seed
from driverisk.trips import EARTH_RADIUS_M

ROAD_CLASSES = ("motorway", "primary", "secondary", "residential")
DEFAULT_LIMITS = {"motorway": 110.0, "primary": 80.0, "secondary": 60.0, "residential": 40.0}


class OfflineRoads:
    """Synthetic road map: each cell of `cell_deg` degrees has one class and its default limit."""

    name = "offline"

    def __init__(self, seed: int = 0, cell_deg: float = 0.02):
        self.seed = seed
        self.cell_deg = cell_deg
        self._cache: dict[tuple[int, int], str] = {}

    def lookup(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        classes = np.full(lat.shape, "unknown", dtype=object)
        limits = np.full(lat.shape, np.nan)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        cache = self._cache
        for i in np.flatnonzero(ok):
            key = (int(round(lat[i] / self.cell_deg)), int(round(lon[i] / self.cell_deg)))
            if key not in cache:
                u = (_stable_seed(self.seed, "road", key) % 10_000) / 10_000
                cache[key] = ROAD_CLASSES[int(np.searchsorted([0.12, 0.42, 0.70], u, side="right"))]
            classes[i] = cache[key]
            limits[i] = DEFAULT_LIMITS[cache[key]]
        return classes, limits


class RoadSegmentsCsv:
    name = "csv"

    def __init__(self, path: str | None = None, frame: pd.DataFrame | None = None, max_m: float = 50.0):
        df = frame if frame is not None else pd.read_csv(path)
        need = {"lat", "lon", "road_class", "speed_limit_kmh"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"road CSV misses columns {sorted(missing)}")
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("road CSV has no rows with a position")
        self.classes = df["road_class"].astype(str).to_numpy(dtype=object)
        self.limits = pd.to_numeric(df["speed_limit_kmh"], errors="coerce").to_numpy(dtype=float)
        self.tree = BallTree(np.radians(df[["lat", "lon"]].to_numpy(dtype=float)), metric="haversine")
        self.max_m = max_m

    def lookup(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        classes = np.full(lat.shape, "unknown", dtype=object)
        limits = np.full(lat.shape, np.nan)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        if ok.any():
            dist, idx = self.tree.query(np.radians(np.column_stack([lat[ok], lon[ok]])), k=1)
            near = dist[:, 0] * EARTH_RADIUS_M <= self.max_m
            pos = np.flatnonzero(ok)
            classes[pos[near]] = self.classes[idx[near, 0]]
            limits[pos[near]] = self.limits[idx[near, 0]]
        return classes, limits


def attach_roads(points: pd.DataFrame, provider) -> pd.DataFrame:
    out = points.copy()
    classes, limits = provider.lookup(out["lat"].to_numpy(), out["lon"].to_numpy())
    out["road_class"] = classes
    out["speed_limit_kmh"] = limits
    return out
