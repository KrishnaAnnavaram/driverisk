"""Collision density prior from a road-safety collision table with coordinates.

The UK DfT *collision* table has `latitude` and `longitude`, so it can give a count of
collisions for each grid cell. The DfT *casualty* table has no location, so it cannot be
joined to a trip: `from_csv` refuses it with a clear message. The prior is used only when
most readings are inside the area of the table (`coverage`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CollisionPrior:
    def __init__(self, lat, lon, cell_deg: float = 0.01):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        if not ok.any():
            raise ValueError("the collision table has no valid coordinates")
        self.cell_deg = cell_deg
        keys = pd.Series(list(zip(np.round(lat[ok] / cell_deg).astype(int), np.round(lon[ok] / cell_deg).astype(int))))
        self.counts = keys.value_counts().to_dict()
        self.bbox = (lat[ok].min(), lat[ok].max(), lon[ok].min(), lon[ok].max())

    @classmethod
    def from_csv(cls, path: str, cell_deg: float = 0.01) -> "CollisionPrior":
        head = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        cols = {c.lower(): c for c in head.columns}
        if "latitude" not in cols or "longitude" not in cols:
            raise ValueError(
                "the collision prior needs `latitude` and `longitude` columns. "
                "A casualty table has no location: use the collision table of the same source"
            )
        df = pd.read_csv(path, usecols=[cols["latitude"], cols["longitude"]], encoding="utf-8-sig")
        lat = pd.to_numeric(df[cols["latitude"]], errors="coerce")
        lon = pd.to_numeric(df[cols["longitude"]], errors="coerce")
        return cls(lat, lon, cell_deg)

    def coverage(self, lat, lon) -> float:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        if not ok.any():
            return 0.0
        a, b, c, d = self.bbox
        inside = (lat[ok] >= a) & (lat[ok] <= b) & (lon[ok] >= c) & (lon[ok] <= d)
        return float(inside.mean())

    def density(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out = np.full(lat.shape, np.nan)
        ok = ~(np.isnan(lat) | np.isnan(lon))
        for i in np.flatnonzero(ok):
            key = (int(round(lat[i] / self.cell_deg)), int(round(lon[i] / self.cell_deg)))
            out[i] = self.counts.get(key, 0)
        return out
