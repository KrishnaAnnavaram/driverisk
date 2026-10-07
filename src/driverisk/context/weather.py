"""Hourly weather for the place and the hour of each reading.

Each provider answers `hourly(lat, lon, start, end)` with a frame indexed by UTC hour and the
columns `temp_c`, `precip_mm`, `wind_ms`. `attach_weather` asks once for each weather cell
(0.1 degree) and day range, then joins on (cell, hour). A reading with no position gets no
weather: the join never uses the hour alone.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from driverisk.trips import haversine_m

WEATHER_COLUMNS = ("temp_c", "precip_mm", "wind_ms")
CELL_DEG = 0.1


class WeatherProvider(Protocol):
    name: str

    def hourly(self, lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...


def _hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little")


class OfflineWeather:
    """Deterministic synthetic weather. The same (seed, cell, day) always gives the same hours."""

    name = "offline"

    def __init__(self, seed: int = 0, rain_day_share: float = 0.35):
        self.seed = seed
        self.rain_day_share = rain_day_share

    def hourly(self, lat, lon, start, end) -> pd.DataFrame:
        idx = _hours(start, end)
        cell = (round(lat / CELL_DEG), round(lon / CELL_DEG))
        rows = []
        for day, hours in pd.Series(idx, index=idx).groupby(idx.floor("D")):
            rng = np.random.default_rng(_stable_seed(self.seed, cell, day.date()))
            rainy = rng.random() < self.rain_day_share
            start_h, length = rng.integers(0, 24), rng.integers(2, 9)
            base_t = 12 + 8 * np.sin(2 * np.pi * (day.dayofyear - 110) / 365) + rng.normal(0, 2)
            wind = rng.gamma(2.0, 1.8)
            for h in hours:
                in_rain = rainy and (start_h <= h.hour < start_h + length)
                rows.append(
                    {
                        "time": h,
                        "temp_c": base_t + 4 * np.sin(2 * np.pi * (h.hour - 9) / 24),
                        "precip_mm": float(rng.gamma(1.5, 1.2)) if in_rain else 0.0,
                        "wind_ms": max(0.0, wind + rng.normal(0, 0.8)),
                    }
                )
        return pd.DataFrame(rows).set_index("time")


class CsvWeather:
    """Station observations from a CSV: `station_id, lat, lon, time, temp_c, precip_mm, wind_ms`.

    A cell uses the nearest station within `max_km`. With no station that near, the cell
    gets no weather (the prototype joined a station from another country on the hour only).
    """

    name = "csv"

    def __init__(self, path: str, max_km: float = 50.0):
        df = pd.read_csv(path)
        need = {"station_id", "lat", "lon", "time", *WEATHER_COLUMNS}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"weather CSV misses columns {sorted(missing)}")
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
        self.obs = df.groupby(["station_id", "time"])[list(WEATHER_COLUMNS)].mean()
        self.stations = df.groupby("station_id")[["lat", "lon"]].first()
        self.max_km = max_km

    def nearest_station(self, lat: float, lon: float):
        d = haversine_m(lat, lon, self.stations["lat"], self.stations["lon"]) / 1000
        i = int(np.argmin(d))
        return (self.stations.index[i], float(d[i])) if d[i] <= self.max_km else (None, float(d[i]))

    def hourly(self, lat, lon, start, end) -> pd.DataFrame:
        idx = _hours(start, end)
        station, _ = self.nearest_station(lat, lon)
        if station is None:
            return pd.DataFrame(np.nan, index=idx, columns=list(WEATHER_COLUMNS))
        return self.obs.loc[station].reindex(idx)


class OpenMeteoWeather:
    """Open-Meteo historical weather API (no key). One request for each cell and day range, cached on disk."""

    name = "open-meteo"
    URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, cache_dir: str = ".cache/driverisk", user_agent: str = "driverisk/0.1", timeout: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self.timeout = timeout

    def _fetch_json(self, url: str) -> dict:  # pragma: no cover - network
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def url(self, lat, lon, start, end) -> str:
        q = {
            "latitude": f"{lat:.2f}",
            "longitude": f"{lon:.2f}",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        return f"{self.URL}?{urllib.parse.urlencode(q)}"

    def hourly(self, lat, lon, start, end) -> pd.DataFrame:
        url = self.url(lat, lon, start, end)
        cache = self.cache_dir / f"openmeteo_{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"
        if cache.is_file():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = self._fetch_json(url)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        h = data["hourly"]
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(h["time"], utc=True),
                "temp_c": h["temperature_2m"],
                "precip_mm": h["precipitation"],
                "wind_ms": h["wind_speed_10m"],
            }
        ).set_index("time")
        return frame.reindex(_hours(start, end))


def attach_weather(points: pd.DataFrame, provider: WeatherProvider) -> pd.DataFrame:
    """Add the weather columns to each reading by (cell, hour). Readings with no position get NaN."""
    out = points.copy()
    for col in WEATHER_COLUMNS:
        out[col] = np.nan
    has_pos = out["lat"].notna() & out["lon"].notna()
    if not has_pos.any():
        return out
    cells = pd.DataFrame(
        {
            "clat": np.round(out.loc[has_pos, "lat"] / CELL_DEG).astype(int),
            "clon": np.round(out.loc[has_pos, "lon"] / CELL_DEG).astype(int),
            "hour": out.loc[has_pos, "time"].dt.floor("h"),
        },
        index=out.index[has_pos],
    )
    frames = []
    for (clat, clon), g in cells.groupby(["clat", "clon"]):
        start, end = g["hour"].min(), g["hour"].max()
        w = provider.hourly(clat * CELL_DEG, clon * CELL_DEG, start.floor("D"), end.floor("D") + pd.Timedelta(hours=23))
        w = w.reindex(columns=list(WEATHER_COLUMNS))
        w.index.name = "hour"
        frames.append(w.reset_index().assign(clat=clat, clon=clon))
    table = pd.concat(frames, ignore_index=True)
    merged = cells.reset_index().merge(table, on=["clat", "clon", "hour"], how="left").set_index("index")
    out.loc[merged.index, list(WEATHER_COLUMNS)] = merged[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
    return out
