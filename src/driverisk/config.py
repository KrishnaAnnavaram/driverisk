"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PREFIX = "DRIVERISK_"
WEATHER_SOURCES = ("offline", "csv", "open-meteo")
ROAD_SOURCES = ("offline", "csv")


def load_env_file(path: str | os.PathLike = ".env") -> int:
    """Read KEY=VALUE lines into os.environ. Existing variables win. Returns the count of new keys."""
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            added += 1
    return added


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(PREFIX + name, "")
    return value.strip() if value.strip() else default


@dataclass(frozen=True)
class Settings:
    data_path: str | None
    seed: int
    output_dir: str
    weather: str
    weather_csv: str | None
    roads: str
    roads_csv: str | None
    collisions_csv: str | None
    cache_dir: str
    user_agent: str
    acc_unit: str
    threads: int

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            data_path=_get("DATA") or None,
            seed=int(_get("SEED", "42")),
            output_dir=_get("OUTPUT", "runs"),
            weather=_get("WEATHER", "offline").lower(),
            weather_csv=_get("WEATHER_CSV") or None,
            roads=_get("ROADS", "offline").lower(),
            roads_csv=_get("ROADS_CSV") or None,
            collisions_csv=_get("COLLISIONS_CSV") or None,
            cache_dir=_get("CACHE_DIR", ".cache/driverisk"),
            user_agent=_get("USER_AGENT", "driverisk/0.1 (research use)"),
            acc_unit=_get("ACC_UNIT", "ms2").lower(),
            threads=int(_get("THREADS", "4")),
        )
        s.check()
        return s

    def check(self) -> None:
        if self.weather not in WEATHER_SOURCES:
            raise ValueError(f"DRIVERISK_WEATHER must be one of {WEATHER_SOURCES}")
        if self.roads not in ROAD_SOURCES:
            raise ValueError(f"DRIVERISK_ROADS must be one of {ROAD_SOURCES}")
        if self.weather == "csv" and not self.weather_csv:
            raise ValueError("DRIVERISK_WEATHER=csv needs DRIVERISK_WEATHER_CSV")
        if self.roads == "csv" and not self.roads_csv:
            raise ValueError("DRIVERISK_ROADS=csv needs DRIVERISK_ROADS_CSV")
        if self.acc_unit not in ("ms2", "g"):
            raise ValueError("DRIVERISK_ACC_UNIT must be ms2 or g")
        if self.threads < 1:
            raise ValueError("DRIVERISK_THREADS must be 1 or more")
