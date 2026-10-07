"""Long-format telematics -> one wide row for each device and second.

The raw stream has one row for each reading: `deviceId, timestamp, variable, value`.
`value` holds a different signal on each row (speed, RPM, a `lat,lon,alt` text, ...), so it
is never used as one column. `pivot_wide` gives each signal its own typed column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED = ("deviceId", "timestamp", "variable", "value")
G = 9.80665

# normalised variable name -> canonical signal
SIGNALS = {
    "VEHICLESPEED": "speed_kmh",
    "SPEED": "speed_kmh",
    "ENGINERPM": "rpm",
    "RPM": "rpm",
    "ACCELERATIONX": "acc_x",
    "ACCELERATIONY": "acc_y",
    "ACCELERATIONZ": "acc_z",
    "POSITION": "position",
    "IGNITIONSTATUS": "ignition",
}
NUMERIC_SIGNALS = ("speed_kmh", "rpm", "acc_x", "acc_y", "acc_z")
WIDE_COLUMNS = ("device", "time", "speed_kmh", "rpm", "acc_x", "acc_y", "acc_z", "lat", "lon", "alt", "ignition")


class TelematicsError(ValueError):
    """The telematics table cannot be used."""


@dataclass
class IngestReport:
    rows_in: int = 0
    rows_used: int = 0
    devices: int = 0
    unknown_variables: dict[str, int] = field(default_factory=dict)
    bad_positions: int = 0
    bad_numbers: int = 0
    bad_timestamps: int = 0

    def summary(self) -> str:
        unk = ", ".join(f"{k} ({v})" for k, v in sorted(self.unknown_variables.items())) or "none"
        return (
            f"rows_in={self.rows_in} rows_used={self.rows_used} devices={self.devices} "
            f"bad_timestamps={self.bad_timestamps} bad_numbers={self.bad_numbers} "
            f"bad_positions={self.bad_positions} unknown_variables: {unk}"
        )


def normalise_variable(name: str) -> str:
    return re.sub(r"[^A-Z]", "", str(name).upper())


def parse_position(text) -> tuple[float, float, float]:
    """Parse `lat,lon` or `lat,lon,alt`. Returns NaNs for a value that is not a valid position."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return (np.nan, np.nan, np.nan)
    parts = [p.strip() for p in str(text).strip().strip("()[]").split(",")]
    if len(parts) not in (2, 3):
        return (np.nan, np.nan, np.nan)
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return (np.nan, np.nan, np.nan)
    lat, lon = nums[0], nums[1]
    alt = nums[2] if len(nums) == 3 else np.nan
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return (np.nan, np.nan, np.nan)
    return (lat, lon, alt)


def parse_ignition(value) -> float:
    s = str(value).strip().lower()
    if s in ("1", "on", "true", "yes", "1.0"):
        return 1.0
    if s in ("0", "off", "false", "no", "0.0"):
        return 0.0
    return np.nan


def pivot_wide(long_df: pd.DataFrame, acc_unit: str = "ms2", fill_limit_s: int = 5) -> tuple[pd.DataFrame, IngestReport]:
    """Return one row for each (device, second) with typed signal columns, and a report."""
    missing = [c for c in REQUIRED if c not in long_df.columns]
    if missing:
        raise TelematicsError(f"missing columns: {missing}")
    if acc_unit not in ("ms2", "g"):
        raise ValueError("acc_unit must be ms2 or g")
    rep = IngestReport(rows_in=len(long_df))
    df = long_df.loc[:, list(REQUIRED)].copy()
    df["signal"] = df["variable"].map(lambda v: SIGNALS.get(normalise_variable(v)))
    unknown = df.loc[df["signal"].isna(), "variable"].astype(str).value_counts()
    rep.unknown_variables = {str(k): int(v) for k, v in unknown.items()}
    df = df[df["signal"].notna()]

    time = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    rep.bad_timestamps = int(time.isna().sum())
    df = df.assign(time=time.dt.floor("s"), device=df["deviceId"].astype(str))
    df = df[df["time"].notna()]
    if df.empty:
        raise TelematicsError("no usable rows: check the variable names and the timestamps")

    parts = []
    num = df[df["signal"].isin(NUMERIC_SIGNALS)]
    values = pd.to_numeric(num["value"], errors="coerce")
    rep.bad_numbers = int(values.isna().sum())
    num = num.assign(v=values).dropna(subset=["v"])
    if not num.empty:
        parts.append(num.pivot_table(index=["device", "time"], columns="signal", values="v", aggfunc="last"))

    ign = df[df["signal"] == "ignition"]
    if not ign.empty:
        ign = ign.assign(ignition=ign["value"].map(parse_ignition)).dropna(subset=["ignition"])
        parts.append(ign.groupby(["device", "time"])[["ignition"]].last())

    pos = df[df["signal"] == "position"]
    if not pos.empty:
        parsed = np.array([parse_position(v) for v in pos["value"]], dtype=float).reshape(-1, 3)
        pos = pos.assign(lat=parsed[:, 0], lon=parsed[:, 1], alt=parsed[:, 2])
        rep.bad_positions = int(np.isnan(parsed[:, 0]).sum())
        pos = pos.dropna(subset=["lat", "lon"])
        parts.append(pos.groupby(["device", "time"])[["lat", "lon", "alt"]].last())

    if not parts:
        raise TelematicsError("no usable signals")
    wide = pd.concat(parts, axis=1).sort_index().reset_index()
    for col in WIDE_COLUMNS:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.loc[:, list(WIDE_COLUMNS)]
    if acc_unit == "g":
        wide[["acc_x", "acc_y", "acc_z"]] = wide[["acc_x", "acc_y", "acc_z"]] * G

    # a position or an ignition state stays valid for a few seconds
    filled = []
    for _, g in wide.groupby("device", sort=False):
        g = g.set_index("time")
        g[["lat", "lon", "alt"]] = g[["lat", "lon", "alt"]].ffill(limit=fill_limit_s)
        g["ignition"] = g["ignition"].ffill()
        filled.append(g.reset_index())
    wide = pd.concat(filled, ignore_index=True).loc[:, list(WIDE_COLUMNS)]
    rep.rows_used = len(wide)
    rep.devices = int(wide["device"].nunique())
    return wide, rep


def read_long_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)
