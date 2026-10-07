"""Harsh driving events: the target of the risk model.

The target counts events in the acceleration signals. The model features never use those
signals (see `LABEL_INPUTS` and `features.assert_no_label_inputs`), so the model cannot learn
the event rule back from its own inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LABEL_INPUTS = ("acc_x", "acc_y", "acc_z", "acc_long")


@dataclass(frozen=True)
class EventRules:
    harsh_brake_ms2: float = -3.0
    harsh_accel_ms2: float = 2.5
    harsh_corner_ms2: float = 3.0
    merge_s: float = 3.0


def count_runs(flag: np.ndarray, times_s: np.ndarray, merge_s: float) -> int:
    """Count events: flagged readings closer than `merge_s` seconds belong to one event."""
    t = times_s[np.asarray(flag, dtype=bool)]
    if t.size == 0:
        return 0
    return int(1 + np.sum(np.diff(np.sort(t)) > merge_s))


def trip_events(trip: pd.DataFrame, rules: EventRules = EventRules()) -> dict[str, int]:
    t = (trip["time"] - trip["time"].iloc[0]).dt.total_seconds().to_numpy()
    acc = trip["acc_long"].to_numpy(dtype=float)
    lat = trip["acc_y"].to_numpy(dtype=float)
    brake = count_runs(np.nan_to_num(acc, nan=0.0) <= rules.harsh_brake_ms2, t, rules.merge_s)
    accel = count_runs(np.nan_to_num(acc, nan=0.0) >= rules.harsh_accel_ms2, t, rules.merge_s)
    corner = count_runs(np.abs(np.nan_to_num(lat, nan=0.0)) >= rules.harsh_corner_ms2, t, rules.merge_s)
    return {
        "harsh_brake": brake,
        "harsh_accel": accel,
        "harsh_corner": corner,
        "harsh_events": brake + accel + corner,
    }
