"""Metrics for event counts with exposure. No accuracy, no MAE on labels, no resampled test rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance


def _area(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def poisson_deviance(counts, expected) -> float:
    return float(mean_poisson_deviance(np.asarray(counts, dtype=float), np.clip(np.asarray(expected, dtype=float), 1e-9, None)))


def d2_vs_reference(counts, expected, reference_expected) -> float:
    """Share of the reference deviance that the model removes (1 = perfect, 0 = same as reference)."""
    ref = poisson_deviance(counts, reference_expected)
    return float(1 - poisson_deviance(counts, expected) / ref) if ref > 0 else 0.0


def gini(counts, pred_rate, exposure) -> float:
    """Exposure-weighted Gini of the ranking by predicted rate (the insurance Lorenz-curve Gini).

    Trips with the same predicted rate form one step of the curve, so a constant
    prediction has a Gini of 0 whatever the row order is.
    """
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    df = pd.DataFrame({"p": np.asarray(pred_rate, dtype=float), "e": exposure, "c": counts})
    steps = df.groupby("p", sort=True)[["e", "c"]].sum()
    cum_exp = np.r_[0, np.cumsum(steps["e"].to_numpy()) / exposure.sum()]
    cum_evt = np.r_[0, np.cumsum(steps["c"].to_numpy()) / max(counts.sum(), 1e-12)]
    return float(1 - 2 * _area(cum_exp, cum_evt))


def normalised_gini(counts, pred_rate, exposure) -> float:
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    best = gini(counts, counts / exposure, exposure)
    return float(gini(counts, pred_rate, exposure) / best) if best > 0 else 0.0


def calibration_by_decile(counts, expected, n_bins: int = 10) -> pd.DataFrame:
    """Observed and expected events in bins of the expected count (equal number of trips)."""
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ranks = pd.Series(expected).rank(method="first").to_numpy()
    bins = np.minimum((ranks - 1) * n_bins // len(expected), n_bins - 1).astype(int)
    df = pd.DataFrame({"bin": bins, "observed": counts, "expected": expected})
    out = df.groupby("bin").agg(trips=("observed", "size"), observed=("observed", "sum"), expected=("expected", "sum"))
    out["ratio"] = out["observed"] / out["expected"].clip(lower=1e-9)
    return out.reset_index()


def rate_metrics(counts, pred_rate, exposure, reference_rate: float) -> dict:
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    pred_rate = np.asarray(pred_rate, dtype=float)
    expected = pred_rate * exposure
    return {
        "trips": int(counts.size),
        "events": float(counts.sum()),
        "km": float(exposure.sum()),
        "poisson_deviance": poisson_deviance(counts, expected),
        "reference_deviance": poisson_deviance(counts, reference_rate * exposure),
        "d2": d2_vs_reference(counts, expected, reference_rate * exposure),
        "gini": gini(counts, pred_rate, exposure),
        "normalised_gini": normalised_gini(counts, pred_rate, exposure),
        "expected_over_observed": float(expected.sum() / max(counts.sum(), 1e-12)),
    }


def driver_table(trips: pd.DataFrame, pred_rate, group: str = "device") -> pd.DataFrame:
    """One row for each driver: km, trips, observed and predicted events per 100 km, risk index."""
    df = pd.DataFrame(
        {
            group: trips[group].to_numpy(),
            "km": trips["distance_km"].to_numpy(dtype=float),
            "expected": np.asarray(pred_rate, dtype=float) * trips["distance_km"].to_numpy(dtype=float),
        }
    )
    if "harsh_events" in trips:
        df["events"] = trips["harsh_events"].to_numpy(dtype=float)
    agg = {"km": ("km", "sum"), "trips": ("km", "size"), "expected": ("expected", "sum")}
    if "events" in df:
        agg["events"] = ("events", "sum")
    out = df.groupby(group).agg(**agg)
    out["predicted_per_100km"] = 100 * out["expected"] / out["km"]
    if "events" in out:
        out["observed_per_100km"] = 100 * out["events"] / out["km"]
    fleet = 100 * out["expected"].sum() / out["km"].sum()
    out["risk_index"] = 100 * out["predicted_per_100km"] / fleet
    return out.drop(columns=["expected"]).sort_values("risk_index", ascending=False).reset_index()
