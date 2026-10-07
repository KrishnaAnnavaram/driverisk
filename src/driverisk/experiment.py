"""Grouped validation, training, scoring and permutation importance.

Protocol of `train`:
1. Hold out 25 % of the devices (GroupShuffleSplit). No device has trips on both sides.
2. On the other devices, run GroupKFold cross-validation for each model.
3. Keep the model with the lowest mean CV Poisson deviance.
4. Fit it on all training devices. Score the held-out devices one time.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from driverisk import __version__
from driverisk.evaluate import calibration_by_decile, driver_table, poisson_deviance, rate_metrics
from driverisk.features import EXPOSURE, FEATURES, GROUP, TARGET, assert_no_label_inputs
from driverisk.models import MODEL_NAMES, make_model

ARTIFACT = "model.joblib"


def _xy(trips: pd.DataFrame):
    assert_no_label_inputs(FEATURES)
    X = trips.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return X, trips[TARGET].to_numpy(dtype=float), trips[EXPOSURE].to_numpy(dtype=float), trips[GROUP].to_numpy()


def holdout_split(trips: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(trips, groups=trips[GROUP]))
    train, test = trips.iloc[tr].reset_index(drop=True), trips.iloc[te].reset_index(drop=True)
    if set(train[GROUP]) & set(test[GROUP]):
        raise AssertionError("a device is in both splits")
    return train, test


def grouped_cv(trips: pd.DataFrame, model: str, folds: int = 5, seed: int = 42) -> dict:
    X, y, e, g = _xy(trips)
    n_groups = len(np.unique(g))
    k = min(folds, n_groups)
    if k < 2:
        raise ValueError("grouped CV needs at least 2 devices")
    rows = []
    for tr, va in GroupKFold(n_splits=k).split(X, y, g):
        m = make_model(model, seed).fit(X[tr], y[tr], e[tr])
        ref_rate = y[tr].sum() / e[tr].sum()
        met = rate_metrics(y[va], m.predict_rate(X[va]), e[va], ref_rate)
        rows.append(met)
    dev = np.array([r["poisson_deviance"] for r in rows])
    return {
        "model": model,
        "folds": k,
        "poisson_deviance_mean": float(dev.mean()),
        "poisson_deviance_std": float(dev.std(ddof=1)) if dev.size > 1 else 0.0,
        "d2_mean": float(np.mean([r["d2"] for r in rows])),
        "normalised_gini_mean": float(np.mean([r["normalised_gini"] for r in rows])),
    }


def permutation_importance(model, trips: pd.DataFrame, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Increase of the Poisson deviance when one feature is shuffled across trips."""
    X, y, e, _ = _xy(trips)
    base = poisson_deviance(y, model.predict_count(X, e))
    rng = np.random.default_rng(seed)
    rows = []
    for j, name in enumerate(FEATURES):
        if np.isnan(X[:, j]).all():
            continue
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            incs.append(poisson_deviance(y, model.predict_count(Xp, e)) - base)
        rows.append({"feature": name, "deviance_increase": float(np.mean(incs)), "std": float(np.std(incs))})
    return pd.DataFrame(rows).sort_values("deviance_increase", ascending=False).reset_index(drop=True)


def train(trips: pd.DataFrame, models=("baseline", "glm", "hgb"), seed: int = 42, folds: int = 5) -> tuple[dict, dict]:
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad:
        raise ValueError(f"unknown model(s) {bad}")
    if trips[GROUP].nunique() < 4:
        raise ValueError("training needs at least 4 devices")
    train_part, test_part = holdout_split(trips, seed=seed)
    cv = {m: grouped_cv(train_part, m, folds, seed) for m in models}
    chosen = min(cv, key=lambda m: (cv[m]["poisson_deviance_mean"], m))
    X, y, e, _ = _xy(train_part)
    model = make_model(chosen, seed).fit(X, y, e)
    Xt, yt, et, _ = _xy(test_part)
    rate = model.predict_rate(Xt)
    test = rate_metrics(yt, rate, et, model.fleet_rate_)
    drivers = driver_table(test_part, rate)
    pred_rank, obs_rank = drivers["predicted_per_100km"].rank(), drivers["observed_per_100km"].rank()
    test["driver_spearman"] = float(pred_rank.corr(obs_rank)) if pred_rank.nunique() > 1 else float("nan")
    test["drivers"] = int(len(drivers))
    importance = permutation_importance(model, test_part, seed=seed)
    summary = {
        "version": __version__,
        "seed": seed,
        "trips": int(len(trips)),
        "devices": int(trips[GROUP].nunique()),
        "train_devices": int(train_part[GROUP].nunique()),
        "test_devices": int(test_part[GROUP].nunique()),
        "fleet_rate_per_100km": 100 * model.fleet_rate_,
        "cv": cv,
        "chosen_model": chosen,
        "chosen_by": "lowest mean grouped-CV Poisson deviance",
        "test": test,
        "calibration": calibration_by_decile(yt, rate * et).to_dict(orient="records"),
        "importance": importance.head(10).to_dict(orient="records"),
        "features": list(FEATURES),
    }
    bundle = {"model": model, "features": list(FEATURES), "summary": summary}
    return summary, bundle


def score_drivers(bundle: dict, trips: pd.DataFrame) -> pd.DataFrame:
    X = trips.loc[:, bundle["features"]].to_numpy(dtype=float)
    return driver_table(trips, bundle["model"].predict_rate(X))


def save_run(out_dir, summary: dict, bundle: dict, card: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / ARTIFACT)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "model_card.md").write_text(card, encoding="utf-8")
    return out


def load_run(run_dir) -> dict:
    path = Path(run_dir) / ARTIFACT
    if not path.is_file():
        raise FileNotFoundError(f"no {ARTIFACT} in {run_dir}. Run 'driverisk train' first")
    return joblib.load(path)
