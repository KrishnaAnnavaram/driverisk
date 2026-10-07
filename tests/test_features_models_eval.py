"""Reference problems 1, 6 and 8: label rule on the features, leaky split, metric misuse."""

import numpy as np
import pandas as pd
import pytest

from driverisk.evaluate import (
    calibration_by_decile,
    d2_vs_reference,
    driver_table,
    gini,
    normalised_gini,
    poisson_deviance,
    rate_metrics,
)
from driverisk.experiment import grouped_cv, holdout_split, permutation_importance, train
from driverisk.features import FEATURES, LabelInFeatures, assert_no_label_inputs, build_trip_table
from driverisk.models import RateModel, make_model


def test_no_feature_reads_the_target_signals():
    assert_no_label_inputs(FEATURES)
    with pytest.raises(LabelInFeatures):
        assert_no_label_inputs(["mean_speed_kmh", "p95_acc_long"])
    with pytest.raises(LabelInFeatures):
        assert_no_label_inputs(["harsh_brake"])


def test_changing_acceleration_changes_the_target_but_not_the_features(fleet_points):
    points = fleet_points[0]
    a = build_trip_table(points)
    shaken = points.copy()
    shaken["acc_long"] = shaken["acc_long"] * 2.0
    b = build_trip_table(shaken)
    pd.testing.assert_frame_equal(a.loc[:, list(FEATURES)], b.loc[:, list(FEATURES)])
    assert b["harsh_events"].sum() > a["harsh_events"].sum()


def test_trip_table_shape(fleet_trips, fleet_points):
    _, ingest, ctx = fleet_points
    assert ingest.devices == 8 and ctx.with_weather == 1.0
    assert set(FEATURES) <= set(fleet_trips.columns)
    assert (fleet_trips["distance_km"] >= 0.5).all()
    assert fleet_trips["trip_id"].is_unique
    assert fleet_trips["harsh_events"].ge(0).all()


def test_rate_model_baseline_and_exposure():
    X = np.zeros((4, 2))
    m = RateModel("baseline").fit(X, [1, 2, 3, 4], [1, 1, 2, 6])
    assert m.fleet_rate_ == pytest.approx(1.0)
    assert m.predict_count(X, [2, 2, 2, 2]).tolist() == [2.0] * 4
    with pytest.raises(ValueError):
        RateModel("baseline").fit(X, [1, 1, 1, 1], [1, 0, 1, 1])
    with pytest.raises(ValueError):
        make_model("svm")


def test_glm_recovers_a_rate_effect():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    exposure = rng.uniform(1, 20, n)
    counts = rng.poisson(np.exp(-1 + 0.7 * x) * exposure)
    m = RateModel("glm", alpha=1e-4).fit(x.reshape(-1, 1), counts, exposure)
    coef = m.model_.named_steps["poisson"].coef_[0] / m.model_.named_steps["scale"].scale_[0]
    assert coef == pytest.approx(0.7, abs=0.05)
    hgb = RateModel("hgb").fit(x.reshape(-1, 1), counts, exposure)
    ranks = pd.Series(hgb.predict_rate(x.reshape(-1, 1))).rank().corr(pd.Series(x).rank())
    assert ranks > 0.9


def test_metrics_on_known_values():
    y = np.array([0, 1, 2, 5])
    e = np.array([1.0, 1.0, 1.0, 1.0])
    assert poisson_deviance(y, y + 1e-12) == pytest.approx(0, abs=1e-6)
    assert gini(y, np.ones(4), e) == pytest.approx(0)
    assert normalised_gini(y, y / e, e) == pytest.approx(1)
    assert normalised_gini(y, -y, e) < 0
    assert d2_vs_reference(y, y + 1e-12, np.full(4, 2.0)) == pytest.approx(1, abs=1e-6)
    m = rate_metrics(y, np.full(4, 2.0), e, 2.0)
    assert m["d2"] == pytest.approx(0) and "accuracy" not in m and "mae" not in m
    cal = calibration_by_decile(np.arange(20), np.arange(20) + 0.5, n_bins=5)
    assert len(cal) == 5 and cal["trips"].sum() == 20 and cal["observed"].sum() == 190


def test_driver_table_index():
    trips = pd.DataFrame({"device": ["a", "a", "b"], "distance_km": [10.0, 10.0, 20.0], "harsh_events": [1, 1, 8]})
    t = driver_table(trips, [0.1, 0.1, 0.4])
    assert t.iloc[0]["device"] == "b"
    assert t.set_index("device").loc["a", "predicted_per_100km"] == pytest.approx(10.0)
    assert t.set_index("device").loc["b", "observed_per_100km"] == pytest.approx(40.0)
    fleet = 100 * (2 + 8) / 40
    assert t.set_index("device").loc["b", "risk_index"] == pytest.approx(100 * 40 / fleet)


def test_holdout_keeps_each_device_on_one_side(big_trips):
    train_part, test_part = holdout_split(big_trips, seed=1)
    assert not set(train_part["device"]) & set(test_part["device"])
    assert len(train_part) + len(test_part) == len(big_trips)


def test_grouped_cv_and_train(big_trips):
    cv = grouped_cv(big_trips, "glm", folds=4, seed=0)
    assert cv["folds"] == 4 and cv["poisson_deviance_mean"] > 0
    summary, bundle = train(big_trips, models=("baseline", "glm"), seed=0, folds=4)
    chosen = summary["chosen_model"]
    assert summary["cv"][chosen]["poisson_deviance_mean"] == min(v["poisson_deviance_mean"] for v in summary["cv"].values())
    test = summary["test"]
    # every held-out trip is scored, nothing is resampled
    assert test["trips"] == len(holdout_split(big_trips, seed=0)[1])
    assert summary["test_devices"] + summary["train_devices"] == summary["devices"] == 30
    assert summary["cv"]["glm"]["d2_mean"] > 0  # the synthetic driver style is learnable
    assert 0.8 < test["expected_over_observed"] < 1.25


def test_permutation_importance(big_trips):
    train_part, test_part = holdout_split(big_trips, seed=0)
    from driverisk.features import FEATURES as F

    X = train_part.loc[:, list(F)].to_numpy(dtype=float)
    m = RateModel("glm").fit(X, train_part["harsh_events"], train_part["distance_km"])
    imp = permutation_importance(m, test_part, n_repeats=2)
    assert set(imp.columns) == {"feature", "deviance_increase", "std"}
    assert imp["deviance_increase"].iloc[0] >= imp["deviance_increase"].iloc[-1]
    with pytest.raises(ValueError):
        train(big_trips[big_trips["device"].isin(["DEV000", "DEV001"])])
