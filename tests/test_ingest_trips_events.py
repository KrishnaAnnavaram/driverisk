"""Reference problems 2 and 4: mixed `value` column, broken POSITION parser."""

import numpy as np
import pandas as pd
import pytest

from conftest import long_rows
from driverisk.events import EventRules, count_runs, trip_events
from driverisk.ingest import G, TelematicsError, normalise_variable, parse_ignition, parse_position, pivot_wide
from driverisk.trips import add_kinematics, haversine_m, segment_trips


def test_parse_position_reads_lat_lon_alt():
    assert parse_position("13.3409,74.7421,12.5") == (13.3409, 74.7421, 12.5)
    lat, lon, alt = parse_position(" 52.2, 0.12 ")
    assert (lat, lon) == (52.2, 0.12) and np.isnan(alt)
    for bad in ("abc", "1,2,3,4", "95,10", "0,0,0", "", None, float("nan")):
        assert np.isnan(parse_position(bad)[0])


def test_variable_names_and_ignition():
    assert normalise_variable("Vehicle speed") == "VEHICLESPEED"
    assert normalise_variable("ACCELERATION X") == "ACCELERATIONX"
    assert parse_ignition("ON") == 1.0 and parse_ignition("0") == 0.0 and np.isnan(parse_ignition("?"))


def test_pivot_gives_each_signal_its_own_column():
    rows = [
        ("A", "2024-01-01T10:00:00Z", "Vehicle speed", "50"),
        ("A", "2024-01-01T10:00:00Z", "ENGINE RPM", "2100"),
        ("A", "2024-01-01T10:00:00Z", "POSITION", "52.1,0.1,20"),
        ("A", "2024-01-01T10:00:01Z", "Vehicle speed", "52"),
        ("A", "2024-01-01T10:00:01Z", "TOWING", "0"),
        ("A", "2024-01-01T10:00:01Z", "ACCELERATION X", "0.1"),
    ]
    wide, rep = pivot_wide(long_rows(rows))
    assert len(wide) == 2
    first = wide.iloc[0]
    assert first["speed_kmh"] == 50 and first["rpm"] == 2100 and first["lat"] == 52.1
    # the RPM reading never becomes a speed value
    assert wide["speed_kmh"].max() == 52
    # the position stays valid for the next second
    assert wide.iloc[1]["lat"] == 52.1
    assert rep.unknown_variables == {"TOWING": 1}


def test_pivot_converts_g_and_refuses_bad_tables():
    rows = [("A", "2024-01-01T10:00:00Z", "ACCELERATION X", "0.5")]
    wide, _ = pivot_wide(long_rows(rows), acc_unit="g")
    assert wide["acc_x"].iloc[0] == pytest.approx(0.5 * G)
    with pytest.raises(TelematicsError):
        pivot_wide(long_rows(rows).drop(columns=["variable"]))
    with pytest.raises(TelematicsError):
        pivot_wide(long_rows([("A", "2024-01-01T10:00:00Z", "TOWING", "1")]))


def test_bad_values_are_counted():
    rows = [
        ("A", "2024-01-01T10:00:00Z", "Vehicle speed", "fast"),
        ("A", "not a time", "Vehicle speed", "10"),
        ("A", "2024-01-01T10:00:02Z", "POSITION", "13.3,74.7"),
        ("A", "2024-01-01T10:00:03Z", "POSITION", "nowhere"),
    ]
    _, rep = pivot_wide(long_rows(rows))
    assert rep.bad_numbers == 1 and rep.bad_timestamps == 1 and rep.bad_positions == 1


def _wide(times, speed, ign=None, device="A"):
    n = len(times)
    return pd.DataFrame(
        {
            "device": device,
            "time": pd.to_datetime(times, utc=True),
            "speed_kmh": speed,
            "rpm": np.nan,
            "acc_x": np.nan,
            "acc_y": np.nan,
            "acc_z": np.nan,
            "lat": np.nan,
            "lon": np.nan,
            "alt": np.nan,
            "ignition": ign if ign is not None else [np.nan] * n,
        }
    )


def test_trips_split_on_gap_and_on_ignition():
    t = ["2024-01-01T10:00:00", "2024-01-01T10:00:01", "2024-01-01T10:30:00", "2024-01-01T10:30:01",
         "2024-01-01T10:30:02", "2024-01-01T10:30:03"]
    w = _wide(t, [10, 12, 0, 5, 0, 6], ign=[1, 1, 1, 1, 0, 1])
    trips = segment_trips(w)
    assert list(trips["trip_id"].unique()) == ["A-0000", "A-0001", "A-0002"]
    assert len(trips) == 5  # the ignition-off row is dropped


def test_acceleration_is_derived_from_speed_when_missing():
    t = ["2024-01-01T10:00:00", "2024-01-01T10:00:01", "2024-01-01T10:00:02", "2024-01-01T10:00:10"]
    w = _wide(t, [36.0, 46.8, 36.0, 0.0])
    pts = add_kinematics(segment_trips(w))
    acc = pts["acc_long"].to_numpy()
    assert np.isnan(acc[0])
    assert acc[1] == pytest.approx(3.0) and acc[2] == pytest.approx(-3.0)
    assert np.isnan(acc[3])  # 8 s apart: too far for a derivative
    assert pts["dist_m"].iloc[1] == pytest.approx((10 + 13) / 2)


def test_haversine_one_degree_of_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)


def test_event_runs_are_merged():
    flag = np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 1])
    t = np.arange(10, dtype=float)
    assert count_runs(flag, t, merge_s=3.0) == 2
    assert count_runs(flag, t, merge_s=1.0) == 3
    assert count_runs(np.zeros(5), np.arange(5.0), 3.0) == 0


def test_trip_events_thresholds():
    trip = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=12, freq="s", tz="UTC"),
            "acc_long": [0, -3.5, -3.1, 0, 0, 0, 0, 2.6, 0, 0, 0, -2.9],
            "acc_y": [0, 0, 0, 0, 3.4, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    ev = trip_events(trip, EventRules())
    assert ev == {"harsh_brake": 1, "harsh_accel": 1, "harsh_corner": 1, "harsh_events": 3}
