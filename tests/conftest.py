import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from driverisk.context import OfflineRoads, OfflineWeather  # noqa: E402
from driverisk.pipeline import build_points  # noqa: E402
from driverisk.features import build_trip_table  # noqa: E402
from driverisk.synthetic import simulate_fleet  # noqa: E402


def long_rows(rows):
    """rows: (device, iso time, variable, value) tuples -> long frame of text."""
    return pd.DataFrame(rows, columns=["deviceId", "timestamp", "variable", "value"]).astype(str)


@pytest.fixture(scope="session")
def fleet_long():
    return simulate_fleet(n_drivers=8, trips_per_driver=4, seed=5, min_trip_min=3, max_trip_min=6)


@pytest.fixture(scope="session")
def fleet_points(fleet_long):
    points, ingest, ctx = build_points(fleet_long, OfflineWeather(5), OfflineRoads(5))
    return points, ingest, ctx


@pytest.fixture(scope="session")
def fleet_trips(fleet_points):
    return build_trip_table(fleet_points[0])


@pytest.fixture(scope="session")
def big_trips():
    long = simulate_fleet(n_drivers=30, trips_per_driver=8, seed=12, min_trip_min=3, max_trip_min=8)
    points, _, _ = build_points(long, OfflineWeather(12), OfflineRoads(12))
    return build_trip_table(points)
