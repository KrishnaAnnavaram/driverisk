"""Reference problems 3, 4 and 5: weather from the wrong place, per-row geocoding, unused casualty data."""

import json

import numpy as np
import pandas as pd
import pytest

from driverisk.context import (
    CollisionPrior,
    CsvWeather,
    OfflineRoads,
    OfflineWeather,
    OpenMeteoWeather,
    RoadSegmentsCsv,
    attach_roads,
    attach_weather,
)
from driverisk.pipeline import build_points


def _points(lats, lons, times):
    return pd.DataFrame({"lat": lats, "lon": lons, "time": pd.to_datetime(times, utc=True)})


class CountingWeather:
    name = "counting"

    def __init__(self):
        self.calls = []

    def hourly(self, lat, lon, start, end):
        self.calls.append((round(lat, 1), round(lon, 1)))
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame({"temp_c": lat, "precip_mm": lon, "wind_ms": 1.0}, index=idx)


def test_weather_join_uses_place_and_hour():
    pts = _points([52.0, 52.0, 48.0, np.nan], [0.1, 0.1, 2.3, np.nan],
                  ["2024-01-01T10:05", "2024-01-01T10:55", "2024-01-01T10:05", "2024-01-01T10:05"])
    w = CountingWeather()
    out = attach_weather(pts, w)
    assert out["temp_c"].iloc[0] == pytest.approx(52.0) and out["temp_c"].iloc[2] == pytest.approx(48.0)
    assert np.isnan(out["temp_c"].iloc[3])  # no position, no weather: never an hour-only join
    assert sorted(w.calls) == [(48.0, 2.3), (52.0, 0.1)]  # one call for each cell, not for each row


def test_offline_weather_is_deterministic_and_local():
    w = OfflineWeather(seed=1)
    a = w.hourly(52.0, 0.1, pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-03 23:00", tz="UTC"))
    b = OfflineWeather(seed=1).hourly(52.0, 0.1, pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-03 23:00", tz="UTC"))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 72 and set(a.columns) == {"temp_c", "precip_mm", "wind_ms"}
    c = w.hourly(40.0, 10.0, pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-03 23:00", tz="UTC"))
    assert not a.equals(c)


def test_csv_weather_refuses_a_distant_station(tmp_path):
    path = tmp_path / "w.csv"
    pd.DataFrame(
        {
            "station_id": ["jena", "jena"],
            "lat": [50.93, 50.93],
            "lon": [11.59, 11.59],
            "time": ["2024-01-01T10:00Z", "2024-01-01T11:00Z"],
            "temp_c": [3.0, 4.0],
            "precip_mm": [0.0, 1.0],
            "wind_ms": [2.0, 3.0],
        }
    ).to_csv(path, index=False)
    w = CsvWeather(str(path), max_km=50)
    near = attach_weather(_points([50.9], [11.6], ["2024-01-01T11:20"]), w)
    assert near["precip_mm"].iloc[0] == 1.0
    far = attach_weather(_points([13.34], [74.74], ["2024-01-01T11:20"]), w)
    assert np.isnan(far["precip_mm"].iloc[0])
    with pytest.raises(ValueError):
        pd.DataFrame({"lat": [1]}).to_csv(tmp_path / "bad.csv", index=False)
        CsvWeather(str(tmp_path / "bad.csv"))


def test_open_meteo_parses_and_caches_without_network(tmp_path, monkeypatch):
    w = OpenMeteoWeather(cache_dir=str(tmp_path), user_agent="test")
    calls = []

    def fake(url):
        calls.append(url)
        return {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"], "temperature_2m": [1.0, 2.0],
                           "precipitation": [0.0, 0.4], "wind_speed_10m": [3.0, 4.0]}}

    monkeypatch.setattr(w, "_fetch_json", fake)
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01 01:00", tz="UTC")
    a = w.hourly(52.2, 0.1, start, end)
    b = w.hourly(52.2, 0.1, start, end)
    assert len(calls) == 1 and "latitude=52.20" in calls[0] and "wind_speed_unit=ms" in calls[0]
    assert a["precip_mm"].tolist() == [0.0, 0.4] and a.equals(b)
    assert len(list(tmp_path.glob("openmeteo_*.json"))) == 1
    json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))


def test_road_segments_nearest_point_within_limit():
    segs = pd.DataFrame({"lat": [52.0, 52.001], "lon": [0.1, 0.1], "road_class": ["primary", "residential"],
                         "speed_limit_kmh": [80, 30]})
    roads = RoadSegmentsCsv(frame=segs, max_m=50)
    cls, lim = roads.lookup([52.0001, 52.0009, 52.01, np.nan], [0.1, 0.1, 0.1, np.nan])
    assert list(cls) == ["primary", "residential", "unknown", "unknown"]
    assert lim[0] == 80 and np.isnan(lim[2])
    out = attach_roads(_points([52.0001], [0.1], ["2024-01-01"]), roads)
    assert out["road_class"].iloc[0] == "primary"
    with pytest.raises(ValueError):
        RoadSegmentsCsv(frame=segs.drop(columns=["road_class"]))


def test_offline_roads_are_stable():
    r = OfflineRoads(seed=3)
    a = r.lookup([52.0, 52.5], [0.1, 0.4])
    b = OfflineRoads(seed=3).lookup([52.0, 52.5], [0.1, 0.4])
    assert list(a[0]) == list(b[0]) and np.allclose(a[1], b[1])


def test_casualty_table_is_refused_and_collision_table_works(tmp_path):
    casualty = tmp_path / "casualty.csv"
    pd.DataFrame({"collision_index": [1], "casualty_severity": [3]}).to_csv(casualty, index=False)
    with pytest.raises(ValueError, match="casualty table has no location"):
        CollisionPrior.from_csv(str(casualty))
    coll = tmp_path / "collision.csv"
    pd.DataFrame({"latitude": [52.0, 52.0, 52.001, 51.5], "longitude": [0.1, 0.1, 0.1, -0.1]}).to_csv(coll, index=False)
    prior = CollisionPrior.from_csv(str(coll))
    assert prior.density([52.0, 60.0], [0.1, 5.0]).tolist() == [3.0, 0.0]
    assert prior.coverage([52.0, 13.3], [0.0, 74.7]) == pytest.approx(0.5)


def test_prior_from_another_region_is_not_used(fleet_long, tmp_path):
    coll = tmp_path / "india.csv"
    pd.DataFrame({"latitude": [13.3, 13.4], "longitude": [74.7, 74.8]}).to_csv(coll, index=False)
    prior = CollisionPrior.from_csv(str(coll))
    points, _, ctx = build_points(fleet_long, OfflineWeather(5), OfflineRoads(5), prior)
    assert ctx.collision_coverage == 0.0
    assert "collision_density" not in points.columns
    assert any("not used" in n for n in ctx.notes)
