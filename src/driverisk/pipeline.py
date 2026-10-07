"""From the long telematics stream to the trip table: ingest, trips, kinematics, context, features."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from driverisk.context.collisions import CollisionPrior
from driverisk.context.roads import OfflineRoads, RoadSegmentsCsv, attach_roads
from driverisk.context.weather import CsvWeather, OfflineWeather, OpenMeteoWeather, attach_weather
from driverisk.features import TripRules, build_trip_table
from driverisk.ingest import IngestReport, pivot_wide
from driverisk.trips import add_kinematics, segment_trips

MIN_PRIOR_COVERAGE = 0.5


@dataclass
class ContextReport:
    weather_source: str = ""
    road_source: str = ""
    points: int = 0
    with_position: float = 0.0
    with_weather: float = 0.0
    with_road_class: float = 0.0
    collision_coverage: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        cov = "not used" if self.collision_coverage is None else f"{self.collision_coverage:.2f}"
        lines = [
            f"points={self.points} position={self.with_position:.3f} weather({self.weather_source})="
            f"{self.with_weather:.3f} road_class({self.road_source})={self.with_road_class:.3f} collision_coverage={cov}"
        ]
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def make_providers(settings):
    """Weather and road providers from the settings."""
    if settings.weather == "csv":
        weather = CsvWeather(settings.weather_csv)
    elif settings.weather == "open-meteo":
        weather = OpenMeteoWeather(settings.cache_dir, settings.user_agent)
    else:
        weather = OfflineWeather(settings.seed)
    roads = RoadSegmentsCsv(settings.roads_csv) if settings.roads == "csv" else OfflineRoads(settings.seed)
    collisions = CollisionPrior.from_csv(settings.collisions_csv) if settings.collisions_csv else None
    return weather, roads, collisions


def build_points(long_df: pd.DataFrame, weather, roads, collisions: CollisionPrior | None = None,
                 acc_unit: str = "ms2", max_gap_s: float = 600.0) -> tuple[pd.DataFrame, IngestReport, ContextReport]:
    wide, ingest = pivot_wide(long_df, acc_unit=acc_unit)
    points = add_kinematics(segment_trips(wide, max_gap_s=max_gap_s))
    points = attach_weather(points, weather)
    points = attach_roads(points, roads)
    ctx = ContextReport(weather_source=weather.name, road_source=roads.name, points=len(points))
    if len(points):
        ctx.with_position = float(points["lat"].notna().mean())
        ctx.with_weather = float(points["precip_mm"].notna().mean())
        ctx.with_road_class = float((points["road_class"] != "unknown").mean())
    if collisions is not None:
        ctx.collision_coverage = collisions.coverage(points["lat"], points["lon"])
        if ctx.collision_coverage >= MIN_PRIOR_COVERAGE:
            points["collision_density"] = collisions.density(points["lat"], points["lon"])
        else:
            ctx.notes.append(
                f"collision prior covers only {ctx.collision_coverage:.0%} of the readings. It is not used"
            )
    if weather.name == "offline" or roads.name == "offline":
        ctx.notes.append("offline context is synthetic. Use it for the demo and the tests only")
    return points, ingest, ctx


def build_trips(long_df: pd.DataFrame, weather, roads, collisions=None, acc_unit: str = "ms2",
                rules: TripRules = TripRules()) -> tuple[pd.DataFrame, IngestReport, ContextReport]:
    points, ingest, ctx = build_points(long_df, weather, roads, collisions, acc_unit)
    return build_trip_table(points, rules), ingest, ctx
