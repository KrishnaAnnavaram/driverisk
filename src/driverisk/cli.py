"""Command line: simulate | validate | trips | train | score | explain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from driverisk.config import Settings, load_env_file
from driverisk.experiment import load_run, permutation_importance, save_run, score_drivers, train
from driverisk.features import FEATURES
from driverisk.ingest import TelematicsError, pivot_wide, read_long_csv
from driverisk.models import MODEL_NAMES
from driverisk.pipeline import build_trips, make_providers
from driverisk.report import model_card
from driverisk.synthetic import simulate_fleet


def _models(text: str) -> list[str]:
    names = [m.strip() for m in text.split(",") if m.strip()]
    if not names or any(m not in MODEL_NAMES for m in names):
        raise argparse.ArgumentTypeError(f"choose from {MODEL_NAMES}")
    return names


def _data_args(p):
    p.add_argument("--data", help="long telematics CSV. Default: DRIVERISK_DATA, else a synthetic fleet")
    p.add_argument("--trip-table", help="a trip table CSV from 'driverisk trips' (skips the ingest)")
    p.add_argument("--drivers", type=int, default=40, help="synthetic drivers when no CSV is given")
    p.add_argument("--trips", type=int, default=12, help="synthetic trips for each driver")
    p.add_argument("--seed", type=int, help="seed (default DRIVERISK_SEED or 42)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="driverisk", description=__doc__)
    ap.add_argument("--env-file", default=".env")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="write a synthetic long-format telematics CSV")
    p.add_argument("--drivers", type=int, default=40)
    p.add_argument("--trips", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic_telematics.csv")

    p = sub.add_parser("validate", help="check a long telematics CSV and print the ingest report")
    p.add_argument("csv")

    p = sub.add_parser("trips", help="build the trip table (features, exposure, target)")
    _data_args(p)
    p.add_argument("--out", default="data/trips.csv")

    p = sub.add_parser("train", help="grouped CV, pick a model, score held-out devices once")
    _data_args(p)
    p.add_argument("--models", type=_models, default=list(MODEL_NAMES))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", help="run folder (default <DRIVERISK_OUTPUT>/latest)")

    p = sub.add_parser("score", help="risk index for each driver")
    _data_args(p)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="-", help="CSV path, '-' for standard output")

    p = sub.add_parser("explain", help="permutation importance on a trip table")
    _data_args(p)
    p.add_argument("--run", required=True)
    return ap


def _trip_table(args, settings: Settings):
    seed = settings.seed if args.seed is None else args.seed
    if args.trip_table:
        table = pd.read_csv(args.trip_table, parse_dates=["start"])
        return table, args.trip_table, "trip table from CSV", seed
    path = args.data or settings.data_path
    if path:
        long_df, source = read_long_csv(path), path
    else:
        long_df = simulate_fleet(args.drivers, args.trips, seed=seed)
        source = f"synthetic(drivers={args.drivers}, trips={args.trips}, seed={seed})"
    weather, roads, collisions = make_providers(settings)
    table, ingest, ctx = build_trips(long_df, weather, roads, collisions, settings.acc_unit)
    print(f"data: {source}\ningest: {ingest.summary()}\ncontext: {ctx.summary()}\ntrips: {len(table)}", file=sys.stderr)
    if table.empty:
        raise TelematicsError("no trip passed the minimum distance and duration")
    return table, source, ctx.summary(), seed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    try:
        settings = Settings.from_env()
        with threadpool_limits(limits=settings.threads):
            return _run(args, settings)
    except (TelematicsError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args, settings: Settings) -> int:
    if args.cmd == "simulate":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = simulate_fleet(args.drivers, args.trips, seed=args.seed)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} readings for {args.drivers} devices to {out}")
        return 0

    if args.cmd == "validate":
        _, rep = pivot_wide(read_long_csv(args.csv), acc_unit=settings.acc_unit)
        print(rep.summary())
        return 0

    table, source, ctx, seed = _trip_table(args, settings)

    if args.cmd == "trips":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"wrote {len(table)} trips to {out}")
        return 0

    if args.cmd == "train":
        summary, bundle = train(table, args.models, seed, args.folds)
        out = save_run(args.out or Path(settings.output_dir) / "latest", summary, bundle, model_card(summary, source, ctx))
        for name, cv in summary["cv"].items():
            print(f"  cv {name:8s} deviance={cv['poisson_deviance_mean']:.4f} d2={cv['d2_mean']:.3f} "
                  f"gini={cv['normalised_gini_mean']:.3f}")
        t = summary["test"]
        print(f"chosen: {summary['chosen_model']} | held-out d2={t['d2']:.3f} gini={t['normalised_gini']:.3f} "
              f"driver_spearman={t['driver_spearman']:.3f}")
        print(f"saved run to {out}")
        return 0

    bundle = load_run(args.run)
    if args.cmd == "score":
        drivers = score_drivers(bundle, table)
        if args.out == "-":
            drivers.to_csv(sys.stdout, index=False, float_format="%.3f")
        else:
            drivers.to_csv(args.out, index=False, float_format="%.3f")
            print(f"wrote {len(drivers)} drivers to {args.out}")
        return 0

    if args.cmd == "explain":
        imp = permutation_importance(bundle["model"], table, seed=seed)
        print(json.dumps({"features": len(FEATURES), "importance": imp.to_dict(orient="records")}, indent=2))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
