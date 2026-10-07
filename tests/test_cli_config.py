import json

import pandas as pd
import pytest

from driverisk.cli import main
from driverisk.config import Settings, load_env_file
from driverisk.synthetic import simulate_fleet


def test_simulator_is_seeded():
    a = simulate_fleet(2, 1, seed=3, min_trip_min=2, max_trip_min=2)
    b = simulate_fleet(2, 1, seed=3, min_trip_min=2, max_trip_min=2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["variable"]) == {"Vehicle speed", "ACCELERATION X", "ACCELERATION Y", "ENGINE RPM", "POSITION", "IGNITION_STATUS"}
    with pytest.raises(ValueError):
        simulate_fleet(1, 1)


def test_settings(monkeypatch, tmp_path):
    for k in ("DRIVERISK_WEATHER", "DRIVERISK_ROADS", "DRIVERISK_SEED"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.weather == "offline" and s.roads == "offline" and s.seed == 42
    monkeypatch.setenv("DRIVERISK_WEATHER", "csv")
    with pytest.raises(ValueError, match="WEATHER_CSV"):
        Settings.from_env()
    monkeypatch.setenv("DRIVERISK_WEATHER", "radar")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.delenv("DRIVERISK_WEATHER")
    monkeypatch.setenv("DRIVERISK_ACC_UNIT", "furlongs")
    with pytest.raises(ValueError):
        Settings.from_env()
    env = tmp_path / ".env"
    env.write_text("DRIVERISK_SEED=7\n", encoding="utf-8")
    monkeypatch.delenv("DRIVERISK_ACC_UNIT")
    assert load_env_file(env) == 1
    assert Settings.from_env().seed == 7


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DRIVERISK_DATA", raising=False)
    monkeypatch.delenv("DRIVERISK_SEED", raising=False)
    csv = tmp_path / "fleet.csv"
    assert main(["simulate", "--drivers", "8", "--trips", "4", "--seed", "1", "--out", str(csv)]) == 0
    assert main(["validate", str(csv)]) == 0
    assert "devices=8" in capsys.readouterr().out
    trips = tmp_path / "trips.csv"
    assert main(["trips", "--data", str(csv), "--out", str(trips)]) == 0
    run = tmp_path / "run"
    assert main(["train", "--trip-table", str(trips), "--models", "baseline,glm", "--folds", "3", "--out", str(run)]) == 0
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["chosen_model"] in ("baseline", "glm")
    assert (run / "model_card.md").read_text(encoding="utf-8").startswith("# Model card: driverisk")
    capsys.readouterr()
    assert main(["score", "--run", str(run), "--trip-table", str(trips)]) == 0
    assert capsys.readouterr().out.startswith("device,km,trips")
    assert main(["explain", "--run", str(run), "--trip-table", str(trips)]) == 0
    assert "importance" in json.loads(capsys.readouterr().out)
    assert main(["score", "--run", str(tmp_path / "none"), "--trip-table", str(trips)]) == 1
    assert main(["train", "--drivers", "6", "--trips", "2", "--models", "baseline", "--folds", "2",
                 "--out", str(tmp_path / "r2")]) == 0
