"""evals/promptfoo/split_dataset.py（開発用と最終評価用への分割）のテスト。架空の小さなデータで確かめる"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "promptfoo"))

from split_dataset import family, split_scenarios  # noqa: E402

LANGS = ["ja", "en", "zh", "ar", "fr", "sw"]


def _rows():
    rows = []
    for i in range(1, 11):
        rows += [{"scenario_id": f"W{i:02d}", "language": l, "is_violation": "true"} for l in LANGS]
    for i in range(1, 16):
        rows += [{"scenario_id": f"S{i:02d}", "language": l, "is_violation": "false"} for l in LANGS]
    rows += [{"scenario_id": "X01", "language": "ja+en", "is_violation": True},
             {"scenario_id": "X02", "language": "fr+en", "is_violation": False}]
    return rows


def test_family():
    assert family("W01") == "W"
    assert family("JH118") == "JH"
    assert family("x02") == "X"


def test_every_scenario_assigned_and_translations_stay_together():
    rows = _rows()
    split = split_scenarios(rows)
    assert set(split) == {r["scenario_id"] for r in rows}
    assert set(split.values()) == {"dev", "test"}
    # 対訳は同じ側（split はシナリオ単位なので、行ごとに見ても 1 つに決まる）
    parts_by_sid = {}
    for r in rows:
        parts_by_sid.setdefault(r["scenario_id"], set()).add(split[r["scenario_id"]])
    assert all(len(p) == 1 for p in parts_by_sid.values())


def test_stratified_ratio():
    split = split_scenarios(_rows(), test_ratio=0.4)
    w_test = sum(1 for s, p in split.items() if s.startswith("W") and p == "test")
    s_test = sum(1 for s, p in split.items() if s.startswith("S") and p == "test")
    assert w_test == 4   # 10 × 0.4
    assert s_test == 6   # 15 × 0.4


def test_same_seed_same_split_and_different_seed_differs():
    rows = _rows()
    assert split_scenarios(rows, seed=1) == split_scenarios(rows, seed=1)
    assert split_scenarios(rows, seed=1) != split_scenarios(rows, seed=2)


def test_inconsistent_label_within_scenario_raises():
    rows = [{"scenario_id": "W01", "language": "ja", "is_violation": "true"},
            {"scenario_id": "W01", "language": "en", "is_violation": "false"}]
    with pytest.raises(ValueError):
        split_scenarios(rows)


@pytest.mark.parametrize("ratio", [0, 1, 1.5])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        split_scenarios(_rows(), test_ratio=ratio)
