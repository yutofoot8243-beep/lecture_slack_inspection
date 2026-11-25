#!/usr/bin/env python3
"""評価用データセットを、シナリオ単位で開発用（dev）と最終評価用（test）に分ける。

過学習を避けるため、判定方式の改善は dev だけを見て行い、test は最後に測る。
同じシナリオの対訳（6 言語）は、必ず同じ側に入れる（言語だけ違う同じ内容が両側にあると、test が dev の答えを
知っていることになるため）。

分け方:
- シナリオを「系列（ID の英字部分: W・S・JH・X など）× 警告すべきか」で層に分け、層ごとに同じ割合で分ける
- 乱数のシードを固定するので、同じデータ・同じシードなら毎回同じ分け方になる

使い方（evals/promptfoo/ で）:
    python split_dataset.py <eval_posts.jsonl|csv> [--test-ratio 0.4] [--seed 20261125] [--outdir datasets]
    → datasets/split.json（シナリオ → dev/test）、datasets/eval_posts_dev.jsonl、datasets/eval_posts_test.jsonl
    python load_dataset.py datasets/eval_posts_dev.jsonl    # dev だけで評価するとき
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from load_dataset import _bool, read_rows

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20261125
DEFAULT_TEST_RATIO = 0.4


def family(scenario_id: str) -> str:
    m = re.match(r"[A-Za-z]+", scenario_id)
    return m.group(0).upper() if m else "?"


def split_scenarios(rows: list[dict], test_ratio: float = DEFAULT_TEST_RATIO,
                    seed: int = DEFAULT_SEED) -> dict[str, str]:
    """シナリオ ID → "dev" / "test"。層（系列 × 警告すべきか）ごとに、test の割合を test_ratio に近づける"""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio は 0 より大きく 1 より小さい値にする")
    label: dict[str, bool] = {}
    for r in rows:
        sid, v = r["scenario_id"], _bool(r["is_violation"])
        # 対訳は同じ正解のはず。違えば、シナリオ単位で分けられないので止める
        if label.setdefault(sid, v) != v:
            raise ValueError(f"シナリオ {sid} の中で正解が分かれている")
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for sid, v in label.items():
        strata[(family(sid), v)].append(sid)

    rng = random.Random(seed)
    out: dict[str, str] = {}
    for key in sorted(strata):
        sids = sorted(strata[key])
        rng.shuffle(sids)
        n_test = round(len(sids) * test_ratio)
        for i, sid in enumerate(sids):
            out[sid] = "test" if i < n_test else "dev"
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--outdir", type=Path, default=HERE / "datasets")
    a = ap.parse_args()

    rows = read_rows(a.dataset)
    split = split_scenarios(rows, a.test_ratio, a.seed)
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "split.json").write_text(
        json.dumps({"seed": a.seed, "test_ratio": a.test_ratio, "scenarios": split}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    for part in ("dev", "test"):
        sel = [r for r in rows if split[r["scenario_id"]] == part]
        with open(a.outdir / f"eval_posts_{part}.jsonl", "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_sc = sum(1 for v in split.values() if v == part)
        n_v = sum(1 for r in sel if _bool(r["is_violation"]))
        print(f"{part}: シナリオ {n_sc}・投稿 {len(sel)} 件（警告すべき {n_v}・警告しない {len(sel) - n_v}）")
    by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    labels = {r["scenario_id"]: _bool(r["is_violation"]) for r in rows}
    for sid, part in split.items():
        by_stratum[f"{family(sid)}・{'警告' if labels[sid] else '非警告'}"][part == "test"] += 1
    print("層ごとの dev / test のシナリオ数: " + "、".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_stratum.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
