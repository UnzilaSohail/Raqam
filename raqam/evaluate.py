"""Phase-9 evaluation harness — measure the pipeline against a manual baseline.

Real forms:  python -m raqam.evaluate --scans path/to/dir  (dir has *.png + labels.csv: filename,value)
No forms yet: python -m raqam.evaluate --synthetic 300     (renders labelled synthetic fields)

Writes models/evaluation.json — the district dashboard reads it.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .pipeline import _triage
from .segment import extract_fields, synth_form

OUT = Path(__file__).resolve().parent.parent / "models" / "evaluation.json"
MANUAL_SEC_PER_FIELD = 25.0  # plan §10 baseline: ~20–30s manual re-entry


def _score_one(img, truth: str, tri):
    t0 = time.perf_counter()
    cells_x, boxes, _ = extract_fields(img)          # capture-side timing excluded (no camera here)
    cells = tri.run(cells_x) if len(cells_x) else []
    dt = time.perf_counter() - t0
    pred = "".join(str(c.digit) for c in cells)
    n = min(len(truth), len(pred))
    wrong = sum(truth[i] != pred[i] for i in range(n)) + abs(len(truth) - len(pred))
    # residual = wrong digits that were NOT flagged (a human never saw them)
    residual = sum(truth[i] != str(cells[i].digit) and not cells[i].flagged
                   for i in range(n))
    auto = sum(not c.flagged for c in cells)
    return {"digits": max(len(truth), len(pred)), "wrong": wrong,
            "residual": residual, "auto": auto, "flagged": len(cells) - auto,
            "sec": dt, "detected": len(cells) == len(truth)}


def evaluate(pairs, threshold=0.95):
    tri = _triage(threshold)
    agg = {"n": 0, "digits": 0, "wrong": 0, "residual": 0, "auto": 0,
           "flagged": 0, "sec": 0.0, "detect_fail": 0}
    for img, truth in pairs:
        r = _score_one(img, truth, tri)
        agg["n"] += 1
        for k in ("digits", "wrong", "residual", "auto", "flagged", "sec"):
            agg[k] += r[k]
        agg["detect_fail"] += (not r["detected"])
    d = max(agg["digits"], 1)
    fields = max(agg["n"], 1)
    out = {
        "n": agg["n"],
        "digit_error": agg["wrong"] / d,
        "residual_error": agg["residual"] / d,
        "auto_rate": agg["auto"] / max(agg["auto"] + agg["flagged"], 1),
        "box_detection_fail_rate": agg["detect_fail"] / fields,
        "sec_per_field": agg["sec"] / fields,
        "manual_sec_per_field": MANUAL_SEC_PER_FIELD,
        "time_saved_pct": 1 - (agg["sec"] / fields) / MANUAL_SEC_PER_FIELD,
        "threshold": threshold,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    return out


def _load_scans(d: Path):
    csv_path = d / "labels.csv"
    if not csv_path.exists():
        raise SystemExit(f"no labels.csv in {d} — see data/real_forms/README.md")
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    if rows and rows[0][:2] == ["filename", "value"]:
        rows = rows[1:]                       # skip header
    n = 0
    for row in rows:
        if len(row) < 2 or not row[0].strip():
            continue
        name, val = row[0].strip(), row[1].strip()
        if val.upper() == "SKIP" or val == "":
            continue
        img = cv2.imread(str(d / name))
        if img is None:
            print(f"  ! cannot read {name}")
            continue
        n += 1
        yield img, val
    if not n:
        raise SystemExit(f"no readable labelled images in {d}")


def _synthetic(n: int, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        k = rng.integers(4, 8)
        digits = rng.integers(0, 10, k).tolist()
        yield synth_form(digits, seed=int(rng.integers(1e9))), "".join(map(str, digits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=Path)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--sweep", action="store_true",
                    help="try several thresholds and print the accuracy/auto-accept tradeoff")
    a = ap.parse_args()

    if a.scans:
        pairs = list(_load_scans(a.scans))
    elif a.synthetic:
        pairs = list(_synthetic(a.synthetic))
    else:
        ap.error("pass --scans DIR or --synthetic N")

    if a.sweep:
        print(f"{'thresh':>7} {'auto-accept':>12} {'residual err':>13} {'digit err':>10}")
        for t in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
            r = evaluate(pairs, t)
            print(f"{t:>7.2f} {r['auto_rate']*100:>11.1f}% {r['residual_error']*100:>12.2f}% "
                  f"{r['digit_error']*100:>9.2f}%")
        return

    res = evaluate(pairs, a.threshold)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # self-check: synthetic run produces sane numbers
    res = evaluate(list(_synthetic(25)), 0.95)
    assert res["n"] == 25
    assert 0 <= res["digit_error"] <= 0.5
    assert res["residual_error"] <= res["digit_error"] + 1e-9
    assert res["sec_per_field"] < MANUAL_SEC_PER_FIELD
    print("evaluate ok", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in res.items() if k in
                          ("n", "digit_error", "residual_error", "auto_rate")})
