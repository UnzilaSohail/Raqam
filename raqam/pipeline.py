"""Capture -> Preprocess -> Segment -> Recognize -> Triage -> Export.

The one call a caller (CLI or web) needs: `digitize(image, form, field)`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .mlp import MLP
from .segment import extract_fields
from .train import CNN_MODEL, MODEL
from .triage import Triage
from . import store

_MODEL_CACHE: dict[str, Triage] = {}
ENGINE = "none"


def _load_recognizer():
    """CNN on MNIST+HODA if trained, else the MNIST MLP. Returns (net, test_set)."""
    global ENGINE
    from .data import load, load_numerals
    if Path(CNN_MODEL).exists():
        from .cnn import CNN
        (_, _), (xte, yte) = load_numerals()
        ENGINE = "CNN (MNIST+HODA)"
        return CNN.load(CNN_MODEL), (xte, yte)
    (_, _), (xte, yte) = load()
    ENGINE = "MLP (MNIST)"
    return MLP.load(MODEL), (xte, yte)


def _triage(threshold: float) -> Triage:
    key = f"{threshold}"
    if key not in _MODEL_CACHE:
        net, (xte, yte) = _load_recognizer()
        n = min(6000, len(xte) // 3)
        _MODEL_CACHE[key] = Triage.calibrate(net, xte[-n:], yte[-n:], threshold)
    return _MODEL_CACHE[key]


def digitize(image: np.ndarray, form: str, field: str, *, threshold: float = 0.95,
             persist: bool = True) -> dict:
    sha = hashlib.sha256(np.ascontiguousarray(image)).hexdigest()
    cells_x, boxes, gray = extract_fields(image)
    tri = _triage(threshold)
    cells = tri.run(cells_x) if len(cells_x) else []
    value, needs = tri.field_value(cells)
    cell_rows = [
        {"digit": c.digit, "confidence": round(c.confidence, 4),
         "flagged": c.flagged, "bbox": list(map(int, b))}
        for c, b in zip(cells, boxes)
    ]
    rec = {"form": form, "field": field, "value": value, "needs_review": needs,
           "img_sha256": sha, "cells": cell_rows, "engine": ENGINE}
    if persist:
        rec["id"] = store.add(form, field, value, needs, sha, cell_rows)
    rec["_gray"] = gray
    return rec


def annotate(rec: dict) -> np.ndarray:
    """Review image: green box = auto-accepted, red = flagged for a human."""
    g = rec.pop("_gray") if "_gray" in rec else None
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g is not None else None
    if img is None:
        return np.zeros((10, 10, 3), np.uint8)
    for cell in rec["cells"]:
        x, y, w, h = cell["bbox"]
        col = (0, 0, 255) if cell["flagged"] else (0, 170, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        cv2.putText(img, str(cell["digit"]), (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return img


def export_csv(path: str | Path) -> Path:
    import csv
    rows = store.all_records()
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ts", "form", "field", "value", "needs_review",
                    "reviewed", "img_sha256"])
        for r in rows:
            w.writerow([r["id"], r["ts"], r["form"], r["field"], r["value"],
                        r["needs_review"], r["reviewed"], r["img_sha256"]])
    return path


def export_xlsx(path: str | Path) -> Path:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["id", "ts", "form", "field", "value", "needs_review", "reviewed",
               "img_sha256"])
    for r in store.all_records():
        ws.append([r["id"], r["ts"], r["form"], r["field"], r["value"],
                   r["needs_review"], r["reviewed"], r["img_sha256"]])
    wb.save(path)
    return Path(path)


if __name__ == "__main__":
    from .segment import synth_form

    truth = [1, 2, 3, 4, 5, 6]
    rec = digitize(synth_form(truth, seed=1), "demo", "roll_no", persist=False)
    print("value:", rec["value"], "needs_review:", rec["needs_review"])
    assert len(rec["cells"]) == 6
    assert rec["value"].replace("?", "") != "" or rec["needs_review"]
    img = annotate(rec)
    assert img.ndim == 3
    print("pipeline ok")
