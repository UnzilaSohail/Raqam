"""Raqam web UI — training viz, dream gallery, draw-and-predict, form scanner."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from PIL import Image

from ..data import load
from ..dreams import gallery
from ..mlp import MLP
from ..pipeline import annotate, digitize, export_csv
from ..train import MODEL
from .. import store

app = FastAPI(title="Raqam")
_STATIC = Path(__file__).resolve().parent / "static"


def _net() -> MLP:
    if not Path(MODEL).exists():
        raise RuntimeError("no model — run: python -m raqam.train")
    return MLP.load(MODEL)


def _png_b64(arr: np.ndarray) -> str:
    arr = np.clip(arr * 255, 0, 255).astype("uint8") if arr.dtype != np.uint8 else arr
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/train")
def train_stream(epochs: int = 3):
    """SSE: one event per 50 steps with loss + validation accuracy."""
    (xtr, ytr), (xte, yte) = load()
    net = MLP(sizes=(784, 128, 64, 10))

    def gen():
        for m in net.fit(xtr, ytr, epochs=epochs, val=(xte[:2000], yte[:2000])):
            if m["step"] % 50 == 0:
                yield f"data: {json.dumps(m)}\n\n"
        net.save(MODEL)
        yield f"data: {json.dumps({'done': True, 'val_acc': net.score(xte, yte)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/dreams")
def dreams():
    strip = gallery(_net())
    return {"tiles": [_png_b64(strip[:, i * 28:(i + 1) * 28]) for i in range(10)]}


@app.post("/api/predict")
async def predict(payload: dict):
    """payload: {pixels: [784 floats 0..1]}  ->  digit + probabilities."""
    x = np.asarray(payload["pixels"], dtype="float32").reshape(1, 784)
    p = _net().forward(x)[0]
    return {"digit": int(p.argmax()), "probs": [round(float(v), 4) for v in p]}


@app.post("/api/scan")
async def scan(file: UploadFile = File(...), form: str = "form", field: str = "field",
               threshold: float = 0.95):
    raw = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "bad image"}, status_code=400)
    rec = digitize(img, form, field, threshold=threshold)
    review_png = _png_b64(cv2.cvtColor(annotate(rec), cv2.COLOR_BGR2RGB))
    rec.pop("_gray", None)
    return {**rec, "review_image": review_png}


@app.get("/api/pending")
def pending():
    return store.pending()


@app.post("/api/resolve")
async def resolve(payload: dict):
    store.resolve(int(payload["id"]), str(payload["value"]))
    return {"ok": True}


@app.get("/api/export.csv")
def export():
    p = export_csv(Path(store.DB).with_name("export.csv"))
    return FileResponse(p, filename="raqam_export.csv", media_type="text/csv")
