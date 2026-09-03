"""Raqam web app + API.

Serves the offline-first PWA (app shell in raqam/web/static) and the online
helpers: server-side scan (CNN + OpenCV), the learning-engine demo endpoints,
and the sync target for records queued on field devices.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .. import store
from ..data import load
from ..dreams import gallery
from ..mlp import MLP
from ..pipeline import annotate, digitize, export_csv
from ..segment import synth_form
from ..train import CNN_MODEL, MODEL

app = FastAPI(title="Raqam")
_STATIC = Path(__file__).resolve().parent / "static"
_MODELS = Path(__file__).resolve().parent.parent.parent / "models"


def _mlp() -> MLP:
    if not Path(MODEL).exists():
        raise RuntimeError("no MLP — run: python -m raqam.train")
    return MLP.load(MODEL)


def _png_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- shell -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(_STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def sw():
    # served from root so its scope covers the whole app
    return FileResponse(_STATIC / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.get("/static/numerals_cnn.json")
def model_json():
    p = _MODELS / "numerals_cnn.json"
    if not p.exists():
        return JSONResponse({"error": "CNN not trained yet"}, status_code=404)
    return FileResponse(p, media_type="application/json")


# ---- learning-engine demo (Phases 1-4) ----------------------
@app.get("/api/train")
def train_stream(epochs: int = 3):
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
    strip = gallery(_mlp())
    return {"tiles": [_png_b64(strip[:, i * 28:(i + 1) * 28]) for i in range(10)]}


@app.post("/api/predict")
async def predict(payload: dict):
    x = np.asarray(payload["pixels"], dtype="float32").reshape(1, 784)
    p = _mlp().forward(x)[0]
    return {"digit": int(p.argmax()), "probs": [round(float(v), 4) for v in p]}


# ---- form scanning (Phases 5-6) -----------------------------
@app.post("/api/scan")
async def scan(file: UploadFile = File(...), form: str = "form", field: str = "field",
               threshold: float = 0.95):
    raw = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "bad image"}, status_code=400)
    rec = digitize(img, form, field, threshold=threshold)
    review = _png_b64(cv2.cvtColor(annotate(rec), cv2.COLOR_BGR2RGB))
    rec.pop("_gray", None)
    return {**rec, "review_image": review}


@app.get("/api/sample-form")
def sample_form():
    import random
    digits = [random.randint(0, 9) for _ in range(6)]
    png = cv2.imencode(".png", synth_form(digits, seed=random.randint(0, 9999)))[1]
    return Response(png.tobytes(), media_type="image/png")


# ---- field-device sync (Phase 8) ---------------------------
@app.post("/api/sync")
async def sync(payload: dict):
    recs = payload.get("records", [])
    ids = []
    for r in recs:
        ids.append(store.add(r.get("form", ""), r.get("field", ""), r.get("value", ""),
                             bool(r.get("needsReview")), r.get("imgSha", ""),
                             r.get("cells", [])))
    return {"ok": True, "stored": len(ids), "server_ids": ids, "ts": time.time()}


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


# ---- pilot evaluation (Phase 9) ---------------------------
@app.get("/api/evaluation")
def evaluation():
    p = _MODELS / "evaluation.json"
    return json.loads(p.read_text()) if p.exists() else {}


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
