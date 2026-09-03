"""Train a recognizer and save its weights to models/.

    python -m raqam.train                      # MLP on MNIST  -> models/mnist_mlp.npz
    python -m raqam.train --model cnn --data numerals   # CNN on MNIST+HODA -> models/numerals_cnn.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data import load, load_numerals

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL = MODELS / "mnist_mlp.npz"          # default, kept for back-compat
CNN_MODEL = MODELS / "numerals_cnn.npz"   # the production recognizer


def _data(name):
    return load_numerals() if name == "numerals" else load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--data", choices=["mnist", "numerals"], default="mnist")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()

    (xtr, ytr), (xte, yte) = _data(a.data)
    print(f"{a.model} on {a.data}: {xtr.shape[0]} train / {xte.shape[0]} test")

    if a.model == "cnn":
        from .cnn import CNN
        net = CNN()
        epochs = a.epochs or 10
        out = a.out or CNN_MODEL
        kw = dict(epochs=epochs, batch=128, lr=1e-3)
    else:
        from .mlp import MLP
        net = MLP(sizes=(784, 128, 64, 10))
        epochs = a.epochs or 12
        out = a.out or (MODELS / f"{a.data}_mlp.npz" if a.data == "numerals" else MODEL)
        kw = dict(epochs=epochs, batch=128, lr=0.1)

    t0 = time.time()
    for m in net.fit(xtr, ytr, val=(xte, yte), **kw):
        if m["step"] % 200 == 0:
            print(f"ep{m['epoch']:2d} step{m['step']:6d} loss={m['loss']:.3f} "
                  f"val_acc={m.get('val_acc', float('nan')):.4f} "
                  f"[{time.time()-t0:.0f}s]")
    acc = net.score(xte, yte)
    net.save(out)
    if hasattr(net, "export_json"):
        from .triage import Triage
        n = min(6000, len(xte) // 3)
        temp = Triage.calibrate(net, xte[-n:], yte[-n:]).t
        net.export_json(MODELS / "numerals_cnn.json", temperature=temp)  # web app serves this
        print(f"calibrated temperature T={temp:.2f}")
    print(f"\ndone in {time.time()-t0:.0f}s — test accuracy {acc:.4f} — saved {out}")
    assert acc >= 0.92


if __name__ == "__main__":
    main()
