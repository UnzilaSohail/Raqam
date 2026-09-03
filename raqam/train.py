"""Train the MNIST MLP and save weights to models/mnist_mlp.npz.

    python -m raqam.train --epochs 12
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data import load
from .mlp import MLP

MODEL = Path(__file__).resolve().parent.parent / "models" / "mnist_mlp.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=str(MODEL))
    a = ap.parse_args()

    (xtr, ytr), (xte, yte) = load()
    net = MLP(sizes=(784, 128, 64, 10))
    t0 = time.time()
    last = {}
    for m in net.fit(xtr, ytr, epochs=a.epochs, batch=a.batch, lr=a.lr,
                     val=(xte, yte)):
        last = m
        if m["step"] % 200 == 0:
            print(f"ep{m['epoch']:2d} step{m['step']:6d} "
                  f"loss={m['loss']:.3f} val_acc={m.get('val_acc', float('nan')):.4f}")
    acc = net.score(xte, yte)
    net.save(a.out)
    print(f"\ndone in {time.time()-t0:.1f}s — test accuracy {acc:.4f} — saved {a.out}")
    assert acc >= 0.92, "below Phase 1 exit criteria"


if __name__ == "__main__":
    main()
