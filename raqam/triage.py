"""Confidence triage — the actual product.

Raw softmax probabilities are overconfident, so we temperature-scale them on a
held-out set first, then route every digit whose calibrated top probability is
below `threshold` to a human instead of guessing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mlp import MLP, _softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """1-D search for the T that minimises NLL. Good enough; no optimiser needed."""
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 5.0, 91):
        p = _softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


@dataclass
class Cell:
    index: int
    digit: int
    confidence: float
    flagged: bool


class Triage:
    def __init__(self, net: MLP, temperature: float = 1.0, threshold: float = 0.90):
        self.net = net
        self.t = temperature
        self.threshold = threshold

    @classmethod
    def calibrate(cls, net: MLP, x_val, y_val, threshold: float = 0.90):
        # logits = pre-softmax output of the last layer
        _, acts, _ = net.forward(x_val, cache=True)
        t = fit_temperature(acts[-1], y_val)
        return cls(net, t, threshold)

    def proba(self, x):
        _, acts, _ = self.net.forward(x, cache=True)
        return _softmax(acts[-1] / self.t)

    def run(self, cells_x: np.ndarray) -> list[Cell]:
        p = self.proba(cells_x)
        digit = p.argmax(1)
        conf = p.max(1)
        return [
            Cell(i, int(d), float(c), bool(c < self.threshold))
            for i, (d, c) in enumerate(zip(digit, conf))
        ]

    def field_value(self, cells: list[Cell]) -> tuple[str, bool]:
        """Join digits into a field string; second value = needs review."""
        s = "".join("?" if c.flagged else str(c.digit) for c in cells)
        return s, any(c.flagged for c in cells)


if __name__ == "__main__":
    from .data import load

    from .train import MODEL

    (xtr, ytr), (xte, yte) = load()
    net = MLP.load(MODEL)
    # calibrate on a held-out slice of test, evaluate on the rest
    xcal, ycal, xev, yev = xte[8000:], yte[8000:], xte[:8000], yte[:8000]
    tri = Triage.calibrate(net, xcal, ycal, threshold=0.99)
    cells = tri.run(xev)
    auto = [c for c in cells if not c.flagged]
    acc_auto = np.mean([c.digit == yev[c.index] for c in auto])
    raw = net.score(xev, yev)
    print(f"T={tri.t:.2f}  auto-accepted {100*len(auto)/len(xev):.1f}%  "
          f"acc(auto)={acc_auto:.4f}  raw={raw:.4f}  flagged={len(xev)-len(auto)}")
    # the whole point: auto-accepted digits are cleaner than blind recognition
    assert acc_auto > raw and acc_auto > 0.99
    print("triage ok")
