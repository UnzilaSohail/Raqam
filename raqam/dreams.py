"""Phase 3 — "dreams": what the network thinks each digit looks like.

Gradient ascent on the input to maximise a class logit, with light L2 decay and
blur so the result is a smooth glyph rather than adversarial noise.
"""
from __future__ import annotations

import numpy as np

from .mlp import MLP


def _blur(img):  # 3x3 box blur, no scipy
    k = np.ones((3, 3)) / 9.0
    p = np.pad(img.reshape(28, 28), 1, mode="edge")
    out = sum(k[i, j] * p[i:i + 28, j:j + 28] for i in range(3) for j in range(3))
    return out.ravel()


def dream(net: MLP, digit: int, steps: int = 200, lr: float = 2.0,
          decay: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 0.1, (1, 784)).astype("float32")
    for t in range(steps):
        g = net.input_gradient(x, digit)
        x += lr * g / (np.abs(g).mean() + 1e-8)
        x *= (1 - decay)
        if t % 4 == 0:
            x = _blur(x)[None, :]
        np.clip(x, 0, 1, out=x)
    x -= x.min()
    x /= x.max() + 1e-9
    return x.reshape(28, 28)


def gallery(net: MLP) -> np.ndarray:
    """10 dreamed digits tiled into one 28x280 strip."""
    return np.concatenate([dream(net, d) for d in range(10)], axis=1)


if __name__ == "__main__":
    from .train import MODEL

    net = MLP.load(MODEL)
    # a dreamed digit should read as that class to the same net
    hits = sum(net.predict(dream(net, d).reshape(1, 784))[0] == d for d in range(10))
    print(f"dreams recognised by the net: {hits}/10")
    assert hits >= 7
    print("dreams ok")
