"""From-scratch NumPy CNN — the production recognizer (plan Phase 7).

conv-pool-conv-pool-fc-fc, im2col via sliding windows, Adam. Pure NumPy so it
deploys anywhere (Raspberry Pi, browser via exported weights) with no ML runtime.

Same surface as MLP: forward / predict / fit (generator) / score / save / load.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import one_hot


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --- conv / pool primitives ----------------------------------------
def _conv_fwd(x, W, b, pad):
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))          # N,Cin,Ho,Wo,kh,kw
    Ho, Wo = win.shape[2], win.shape[3]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, Ho * Wo, Cin * kh * kw)
    out = cols @ W.reshape(Cout, -1).T + b                        # N, Ho*Wo, Cout
    return out.transpose(0, 2, 1).reshape(N, Cout, Ho, Wo), (cols, x.shape, xp.shape)


def _conv_bwd(dout, W, cache, pad):
    cols, x_shape, xp_shape = cache
    N, Cin, H, Wd = x_shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = dout.shape[2], dout.shape[3]
    dcol_out = dout.reshape(N, Cout, -1).transpose(0, 2, 1)       # N, Ho*Wo, Cout
    # dout already carries the 1/batch factor (from dlogits) — do not divide again
    dW = np.einsum("npk,npc->ck", cols, dcol_out).reshape(W.shape)
    db = dcol_out.sum((0, 1))
    dcols = dcol_out @ W.reshape(Cout, -1)                        # N, Ho*Wo, Cin*kh*kw
    dc = dcols.reshape(N, Ho, Wo, Cin, kh, kw)
    dxp = np.zeros(xp_shape)
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i:i + Ho, j:j + Wo] += dc[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    dx = dxp[:, :, pad:pad + H, pad:pad + Wd]
    return dx, dW, db


def _pool_fwd(x):
    N, C, H, W = x.shape
    xr = x.reshape(N, C, H // 2, 2, W // 2, 2)
    out = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None])
    return out, (mask, x.shape)


def _pool_bwd(dout, cache):
    mask, shape = cache
    N, C, H, W = shape
    d = dout[:, :, :, None, :, None] * mask
    # split ties so gradient sums correctly
    d /= np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1)
    return d.reshape(shape)


def _he(shape, rng):
    fan_in = np.prod(shape[1:])
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype("float64")


class CNN:
    def __init__(self, seed: int = 0, c1: int = 8, c2: int = 16, fc: int = 64):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": _he((c1, 1, 3, 3), rng), "b1": np.zeros(c1),
            "W2": _he((c2, c1, 3, 3), rng), "b2": np.zeros(c2),
            "W3": _he((c2 * 7 * 7, fc), rng), "b3": np.zeros(fc),
            "W4": _he((fc, 10), rng), "b4": np.zeros(10),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    # --- forward ---------------------------------------------------
    def _forward(self, x, cache=False):
        p = self.p
        x = x.reshape(-1, 1, 28, 28).astype("float64")
        a1, c1 = _conv_fwd(x, p["W1"], p["b1"], 1)
        r1 = np.maximum(a1, 0)
        pl1, cp1 = _pool_fwd(r1)
        a2, c2 = _conv_fwd(pl1, p["W2"], p["b2"], 1)
        r2 = np.maximum(a2, 0)
        pl2, cp2 = _pool_fwd(r2)
        flat = pl2.reshape(len(x), -1)
        z3 = flat @ p["W3"] + p["b3"]
        r3 = np.maximum(z3, 0)
        logits = r3 @ p["W4"] + p["b4"]
        prob = _softmax(logits)
        if not cache:
            return prob
        return prob, (x, c1, a1, cp1, c2, a2, cp2, pl2, flat, z3, r3)

    def forward(self, x):
        return self._forward(x)

    def logits(self, x, batch: int = 512):
        p = np.concatenate([self._forward(x[i:i + batch])
                            for i in range(0, len(x), batch)])
        return np.log(np.clip(p, 1e-12, 1.0))  # log-probs: same argmax + temp behaviour

    def predict(self, x, batch: int = 512):
        return np.concatenate([self._forward(x[i:i + batch]).argmax(1)
                               for i in range(0, len(x), batch)])

    def score(self, x, y, batch: int = 512):
        return float((self.predict(x, batch) == y).mean())

    # --- backward + Adam step ------------------------------------
    def _grads(self, x, y):
        p = self.p
        prob, cc = self._forward(x, cache=True)
        (xi, c1, a1, cp1, c2, a2, cp2, pl2, flat, z3, r3) = cc
        n = len(x)
        loss = -np.log(np.clip(prob[np.arange(n), y], 1e-12, 1)).mean()

        dlogits = (prob - one_hot(y)) / n
        g = {}
        g["W4"] = r3.T @ dlogits
        g["b4"] = dlogits.sum(0)
        dr3 = dlogits @ p["W4"].T
        dz3 = dr3 * (z3 > 0)
        g["W3"] = flat.T @ dz3
        g["b3"] = dz3.sum(0)
        dflat = dz3 @ p["W3"].T
        dpl2 = dflat.reshape(pl2.shape)
        dr2 = _pool_bwd(dpl2, cp2)
        da2 = dr2 * (a2 > 0)
        dpl1, g["W2"], g["b2"] = _conv_bwd(da2, p["W2"], c2, 1)
        dr1 = _pool_bwd(dpl1, cp1)
        da1 = dr1 * (a1 > 0)
        _, g["W1"], g["b1"] = _conv_bwd(da1, p["W1"], c1, 1)
        return loss, g

    def _step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in self.p:
            self._m[k] = b1 * self._m[k] + (1 - b1) * g[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * g[k] ** 2
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, x, y, *, epochs=3, batch=128, lr=1e-3, val=None, seed=0,
            eval_every=300):
        rng = np.random.default_rng(seed)
        n = len(x)
        step = 0
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                loss, g = self._grads(x[idx], y[idx])
                self._step(g, lr)
                step += 1
                m = {"step": step, "epoch": ep, "loss": float(loss)}
                if step % eval_every == 0 and val is not None:
                    m["val_acc"] = self.score(val[0][:3000], val[1][:3000])
                yield m

    # --- persistence -------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v.astype("float32") for k, v in self.p.items()})

    @classmethod
    def load(cls, path):
        net = cls()
        with np.load(path) as f:
            net.p = {k: f[k].astype("float64") for k in f.files}
        return net

    def export_json(self, path, temperature: float = 1.7):
        """Weights + calibrated temperature for the browser recognizer (Phase 8 PWA)."""
        import json
        obj = {k: v.astype("float32").ravel().round(6).tolist() for k, v in self.p.items()}
        obj["shapes"] = {k: list(v.shape) for k, v in self.p.items()}
        obj["temperature"] = round(float(temperature), 3)
        Path(path).write_text(json.dumps(obj))


if __name__ == "__main__":
    # gradient check against finite differences on a tiny batch
    rng = np.random.default_rng(0)
    net = CNN(seed=0, c1=3, c2=4, fc=8)
    x = rng.random((4, 784))
    y = rng.integers(0, 10, 4)
    loss0, g = net._grads(x, y)
    for k in ["W1", "W4", "b2", "W3"]:
        flat = net.p[k].ravel()
        i = rng.integers(0, len(flat))
        orig = flat[i]
        e = 1e-5
        flat[i] = orig + e
        lp = net._grads(x, y)[0]
        flat[i] = orig - e
        lm = net._grads(x, y)[0]
        flat[i] = orig
        num = (lp - lm) / (2 * e)
        rel = abs(num - g[k].ravel()[i]) / (abs(num) + abs(g[k].ravel()[i]) + 1e-9)
        print(f"  {k}[{i}] num={num:+.5f} bp={g[k].ravel()[i]:+.5f} rel={rel:.1e}")
        assert rel < 2e-3, k

    # overfit 64 samples
    from .data import load
    (xt, yt), _ = load()
    net = CNN(seed=1)
    for m in net.fit(xt[:64], yt[:64], epochs=30, batch=64, lr=2e-3):
        pass
    acc = net.score(xt[:64], yt[:64])
    print("overfit-64 acc", acc)
    assert acc > 0.95
    print("cnn ok")
