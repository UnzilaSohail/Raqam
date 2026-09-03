"""From-scratch NumPy MLP: ReLU hidden layers, softmax + cross-entropy output.

Plain minibatch SGD with momentum. No autograd — forward/backward by hand,
which is the whole point of the learning project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import one_hot


def _he(shape, rng):
    fan_in = shape[0]
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype("float32")


class MLP:
    def __init__(self, sizes=(784, 128, 64, 10), seed: int = 0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.W = [_he((a, b), rng) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(b, dtype="float32") for b in self.sizes[1:]]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._vb = [np.zeros_like(x) for x in self.b]

    # --- forward ---------------------------------------------------------
    def forward(self, x, cache=False):
        acts = [x]
        pre = []
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            pre.append(z)
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        logits = acts[-1]
        p = _softmax(logits)
        return (p, acts, pre) if cache else p

    def predict(self, x):
        return self.forward(x).argmax(1)

    def logits(self, x):
        return self.forward(x, cache=True)[1][-1]

    # --- backward -------------------------------------------------------
    def _grads(self, x, y, l2=0.0):
        n = len(x)
        p, acts, pre = self.forward(x, cache=True)
        loss = -np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean()
        if l2:
            loss += l2 * sum((w * w).sum() for w in self.W)

        dz = (p - one_hot(y, self.sizes[-1])) / n  # dL/dlogits
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ dz + (2 * l2 * self.W[i] if l2 else 0.0)
            gb[i] = dz.sum(0)
            if i > 0:
                dz = (dz @ self.W[i].T) * (pre[i - 1] > 0)
        return loss, gW, gb

    # --- training -----------------------------------------------------
    def fit(self, x, y, *, epochs=8, batch=128, lr=0.1, momentum=0.9, l2=0.0,
            val=None, seed=0):
        """Generator: yields a dict of metrics after every minibatch step."""
        rng = np.random.default_rng(seed)
        n = len(x)
        step = 0
        for ep in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                loss, gW, gb = self._grads(x[idx], y[idx], l2)
                for i in range(len(self.W)):
                    self._vW[i] = momentum * self._vW[i] - lr * gW[i]
                    self._vb[i] = momentum * self._vb[i] - lr * gb[i]
                    self.W[i] += self._vW[i]
                    self.b[i] += self._vb[i]
                step += 1
                m = {"step": step, "epoch": ep, "loss": float(loss)}
                if step % 50 == 0 and val is not None:
                    m["val_acc"] = self.score(*val)
                yield m

    def score(self, x, y):
        return float((self.predict(x) == y).mean())

    # --- Phase 3: "dreams" — gradient of a class logit w.r.t. the input --
    def input_gradient(self, x, target: int):
        _, _, pre = self.forward(x, cache=True)
        dz = np.zeros((len(x), self.sizes[-1]), dtype="float32")
        dz[:, target] = 1.0  # seed dL/dlogits for the target class
        for i in reversed(range(len(self.W))):
            dz = dz @ self.W[i].T
            if i > 0:
                dz = dz * (pre[i - 1] > 0)  # backprop through that layer's ReLU
        return dz

    # --- persistence -------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sizes=np.array(self.sizes),
                 **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        f = np.load(path)
        net = cls(sizes=tuple(int(s) for s in f["sizes"]))
        net.W = [f[f"W{i}"] for i in range(len(net.W))]
        net.b = [f[f"b{i}"] for i in range(len(net.b))]
        net._vW = [np.zeros_like(w) for w in net.W]
        net._vb = [np.zeros_like(x) for x in net.b]
        return net


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    # gradient check on tiny random net + learn XOR
    rng = np.random.default_rng(1)
    net = MLP(sizes=(4, 5, 3), seed=1)
    x = rng.standard_normal((6, 4)).astype("float32")
    y = rng.integers(0, 3, 6)
    loss0, gW, _ = net._grads(x, y)
    eps = 1e-4
    i, j, k = 0, 1, 2
    net.W[i][j, k] += eps
    lp = net._grads(x, y)[0]
    net.W[i][j, k] -= 2 * eps
    lm = net._grads(x, y)[0]
    net.W[i][j, k] += eps
    num = (lp - lm) / (2 * eps)
    assert abs(num - gW[i][j, k]) < 1e-3, (num, gW[i][j, k])

    xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
    xor_y = np.array([0, 1, 1, 0])
    net = MLP(sizes=(2, 16, 2), seed=0)
    for _ in net.fit(xor_x, xor_y, epochs=400, batch=4, lr=0.5):
        pass
    assert net.score(xor_x, xor_y) == 1.0
    print("mlp ok — grad check + XOR learned")
