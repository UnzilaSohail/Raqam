"""MNIST loader. Downloads the standard Keras mirror once, caches to data/."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_CACHE = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"


def _download() -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MNIST -> {_CACHE}")
    urllib.request.urlretrieve(_URL, _CACHE)


def load(flatten: bool = True, normalize: bool = True):
    """Return (x_train, y_train), (x_test, y_test).

    x: float32, shape (N, 784) if flatten else (N, 28, 28), scaled to [0,1] if normalize.
    y: int64, shape (N,).
    """
    if not _CACHE.exists():
        _download()
    with np.load(_CACHE) as f:
        x_tr, y_tr, x_te, y_te = f["x_train"], f["y_train"], f["x_test"], f["y_test"]

    def prep(x):
        x = x.astype("float32")
        if normalize:
            x /= 255.0
        return x.reshape(len(x), -1) if flatten else x

    return (prep(x_tr), y_tr.astype("int64")), (prep(x_te), y_te.astype("int64"))


def one_hot(y: np.ndarray, classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), classes), dtype="float32")
    out[np.arange(len(y)), y] = 1.0
    return out


if __name__ == "__main__":
    (xtr, ytr), (xte, yte) = load()
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784)
    assert xtr.min() == 0.0 and xtr.max() == 1.0
    assert one_hot(ytr[:3]).sum() == 3
    print("data ok", xtr.shape, xte.shape)
