"""Digit datasets.

- MNIST (Western digits) — standard Keras mirror, auto-downloaded.
- Urdu handwritten numerals ۰–۹ — 8,020 samples from ~900 writers, shipped in
  `data/urdu_digits.npz` (via the MNIST-MIX collection, jwwthu/MNIST-MIX).
- HODA (Perso-Arabic digits) — the large farsiocr.ir set, auto-downloaded, used
  as extra Perso-Arabic coverage alongside the real Urdu data.

`load_numerals()` returns the union, labelled by digit *value* (0–9) regardless
of script — the plan's "recognised natively, not transliterated" (§03). The real
Urdu set is upsampled so it isn't drowned by the much larger MNIST/HODA sets.
"""
from __future__ import annotations

import struct
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_HODA_BASE = ("https://raw.githubusercontent.com/amir-saniyan/HodaDatasetReader/"
              "master/DigitDB/")
_HODA_FILES = {"train": "Train 60000.cdb", "test": "Test 20000.cdb"}


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url.rsplit('/', 1)[-1]} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "raqam"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


# --- MNIST -----------------------------------------------------------
def load(flatten: bool = True, normalize: bool = True):
    """Return (x_train, y_train), (x_test, y_test) for MNIST."""
    cache = _DATA / "mnist.npz"
    if not cache.exists():
        _fetch(_MNIST_URL, cache)
    with np.load(cache) as f:
        x_tr, y_tr, x_te, y_te = f["x_train"], f["y_train"], f["x_test"], f["y_test"]

    def prep(x):
        x = x.astype("float32")
        if normalize:
            x /= 255.0
        return x.reshape(len(x), -1) if flatten else x

    return (prep(x_tr), y_tr.astype("int64")), (prep(x_te), y_te.astype("int64"))


# --- HODA (Perso-Arabic digits) ----------------------------------
def _read_hoda_cdb(path: Path):
    """Minimal reader for the HODA .cdb binary-image format."""
    data = path.read_bytes()
    o = 2 + 1 + 1  # yy, month, day
    H, W = data[o], data[o + 1]
    o += 2
    total = struct.unpack_from("I", data, o)[0]
    o += 4 + 128 * 4 + 1 + 256 + 245  # LetterCount, imgType, Comments, Reserved
    imgs, labels = [], []
    for _ in range(total):
        o += 1  # start byte 0xff
        labels.append(data[o]); o += 1
        w, h = (data[o], data[o + 1]) if not (W and H) else (W, H)
        if not (W and H):
            o += 2
        o += 2  # byte count
        img = np.zeros((h, w), np.uint8)
        for y in range(h):
            white, c = True, 0
            while c < w:
                n = data[o]; o += 1
                if not white:
                    img[y, c:c + n] = 255
                white = not white
                c += n
        imgs.append(img)
    return imgs, np.array(labels, "int64")


def _to_mnist_frame(glyph: np.ndarray) -> np.ndarray:
    """Any white-on-black glyph -> 28x28, 20px tall, centred by mass (MNIST rule)."""
    ys, xs = np.where(glyph > 0)
    if len(xs) < 4:
        return np.zeros((28, 28), "float32")
    g = glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype("float32")
    s = 20.0 / max(g.shape)
    g = cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), "float32")
    oy, ox = (28 - g.shape[0]) // 2, (28 - g.shape[1]) // 2
    out[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    tot = out.sum()
    cy, cx = (np.mgrid[0:28, 0:28][0] * out).sum() / tot, \
             (np.mgrid[0:28, 0:28][1] * out).sum() / tot
    m = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    out = cv2.warpAffine(out, m, (28, 28))
    out = cv2.GaussianBlur(out, (3, 3), 0)  # binary -> closer to MNIST's soft strokes
    return out / (out.max() + 1e-9)


def load_hoda(flatten: bool = True):
    """Return (x_train, y_train), (x_test, y_test) for HODA, MNIST-shaped."""
    npz = _DATA / "hoda_28.npz"
    if not npz.exists():
        parts = {}
        for split, fname in _HODA_FILES.items():
            cdb = _DATA / fname
            if not cdb.exists():
                _fetch(_HODA_BASE + fname.replace(" ", "%20"), cdb)
            imgs, labels = _read_hoda_cdb(cdb)
            x = np.stack([_to_mnist_frame(im) for im in imgs]).astype("float32")
            parts[f"x_{split}"], parts[f"y_{split}"] = x, labels
        np.savez_compressed(npz, **parts)
    with np.load(npz) as f:
        x_tr, y_tr, x_te, y_te = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    if flatten:
        x_tr, x_te = x_tr.reshape(len(x_tr), -1), x_te.reshape(len(x_te), -1)
    return (x_tr, y_tr), (x_te, y_te)


# --- real Urdu handwritten numerals -----------------------------
def load_urdu(flatten: bool = True):
    """Return (x_train, y_train), (x_test, y_test) for the real Urdu digit set."""
    npz = _DATA / "urdu_digits.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} missing — this file ships with the repo; restore it with "
            "`git checkout data/urdu_digits.npz`")
    with np.load(npz) as f:
        raw = {k: f[k] for k in ("x_train", "y_train", "x_test", "y_test")}

    def prep(x):
        x = np.stack([_to_mnist_frame(im) for im in x]).astype("float32")
        return x.reshape(len(x), -1) if flatten else x

    return ((prep(raw["x_train"]), raw["y_train"].astype("int64")),
            (prep(raw["x_test"]), raw["y_test"].astype("int64")))


# --- union: script-agnostic numerals -----------------------------
def load_numerals(flatten: bool = True, seed: int = 0, urdu_upsample: int = 6):
    """MNIST + real Urdu (upsampled) + HODA, shuffled, labelled by digit value 0–9."""
    (ax, ay), (atx, aty) = load(flatten=flatten)
    (ux, uy), (utx, uty) = load_urdu(flatten=flatten)
    (bx, by), (btx, bty) = load_hoda(flatten=flatten)
    rng = np.random.default_rng(seed)

    ux = np.repeat(ux, urdu_upsample, axis=0)
    uy = np.repeat(uy, urdu_upsample, axis=0)

    def cat(parts):
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        p = rng.permutation(len(x))
        return x[p], y[p]

    train = cat([(ax, ay), (ux, uy), (bx, by)])
    test = cat([(atx, aty), (utx, uty), (btx, bty)])
    return train, test


def one_hot(y: np.ndarray, classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), classes), dtype="float32")
    out[np.arange(len(y)), y] = 1.0
    return out


if __name__ == "__main__":
    (xtr, ytr), (xte, yte) = load()
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784)
    assert xtr.min() == 0.0 and xtr.max() == 1.0
    assert one_hot(ytr[:3]).sum() == 3
    print("mnist ok", xtr.shape)

    (ux, uy), (utx, uty) = load_urdu()
    assert ux.shape[1] == 784 and set(np.unique(uy)) == set(range(10))
    assert 5000 < len(ux) < 9000 and 1000 < len(utx) < 2000
    print("urdu ok", ux.shape, utx.shape)

    (hx, hy), (htx, hty) = load_hoda()
    assert hx.shape[1] == 784 and set(np.unique(hy)) == set(range(10))
    assert 40000 < len(hx) < 70000 and 15000 < len(htx) < 22000
    print("hoda ok", hx.shape, htx.shape)

    (nx, ny), (ntx, nty) = load_numerals()
    assert len(nx) == len(xtr) + 6 * len(ux) + len(hx)
    print("numerals ok", nx.shape, ntx.shape)
