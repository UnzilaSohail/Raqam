"""OpenCV form pipeline: photo -> deskewed image -> printed digit-box crops
-> MNIST-style 28x28 tensors ready for the recognizer.

Tuned for the common case: a printed row (or grid) of empty boxes, one digit
per box. Real templates need the thresholds below adjusted per form — that
knob is deliberate, not laziness (roll-number boxes != a meter dial).
"""
from __future__ import annotations

import cv2
import numpy as np


# --- preprocess --------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate so text/lines are axis-aligned. Angle from the dominant edges."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=gray.shape[1] // 3,
                            maxLineGap=20)
    if lines is None:
        return gray
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if abs(a) < 30:  # near-horizontal only
            angles.append(a)
    if not angles:
        return gray
    ang = float(np.median(angles))
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderValue=255)


def binarize(gray: np.ndarray) -> np.ndarray:
    """White ink on black background (digits = 255), like MNIST."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 10)


# --- box detection ----------------------------------------------------
def find_digit_cells(gray: np.ndarray, min_frac=0.010, max_frac=0.15,
                     ar_range=(0.4, 2.6)) -> list[tuple[int, int, int, int]]:
    """Return printed box rects (x, y, w, h), reading order."""
    inv = binarize(gray)
    inv = cv2.dilate(inv, np.ones((2, 2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = gray.shape[0] * gray.shape[1]
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_frac * area < w * h < max_frac * area):
            continue
        if not (ar_range[0] < w / h < ar_range[1]):
            continue
        if cv2.contourArea(c) < 0.6 * w * h:  # roughly rectangular
            continue
        boxes.append((x, y, w, h))
    boxes = _dedupe(boxes)
    med_h = np.median([h for _, _, _, h in boxes]) if boxes else 1
    boxes.sort(key=lambda b: (round(b[1] / (med_h * 0.7)), b[0]))  # row, then x
    return boxes


def _dedupe(boxes, iou_thresh=0.6):
    keep = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        if all(_iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


# --- cell -> MNIST tensor -------------------------------------------
def cell_to_mnist(gray: np.ndarray, box, pad_frac=0.14) -> np.ndarray:
    """Crop one box, strip the printed border, center the ink by mass,
    scale the glyph to 20px inside a 28x28 frame (MNIST convention)."""
    x, y, w, h = box
    p = int(min(w, h) * pad_frac)
    crop = gray[y + p:y + h - p, x + p:x + w - p]
    if crop.size == 0:
        return np.zeros((28, 28), "float32")
    ink = binarize(crop)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 8:  # empty box
        return np.zeros((28, 28), "float32")
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gh, gw = ink.shape
    s = 20.0 / max(gh, gw)
    ink = cv2.resize(ink, (max(1, round(gw * s)), max(1, round(gh * s))),
                     interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), "float32")
    gh, gw = ink.shape
    oy, ox = (28 - gh) // 2, (28 - gw) // 2
    out[oy:oy + gh, ox:ox + gw] = ink
    # shift so center of mass sits at the middle, like MNIST
    cy, cx = _com(out)
    out = _shift(out, 14 - cx, 14 - cy)
    return (out / 255.0).astype("float32")


def _com(a):
    tot = a.sum()
    if tot == 0:
        return 14.0, 14.0
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (ys * a).sum() / tot, (xs * a).sum() / tot


def _shift(a, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, m, (a.shape[1], a.shape[0]))


def extract_fields(img: np.ndarray):
    """Full path: image -> (list of 784-vectors, list of boxes)."""
    gray = deskew(to_gray(img))
    boxes = find_digit_cells(gray)
    cells = np.stack([cell_to_mnist(gray, b).ravel() for b in boxes]) \
        if boxes else np.zeros((0, 784), "float32")
    return cells, boxes, gray


# --- synthetic form, for the self-check ---------------------------
def synth_form(digits, cell=90, seed=0):
    """Render a row of printed boxes with MNIST glyphs inside — a stand-in
    for a real scan until a real template is supplied."""
    from .data import load

    (xtr, ytr), _ = load(flatten=False, normalize=False)
    rng = np.random.default_rng(seed)
    W = cell * len(digits) + 40
    canvas = np.full((cell + 60, W), 255, np.uint8)
    for i, d in enumerate(digits):
        x0 = 20 + i * cell
        cv2.rectangle(canvas, (x0, 25), (x0 + cell - 8, 25 + cell - 8), 0, 2)
        pool = np.where(ytr == d)[0]
        g = xtr[rng.choice(pool)]
        g = cv2.resize(g, (cell - 34, cell - 34))
        canvas[42:42 + g.shape[0], x0 + 13:x0 + 13 + g.shape[1]] = \
            np.minimum(canvas[42:42 + g.shape[0], x0 + 13:x0 + 13 + g.shape[1]],
                       255 - g)
    return canvas


if __name__ == "__main__":
    from .mlp import MLP
    from .train import MODEL

    truth = [4, 2, 7, 0, 9, 1]
    form = synth_form(truth, seed=3)
    cells, boxes, _ = extract_fields(form)
    assert len(boxes) == len(truth), f"found {len(boxes)} boxes, want {len(truth)}"
    net = MLP.load(MODEL)
    pred = net.predict(cells).tolist()
    print("truth", truth, "pred", pred)
    assert sum(a == b for a, b in zip(truth, pred)) >= 5  # allow 1 synth/scan slip
    print("segment ok")
