# Raqam (رقم)

Offline handwritten-digit digitizer for paper forms. From-scratch NumPy engine,
confidence triage — every low-confidence digit is flagged for a human, never
silently guessed. Recognizes Western digits (0–9) **and** Urdu-Indic / Perso-Arabic
numerals (۰–۹) natively.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Linux/mac: .venv/bin/pip
.venv/Scripts/python -m raqam.train                    # MLP on MNIST, ~10s
.venv/Scripts/python -m raqam.train --model cnn --data numerals --epochs 8   # CNN on MNIST+HODA, ~12 min
.venv/Scripts/python -m raqam.cli serve                # http://127.0.0.1:8000  → install as an app
```

`python selfcheck.py` runs every module's built-in check. Docker: see [docs/deploy.md](docs/deploy.md).

## Phase map (all built)

| Plan phase | What | Where |
|---|---|---|
| 1 · network core | from-scratch ReLU MLP, hand-written backprop, gradient-checked | [raqam/mlp.py](raqam/mlp.py) |
| 2 · live training viz | loss + val-accuracy streamed to a chart while it trains | [raqam/train.py](raqam/train.py), `/api/train` |
| 3 · "dreams" | gradient ascent on the input — what each digit looks like to the net | [raqam/dreams.py](raqam/dreams.py) |
| 4 · draw & predict | canvas → centred 28×28 → live probabilities | Engine tab |
| 5 · confidence triage | temperature-scaled softmax + threshold routing; CSV/XLSX export | [raqam/triage.py](raqam/triage.py) |
| 6 · scanner pipeline | OpenCV deskew → printed-box detection → MNIST-style cells → triage | [raqam/segment.py](raqam/segment.py), [raqam/pipeline.py](raqam/pipeline.py) |
| 7 · CNN + Urdu numerals | from-scratch NumPy CNN (im2col, Adam); trained on MNIST ∪ HODA, one 0–9 model for both scripts | [raqam/cnn.py](raqam/cnn.py), [raqam/data.py](raqam/data.py) |
| 8 · offline field app | installable PWA: on-device inference (JS port of the CNN), IndexedDB queue, camera capture, opportunistic sync — phone **and** desktop | [raqam/web/](raqam/web/) |
| 9 · pilot evaluation | measures digit error rate (pre/post review), auto-accept rate, time vs. manual | [raqam/evaluate.py](raqam/evaluate.py), Dashboard tab |
| 10 · packaging / sustainability | Docker + compose, one-command deploy, data-handling policy, open-core split | [Dockerfile](Dockerfile), [docs/](docs/) |

## The offline app

`serve`, then **Install app** / **Add to Home Screen** in the browser. After the first
load the app works with the network off:

- **Scan** a printed row of digit boxes → per-digit confidence, flagged digits in red,
  annotated review image. Recognition runs in the browser (`raqam/web/static/recognize.js`,
  a hand-written JS port of the CNN + a connected-components box finder).
- **Review queue** / **Records** / **Export CSV** — all client-side, offline.
- **Sync to server** when connected — sends values + image hashes, never the image.
- **Dashboard** — auto-accept rate, pending count, per-form breakdown, Phase-9 eval metrics.
- **Program** — the brief in the app: the problem, six sectors ranked by pilot-readiness,
  the elections position, roadmap, funding (Ignite), sustainability, risks.

## Datasets

- **MNIST** — standard Keras mirror.
- **HODA** — public Perso-Arabic handwritten digits (farsiocr.ir), via the
  `amir-saniyan/HodaDatasetReader` GitHub mirror. Perso-Arabic digits are the same glyphs
  Urdu uses. 60k train / 20k test, auto-downloaded and cached to `data/`.
  *Confirm HODA's licence terms before a production deployment; swap in a Pakistan-specific
  Urdu set from a university partner when available — `load_hoda()` is the only thing to change.*

## Deliberate shortcuts (`ponytail:` comments)

- **Storage is plaintext** SQLite / IndexedDB. Field encryption + key management for shared
  devices is deferred. Sensitive forms → single-operator devices + device lock/FDE for now.
- **Box detection** thresholds suit a clean printed row/grid; real templates need per-form tuning.
- **Perso-Arabic "0"** is a small dot; the MNIST-style normalizer scales it up like any glyph.
- **Calibration** uses held-out MNIST/HODA; a real deployment re-calibrates on real scans.

## What still needs you

1. **One real scanned form + blank template** (dummy data). Phases 6–9 are validated against a
   synthetic box-row generator until then.
2. **Pakistan-specific Urdu digit dataset** (optional upgrade over HODA) — from a university partner.
3. **Pilot sector + named partner** (plan recommends education first) — and the legal/privacy
   review + Ignite application.
