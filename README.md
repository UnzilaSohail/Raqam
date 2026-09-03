# Raqam (رقم)

Offline handwritten-digit digitizer for paper forms. From-scratch NumPy engine,
confidence triage — every low-confidence digit is flagged for a human, never
silently guessed.

Covers Phases 1–6 of [project-raqam.md](project-raqam.md): the learning engine
(MLP, live training, "dreams", draw-and-predict) plus the confidence-triage form
scanner. Phase 7 (CNN + Urdu-Indic numerals), Phase 8 (Android) and the pilots
are not here — they need datasets, form templates and a partner (see bottom).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/mac: .venv/bin/pip
.venv/Scripts/python -m raqam.train                  # downloads MNIST, ~10s, saves models/mnist_mlp.npz
.venv/Scripts/python -m raqam.cli serve              # http://127.0.0.1:8000
```

`python selfcheck.py` runs every module's built-in check.

## What's where

| Plan phase | Module | Notes |
|---|---|---|
| 1 – network core | `raqam/mlp.py` | ReLU MLP, hand-written backprop, SGD+momentum. Grad-checked. 97.8% test. |
| 2 – live training viz | `raqam/train.py`, `/api/train` SSE | loss + val-accuracy streamed to the browser chart |
| 3 – "dreams" | `raqam/dreams.py` | gradient ascent on the input per class |
| 4 – draw & predict | web canvas → `/api/predict` | canvas is centered by mass, MNIST-style |
| 5 – confidence triage | `raqam/triage.py` | temperature-scaled softmax + threshold routing |
| 6 – scanner pipeline | `raqam/segment.py`, `raqam/pipeline.py` | OpenCV deskew → box detection → 28×28 cells → triage → CSV/XLSX |
| 5 – offline queue | `raqam/store.py` | local SQLite, stores only a SHA-256 of the source image |

## CLI

```bash
python -m raqam.cli scan form.jpg --form marksheet --field roll_no
python -m raqam.cli list                 # pending review
python -m raqam.cli review 3 12345       # resolve a flagged record
python -m raqam.cli export out.xlsx
```

No real form? `raqam.segment.synth_form([1,2,3,4,5,6])` renders a printed box
row with real MNIST glyphs to test the pipeline end to end.

## Deliberate shortcuts (see `ponytail:` comments)

- **`store.py`** — records stored as plaintext SQLite. Field-level encryption +
  key management is a Phase-8 concern (shared field phone). Today's guarantee:
  the raw image never lands on disk, only its hash.
- **`segment.py`** — box-detection thresholds are tuned for a clean printed
  row/grid. Real templates need per-form tuning; that knob is intentional.
- **Calibration** — temperature is fit on a held-out slice of MNIST test.
  A real deployment calibrates on real scanned samples of *that* form.

## What this still needs from you

1. **One real form** — a scanned marksheet / tally sheet + its blank template,
   dummy data only. The segmentation step can't be made real without it.
2. **Urdu-Indic numeral dataset** — pick one, check its licence. Needed before
   Phase 7.
3. **Pilot sector + partner** — the plan recommends education first.
4. **Legal/privacy review + Ignite application** — yours, not code.
