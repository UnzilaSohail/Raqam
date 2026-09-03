# Deploying Raqam

Three ways, smallest first. All are offline-capable once the assets are cached.

## 1. Field phone / desktop — the PWA (no install, no store)

Run the server anywhere reachable by the device (a laptop on the same wifi, a Pi, a VPS):

```bash
pip install -r requirements.txt
python -m raqam.train                                   # MLP, ~10s
python -m raqam.train --model cnn --data numerals --epochs 8   # CNN, ~12 min, one time
python -m raqam.cli serve --host 0.0.0.0 --port 8000
```

On the phone/laptop: open `http://<host>:8000`, then **browser menu → Install app** /
**Add to Home Screen**. After the first load it works with the network off — scanning,
the review queue, and export all run in the page. Records sync back when it's online.

Sub-$100 Android with Chrome is the target. Desktop Chrome/Edge install it the same way.

## 2. Server — Docker

```bash
docker compose run --rm trainer          # trains the CNN into ./models (one time)
docker compose up -d                      # serves on :8000, restarts on boot
```

Volumes persist `models/`, `data/`, and the synced-records SQLite. Put it behind a
reverse proxy with TLS for real deployments (sync must be HTTPS).

## 3. Raspberry Pi

Same as (1). Pure NumPy inference — no GPU, no ML runtime. A Pi 4 serves the PWA to
a handful of field devices on a local network and holds the sync queue.

## Updating the recognizer

`models/numerals_cnn.json` is what the PWA runs. After retraining, redeploy so the
service worker picks up the new `raqam-v*` cache (bump the version in `sw.js`).

## What each piece needs

| Piece | Needs network? |
|---|---|
| Scan a field, review, export (in the PWA) | no |
| First load / install | yes, once |
| Sync records to server | yes, when you choose |
| Engine tab (train / dreams demo) | yes |
