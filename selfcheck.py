"""Run every module's built-in self-check. No pytest, no fixtures.

    python selfcheck.py
"""
import runpy
import sys

MODULES = ["raqam.data", "raqam.mlp", "raqam.cnn", "raqam.triage", "raqam.segment",
           "raqam.store", "raqam.dreams", "raqam.pipeline", "raqam.evaluate"]

fail = 0
for m in MODULES:
    print(f"\n=== {m} ===")
    try:
        runpy.run_module(m, run_name="__main__")
    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f"FAIL: {e!r}")

print(f"\n{'ALL OK' if not fail else f'{fail} FAILED'}")
sys.exit(1 if fail else 0)
