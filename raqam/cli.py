"""Raqam CLI.

    python -m raqam.cli scan form.jpg --form marksheet --field roll_no
    python -m raqam.cli list
    python -m raqam.cli review 3 12345
    python -m raqam.cli export out.csv
    python -m raqam.cli serve
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import store
from .pipeline import annotate, digitize, export_csv, export_xlsx


def _scan(a):
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read image: {a.image}")
    rec = digitize(img, a.form, a.field, threshold=a.threshold)
    out = Path(a.image).with_name(Path(a.image).stem + "_review.png")
    cv2.imwrite(str(out), annotate(rec))
    flag = "  ** NEEDS REVIEW **" if rec["needs_review"] else ""
    print(f"#{rec['id']}  {a.field} = {rec['value']}{flag}")
    for c in rec["cells"]:
        mark = "?" if c["flagged"] else " "
        print(f"  {mark} {c['digit']}  conf={c['confidence']:.3f}")
    print(f"review image: {out}")


def _list(_a):
    p = store.pending()
    if not p:
        print("nothing pending review")
    for r in p:
        print(f"#{r['id']}  {r['form']}/{r['field']} = {r['value']}  ({r['img_sha256'][:12]})")


def _review(a):
    store.resolve(a.id, a.value)
    print(f"#{a.id} resolved -> {a.value}")


def _export(a):
    fn = export_xlsx if a.path.lower().endswith(".xlsx") else export_csv
    print("wrote", fn(a.path))


def _serve(a):
    import uvicorn
    uvicorn.run("raqam.web.app:app", host=a.host, port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="raqam")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan"); s.add_argument("image")
    s.add_argument("--form", default="form"); s.add_argument("--field", default="field")
    s.add_argument("--threshold", type=float, default=0.95); s.set_defaults(fn=_scan)

    sub.add_parser("list").set_defaults(fn=_list)

    r = sub.add_parser("review"); r.add_argument("id", type=int)
    r.add_argument("value"); r.set_defaults(fn=_review)

    e = sub.add_parser("export"); e.add_argument("path"); e.set_defaults(fn=_export)

    v = sub.add_parser("serve"); v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000); v.set_defaults(fn=_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
