"""Reconstrucción sin servidor:  python -m scanner3d.cli carpeta_fotos salida/ [--engine opencv]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .pipeline import runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--engine", default="auto", choices=["auto", "colmap", "opencv"])
    ap.add_argument("--quality", default="normal", choices=list(runner.QUALITY))
    a = ap.parse_args()
    t0 = time.time()
    last = {"msg": None}

    def progress(stage, frac, msg):
        if msg != last["msg"]:
            print(f"[{time.time() - t0:6.1f}s {frac * 100:5.1f}%] {stage:9s} {msg}", flush=True)
            last["msg"] = msg

    result = runner.run(a.images, a.out, a.engine, a.quality, progress)
    (a.out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
