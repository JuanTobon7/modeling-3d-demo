"""Orquestación del pipeline completo: fotos -> cámaras -> nube densa -> malla -> archivos."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import dense, export, mesh, preprocess, sfm_colmap, sfm_opencv
from .common import PipelineError, ProgressFn, SparseModel

QUALITY = {
    "fast":   dict(max_side=1200, dense_side=640,  poisson=8,  features=4000),
    "normal": dict(max_side=1600, dense_side=800,  poisson=9,  features=6000),
    "high":   dict(max_side=2400, dense_side=1100, poisson=10, features=10000),
}

# Peso de cada etapa en la barra de progreso global.
STAGES = [("load", 0.03), ("features", 0.07), ("matching", 0.15), ("sfm", 0.25),
          ("dense", 0.30), ("mesh", 0.15), ("export", 0.05)]


def _global_progress(progress: ProgressFn) -> ProgressFn:
    offsets, acc = {}, 0.0
    for name, w in STAGES:
        offsets[name] = (acc, w)
        acc += w

    def cb(stage: str, frac: float, msg: str) -> None:
        start, w = offsets.get(stage, (0.0, 0.0))
        progress(stage, min(0.999, start + w * max(0.0, min(1.0, frac))), msg)
    return cb


def engines_available() -> list[str]:
    return (["colmap"] if sfm_colmap.available() else []) + ["opencv"]


def run(images_dir: Path, out_dir: Path, engine: str = "auto", quality: str = "normal",
        progress: ProgressFn | None = None) -> dict:
    progress = progress or (lambda *a: None)
    p = _global_progress(progress)
    q = QUALITY.get(quality, QUALITY["normal"])
    t0 = time.time()
    log: list[str] = []

    p("load", 0.0, "Cargando fotos")
    views = preprocess.load_views(images_dir, q["max_side"])
    p("load", 1.0, f"{len(views)} fotos cargadas")

    order = {"auto": ["colmap", "opencv"], "colmap": ["colmap"], "opencv": ["opencv"]}.get(engine, ["opencv"])
    order = [e for e in order if e in engines_available()] or ["opencv"]
    model: SparseModel | None = None
    for k, name in enumerate(order):
        try:
            if k > 0:
                views = preprocess.load_views(images_dir, q["max_side"])  # estado limpio
            mod = sfm_colmap if name == "colmap" else sfm_opencv
            model = mod.reconstruct(views, p, q["features"])
            if len(model.registered) < 3:
                raise PipelineError("Menos de 3 cámaras ubicadas.")
            break
        except Exception as e:
            msg = f"Motor {name} falló: {e}"
            log.append(msg)
            print(msg)
            traceback.print_exc()
            model = None
            if k == len(order) - 1:
                raise PipelineError(msg) from e
            p("sfm", 0.0, f"{msg}. Probando motor {order[k + 1]}…")
    assert model is not None
    reg = model.registered
    p("sfm", 1.0, f"{len(reg)}/{len(views)} cámaras ubicadas con {model.engine}")

    cloud = dense.densify(model, p, max_side=q["dense_side"])
    if len(cloud.points) < 2000:
        log.append("Nube densa pobre: se usa también la nube dispersa.")
        import numpy as np
        # Para la nube dispersa, orienta hacia la cámara más cercana.
        cc = np.array([v.center for v in reg])
        nearest = np.argmin(((model.points[:, None] - cc[None]) ** 2).sum(-1), 1)
        centers = cc[nearest]
        cloud = dense.DenseCloud(np.vstack([cloud.points, model.points]),
                                 np.vstack([cloud.colors, model.colors]),
                                 np.vstack([cloud.centers, centers]))
    p("dense", 1.0, f"Nube densa: {len(cloud.points)} puntos")

    m, pcd = mesh.build_mesh(cloud, p, depth=q["poisson"])
    p("export", 0.2, "Exportando GLB / OBJ / PLY")
    stats = export.export_all(m, pcd, reg, out_dir)
    p("export", 1.0, "Listo")
    return {
        "engine": model.engine,
        "images": len(views),
        "registered": len(reg),
        "unregistered": [v.name for v in views if not v.registered],
        "sparse_points": int(len(model.points)),
        "dense_points": int(len(cloud.points)),
        "focal_px": round(float(reg[0].K[0, 0]), 1),
        "seconds": round(time.time() - t0, 1),
        "log": log,
        **stats,
    }
