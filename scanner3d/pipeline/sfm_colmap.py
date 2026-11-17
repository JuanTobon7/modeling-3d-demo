"""Motor SfM basado en COLMAP (pycolmap, licencia BSD)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .common import PipelineError, ProgressFn, SparseModel, View


def available() -> bool:
    try:
        import pycolmap  # noqa: F401
        return True
    except Exception:
        return False


def _cam_from_world(img) -> np.ndarray:
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return np.asarray(cfw.matrix())


def reconstruct(views: list[View], progress: ProgressFn, n_features: int = 6000) -> SparseModel:
    import pycolmap

    try:  # menos ruido en la consola
        pycolmap.logging.minloglevel = 1
    except Exception:
        pass
    work = Path(tempfile.mkdtemp(prefix="colmap_"))
    try:
        img_dir = work / "images"
        img_dir.mkdir()
        names = []
        for k, v in enumerate(views):
            name = f"{k:04d}.jpg"
            cv2.imwrite(str(img_dir / name), cv2.cvtColor(v.rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
        db = work / "database.db"
        K = views[0].K
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        reader.camera_params = f"{K[0, 0]},{K[0, 2]},{K[1, 2]},0"
        extraction = pycolmap.FeatureExtractionOptions()
        try:
            extraction.sift.max_num_features = n_features
        except AttributeError:
            pass
        progress("features", 0.1, "COLMAP: extrayendo SIFT")
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction,
                                  device=pycolmap.Device.cpu)
        progress("matching", 0.1, "COLMAP: emparejando")
        if len(views) <= 60:
            pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = 12
            pairing.quadratic_overlap = True
            pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)
        progress("sfm", 0.1, "COLMAP: reconstrucción incremental")
        opts = pycolmap.IncrementalPipelineOptions()
        opts.num_threads = -1
        counter = {"n": 0}

        def on_image():
            counter["n"] += 1
            progress("sfm", min(1.0, counter["n"] / len(views)),
                     f"COLMAP: {counter['n']}/{len(views)} cámaras")

        out = work / "sparse"
        out.mkdir()
        recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts, next_image_callback=on_image)
        if not recs:
            raise PipelineError("COLMAP no pudo reconstruir la escena.")
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        if rec.num_reg_images() < 3:
            raise PipelineError("COLMAP ubicó menos de 3 cámaras.")

        index = {n: k for k, n in enumerate(names)}
        for img in rec.images.values():
            if not img.has_pose:
                continue
            k = index[img.name]
            cam = rec.cameras[img.camera_id]
            v = views[k]
            v.K = np.asarray(cam.calibration_matrix(), np.float64)
            params = np.asarray(cam.params)
            if cam.model.name == "SIMPLE_RADIAL" and abs(params[3]) > 1e-4:
                # Quita la distorsión radial para que el resto del pipeline use pinhole.
                v.rgb = cv2.undistort(v.rgb, v.K, np.array([params[3], 0, 0, 0]))
            M = _cam_from_world(img)
            v.R, v.t = M[:, :3].copy(), M[:, 3].copy()
        pts = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
        cols = np.array([p.color for p in rec.points3D.values()], np.uint8).reshape(-1, 3)
        return SparseModel(views, pts, cols, engine="colmap")
    finally:
        shutil.rmtree(work, ignore_errors=True)
