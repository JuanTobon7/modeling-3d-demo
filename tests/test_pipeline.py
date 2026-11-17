"""Reconstrucción completa con datos sintéticos, para los dos motores."""
import numpy as np
import pytest
import trimesh

from scanner3d.pipeline import preprocess, runner, sfm_colmap, sfm_opencv
from scanner3d.synthetic import make_dataset


def _align_error(est: np.ndarray, gt: np.ndarray) -> float:
    """Error RMS de centros de cámara tras alinear con una similitud (Umeyama), relativo al radio."""
    me, mg = est.mean(0), gt.mean(0)
    E, G = est - me, gt - mg
    U, S, Vt = np.linalg.svd(G.T @ E)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (E ** 2).sum()
    aligned = s * E @ R.T + mg
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()) / np.linalg.norm(G, axis=1).mean())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    centers = make_dataset(d, views=24, width=640, height=480)
    return d, np.array(centers)


ENGINES = ["opencv"] + (["colmap"] if sfm_colmap.available() else [])


@pytest.mark.parametrize("engine", ENGINES)
def test_sfm_recovers_cameras(dataset, engine):
    d, gt = dataset
    views = preprocess.load_views(d)
    mod = sfm_colmap if engine == "colmap" else sfm_opencv
    model = mod.reconstruct(views, lambda *a: None, 4000)
    reg = [k for k, v in enumerate(views) if v.registered]
    assert len(reg) >= 22
    est = np.array([views[k].center for k in reg])
    assert _align_error(est, gt[reg]) < 0.02


@pytest.mark.parametrize("engine", ENGINES)
def test_full_pipeline(dataset, tmp_path, engine):
    d, _ = dataset
    res = runner.run(d, tmp_path, engine=engine, quality="fast")
    assert res["engine"] == engine
    assert res["registered"] >= 22
    for f in ("model.glb", "model.obj", "model.ply", "points.ply", "cameras.json"):
        assert (tmp_path / f).exists(), f
    mesh = trimesh.load(tmp_path / "model.glb", force="mesh")
    assert len(mesh.faces) > 5000
    ext = mesh.bounds[1] - mesh.bounds[0]
    assert ext.max() == pytest.approx(1.0, abs=0.01)   # normalizado
    # "Arriba" = +Y: las cámaras (elevación ~28°) quedan por encima del suelo del modelo (y=0).
    import json
    cams = json.loads((tmp_path / "cameras.json").read_text())
    assert all(c["center"][1] > 0.2 for c in cams)
