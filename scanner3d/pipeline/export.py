"""Alinea el modelo (arriba = +Y, centrado, escala ~1) y exporta GLB / OBJ / PLY."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from .common import View


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def normalization(mesh: o3d.geometry.TriangleMesh, views: list[View]) -> tuple[np.ndarray, np.ndarray, float]:
    up = -np.mean([v.R[1] for v in views], axis=0)   # -y de cámara = arriba
    R = _rot_between(up, np.array([0.0, 1.0, 0.0]))
    V = np.asarray(mesh.vertices) @ R.T
    lo, hi = V.min(0), V.max(0)
    center = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])  # apoyado en y=0
    scale = 1.0 / max(float((hi - lo).max()), 1e-9)
    return R, center, scale


def export_all(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, views: list[View],
               out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    R, center, scale = normalization(mesh, views)

    def tf(X):
        return (np.asarray(X) @ R.T - center) * scale

    V = tf(mesh.vertices)
    F = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors)
    if len(cols) != len(V):
        cols = np.full((len(V), 3), 0.7)
    rgba = np.hstack([(np.clip(cols, 0, 1) * 255).astype(np.uint8), np.full((len(V), 1), 255, np.uint8)])
    tm = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=rgba, process=False)
    _ = tm.vertex_normals  # fuerza el cálculo para que el GLB incluya NORMAL
    tm.export(out / "model.glb")
    tm.export(out / "model.obj")
    tm.export(out / "model.ply")
    pc_cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    trimesh.PointCloud(tf(pcd.points), colors=pc_cols).export(out / "points.ply")
    cams = [{"name": v.name, "center": tf(v.center[None])[0].round(5).tolist()} for v in views]
    (out / "cameras.json").write_text(json.dumps(cams))
    return {"vertices": int(len(V)), "triangles": int(len(F)), "points": int(len(pcd.points))}
