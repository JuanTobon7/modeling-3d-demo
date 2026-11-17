"""Densificación: rectificación estéreo entre vistas vecinas + SGBM + fusión con chequeo multivista."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .common import ProgressFn, SparseModel, View


@dataclass
class DenseCloud:
    points: np.ndarray    # (N, 3)
    colors: np.ndarray    # (N, 3) uint8
    centers: np.ndarray   # (N, 3) centro de la cámara que vio cada punto (para orientar normales)


@dataclass
class _Cam:
    view: View
    K: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray


def convergence_point(views: list[View]) -> np.ndarray:
    """Punto más cercano a todos los ejes ópticos (el objeto en una captura 360°)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for v in views:
        d = v.axis / np.linalg.norm(v.axis)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ v.center
    return np.linalg.lstsq(A, b, rcond=None)[0]


def object_region(model: SparseModel) -> tuple[np.ndarray, np.ndarray]:
    """Caja que contiene el objeto: puntos dispersos dentro del anillo de cámaras."""
    cams = model.registered
    pts = model.points
    c0 = convergence_point(cams)
    cam_d = np.array([np.linalg.norm(v.center - c0) for v in cams])
    inside = np.linalg.norm(pts - c0, axis=1) < 0.85 * np.median(cam_d)
    if inside.sum() > 50:
        pts = pts[inside]
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    pad = 0.15 * (hi - lo)
    return lo - pad, hi + pad


def _neighbors(cams: list[_Cam], i: int, k: int = 2) -> list[int]:
    vi = cams[i].view
    scored = []
    for j, c in enumerate(cams):
        if j == i:
            continue
        ang = np.degrees(np.arccos(np.clip(vi.axis @ c.view.axis, -1, 1)))
        if 4.0 <= ang <= 35.0:
            scored.append((abs(ang - 12.0), j))
    return [j for _, j in sorted(scored)[:k]]


def _rectify(a: _Cam, b: _Cam):
    w, h = a.rgb.shape[1], a.rgb.shape[0]
    R = b.view.R @ a.view.R.T
    T = b.view.t - R @ a.view.t
    z = np.zeros(5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(a.K, z, b.K, z, (w, h), R, T.reshape(3, 1),
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2, Q


def _pair_points(a: _Cam, b: _Cam, sparse: np.ndarray, lo, hi, stride: int):
    R1, R2, P1, P2, Q = _rectify(a, b)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return None  # par vertical: SGBM solo trabaja en horizontal
    if P2[0, 3] > 0:
        return "swap"
    w, h = a.rgb.shape[1], a.rgb.shape[0]
    m1 = cv2.initUndistortRectifyMap(a.K, None, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(b.K, None, R2, P2, (w, h), cv2.CV_32FC1)
    gl = cv2.remap(a.gray, *m1, cv2.INTER_LINEAR)
    gr = cv2.remap(b.gray, *m2, cv2.INTER_LINEAR)
    cl = cv2.remap(a.rgb, *m1, cv2.INTER_LINEAR)

    # Rango de disparidad a partir de los puntos dispersos.
    Xr = (sparse @ a.view.R.T + a.view.t) @ R1.T
    Xr = Xr[Xr[:, 2] > 1e-6]
    f, B = P1[0, 0], -P2[0, 3] / P2[0, 0]
    if len(Xr) < 10:
        return None
    u = Xr[:, 0] / Xr[:, 2] * f + P1[0, 2]
    v = Xr[:, 1] / Xr[:, 2] * f + P1[1, 2]
    vis = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if vis.sum() < 10:
        return None
    d = f * B / Xr[vis, 2]
    d_lo, d_hi = np.percentile(d, 1), np.percentile(d, 99)
    min_d = int(np.floor(max(0.0, d_lo * 0.8)))
    num_d = int(np.ceil(max(16.0, d_hi * 1.2 - min_d) / 16.0) * 16)
    num_d = min(num_d, 320)

    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=min_d, numDisparities=num_d, blockSize=bs,
                                 P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dl = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    # Chequeo izquierda-derecha usando imágenes espejadas.
    dr = cv2.flip(sgbm.compute(cv2.flip(gr, 1), cv2.flip(gl, 1)), 1).astype(np.float32) / 16.0
    valid = (dl > min_d) & (gl > 0)
    ys, xs = np.nonzero(valid)
    if stride > 1:
        keep = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[keep], xs[keep]
    dd = dl[ys, xs]
    xr = np.round(xs - dd).astype(int)
    ok = (xr >= 0) & (xr < w)
    ok[ok] &= np.abs(dr[ys[ok], xr[ok]] - dd[ok]) <= 1.0
    ys, xs, dd = ys[ok], xs[ok], dd[ok]
    if len(ys) == 0:
        return None
    pts_r = cv2.perspectiveTransform(np.stack([xs, ys, dd], 1).reshape(-1, 1, 3).astype(np.float64), Q).reshape(-1, 3)
    Xc = pts_r @ R1          # = R1^T * p (marco de la cámara a)
    Xw = (Xc - a.view.t) @ a.view.R
    inside = np.all((Xw > lo) & (Xw < hi), axis=1) & (Xc[:, 2] > 0)
    return Xw[inside], cl[ys[inside], xs[inside]], Xc[inside, 2]


def _depth_map(cam: _Cam, Xw: np.ndarray) -> np.ndarray:
    h, w = cam.gray.shape
    D = np.full((h, w), np.inf, np.float32)
    if len(Xw) == 0:
        return D
    Xc = Xw @ cam.view.R.T + cam.view.t
    z = Xc[:, 2]
    good = z > 1e-6
    u = np.round(Xc[good, 0] / z[good] * cam.K[0, 0] + cam.K[0, 2]).astype(int)
    v = np.round(Xc[good, 1] / z[good] * cam.K[1, 1] + cam.K[1, 2]).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    np.minimum.at(D, (v[inb], u[inb]), z[good][inb].astype(np.float32))
    return D


def densify(model: SparseModel, progress: ProgressFn, max_side: int = 800,
            stride: int = 2, rel_tol: float = 0.015) -> DenseCloud:
    reg = model.registered
    cams = []
    for v in reg:
        h, w = v.rgb.shape[:2]
        s = min(1.0, max_side / max(w, h))
        rgb = cv2.resize(v.rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else v.rgb
        K = v.K.copy()
        K[:2] *= s
        cams.append(_Cam(v, K, rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
    lo, hi = object_region(model)

    per_view: dict[int, list] = {}
    done = set()
    for i in range(len(cams)):
        for j in _neighbors(cams, i):
            key = (min(i, j), max(i, j))
            if key in done:
                continue
            done.add(key)
            res = _pair_points(cams[i], cams[j], model.points, lo, hi, stride)
            ref = i
            if res == "swap":
                res = _pair_points(cams[j], cams[i], model.points, lo, hi, stride)
                ref = j
            if res is None or res == "swap":
                continue
            per_view.setdefault(ref, []).append(res)
        progress("dense", (i + 1) / len(cams), f"Estéreo {i + 1}/{len(cams)}")

    if not per_view:
        return DenseCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3)))

    clouds = {r: (np.concatenate([p[0] for p in L]), np.concatenate([p[1] for p in L])) for r, L in per_view.items()}
    depth = {r: _depth_map(cams[r], X) for r, (X, _) in clouds.items()}

    # Chequeo de consistencia geométrica: cada punto debe coincidir con el mapa de
    # profundidad de al menos otra vista.
    P, C, O = [], [], []
    refs = list(clouds)
    for r in refs:
        X, col = clouds[r]
        others = sorted((u for u in refs if u != r),
                        key=lambda u: -(cams[u].view.axis @ cams[r].view.axis))[:4]
        votes = np.zeros(len(X), np.int32)
        for u in others:
            cu = cams[u]
            Xc = X @ cu.view.R.T + cu.view.t
            z = Xc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                px = np.round(Xc[:, 0] / z * cu.K[0, 0] + cu.K[0, 2]).astype(np.int64)
                py = np.round(Xc[:, 1] / z * cu.K[1, 1] + cu.K[1, 2]).astype(np.int64)
            h, w = cu.gray.shape
            inb = (z > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            d = np.full(len(X), np.inf, np.float32)
            d[inb] = depth[u][py[inb], px[inb]]
            votes += (np.abs(d - z) < rel_tol * z).astype(np.int32)
        keep = votes >= 1
        P.append(X[keep])
        C.append(col[keep])
        O.append(np.repeat(cams[r].view.center[None], keep.sum(), 0))
    return DenseCloud(np.concatenate(P), np.concatenate(C).astype(np.uint8), np.concatenate(O))
