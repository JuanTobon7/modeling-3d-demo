"""Structure-from-Motion incremental propio (OpenCV + SciPy).

Pasos: par inicial -> PnP-RANSAC por imagen -> triangulación -> bundle adjustment.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .common import PipelineError, ProgressFn, SparseModel, View
from .features import Features, PairMatch, detect, match_all

REPROJ_PX = 4.0
MIN_ANGLE_DEG = 1.5


def _rotate(rvecs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    dot = np.sum(k * pts, axis=1, keepdims=True)
    return pts * cos + np.cross(k, pts) * sin + k * dot * (1 - cos)


def _angles_deg(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
    a = X - c1
    b = X - c2
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _reproj_err(view: View, X: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ view.R.T + view.t
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (Xc @ view.K.T)[:, :2] / z[:, None]
    return np.linalg.norm(p - x, axis=1), z


class IncrementalSfM:
    def __init__(self, views: list[View], feats: list[Features], matches: list[PairMatch],
                 progress: ProgressFn):
        self.views, self.feats, self.progress = views, feats, progress
        self.adj: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for m in matches:
            self.adj[m.i].append((m.j, m.idx))
            self.adj[m.j].append((m.i, m.idx[:, ::-1].copy()))
        self.matches = matches
        self.point_of = [np.full(len(f.kps), -1, np.int64) for f in feats]
        self.X: list[np.ndarray] = []
        self.obs: list[list[tuple[int, int]]] = []
        self.alive: list[bool] = []
        self.optimize_focal = not all(v.focal_from_exif for v in views)

    # ------------------------------------------------------------------ puntos
    def _add_point(self, X: np.ndarray, observations: list[tuple[int, int]]) -> None:
        pid = len(self.X)
        self.X.append(X)
        self.obs.append(observations)
        self.alive.append(True)
        for img, kp in observations:
            self.point_of[img][kp] = pid

    def _kill_point(self, pid: int) -> None:
        self.alive[pid] = False
        for img, kp in self.obs[pid]:
            if self.point_of[img][kp] == pid:
                self.point_of[img][kp] = -1
        self.obs[pid] = []

    # -------------------------------------------------------------- par inicial
    def initialize(self) -> None:
        best = None
        for m in sorted(self.matches, key=lambda m: -len(m.idx))[:40]:
            vi, vj = self.views[m.i], self.views[m.j]
            x1 = self.feats[m.i].kps[m.idx[:, 0]].astype(np.float64)
            x2 = self.feats[m.j].kps[m.idx[:, 1]].astype(np.float64)
            n1 = cv2.undistortPoints(x1.reshape(-1, 1, 2), vi.K, None).reshape(-1, 2)
            n2 = cv2.undistortPoints(x2.reshape(-1, 1, 2), vj.K, None).reshape(-1, 2)
            _, R, t, mask = cv2.recoverPose(m.E, n1, n2, np.eye(3))
            t = t.ravel()
            vi.R, vi.t = np.eye(3), np.zeros(3)
            vj.R, vj.t = R, t
            X = cv2.triangulatePoints(vi.projection(), vj.projection(), x1.T, x2.T)
            X = (X[:3] / X[3]).T
            e1, z1 = _reproj_err(vi, X, x1)
            e2, z2 = _reproj_err(vj, X, x2)
            ang = _angles_deg(vi.center, vj.center, X)
            ok = (z1 > 0) & (z2 > 0) & (e1 < REPROJ_PX) & (e2 < REPROJ_PX) & (ang > MIN_ANGLE_DEG)
            n_ok = int(ok.sum())
            med = float(np.median(ang[ok])) if n_ok else 0.0
            score = n_ok * min(med, 8.0)
            if n_ok >= 50 and med >= 2.0 and (best is None or score > best[0]):
                best = (score, m, R, t, X, ok)
            vi.R = vj.R = vi.t = vj.t = None
        if best is None:
            raise PipelineError("No se encontró un par inicial con suficiente paralaje. "
                                "Toma las fotos moviéndote alrededor del objeto, con solape.")
        _, m, R, t, X, ok = best
        self.views[m.i].R, self.views[m.i].t = np.eye(3), np.zeros(3)
        self.views[m.j].R, self.views[m.j].t = R, t
        for r in np.nonzero(ok)[0]:
            self._add_point(X[r], [(m.i, int(m.idx[r, 0])), (m.j, int(m.idx[r, 1]))])
        self.bundle_adjust(iters=30)

    # -------------------------------------------------------------- registrar
    def _correspondences(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        kp_list, pid_list = [], []
        for other, idx in self.adj[i]:
            if not self.views[other].registered:
                continue
            pids = self.point_of[other][idx[:, 1]]
            sel = pids >= 0
            kp_list.append(idx[sel, 0])
            pid_list.append(pids[sel])
        if not kp_list:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        kp = np.concatenate(kp_list)
        pid = np.concatenate(pid_list)
        kp, first = np.unique(kp, return_index=True)
        return kp, pid[first]

    def register(self, i: int) -> bool:
        kp, pid = self._correspondences(i)
        if len(kp) < 20:
            return False
        v = self.views[i]
        obj = np.array([self.X[p] for p in pid], np.float64)
        img = self.feats[i].kps[kp].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, v.K, None, iterationsCount=3000,
                                                 reprojectionError=REPROJ_PX, confidence=0.999,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < max(15, 0.2 * len(kp)):
            return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], v.K, None, rvec, tvec)
        v.R, v.t = cv2.Rodrigues(rvec)[0], tvec.ravel()
        for r in inl:
            p = int(pid[r])
            if self.point_of[i][kp[r]] < 0:
                self.point_of[i][kp[r]] = p
                self.obs[p].append((i, int(kp[r])))
        self._triangulate_new(i)
        return True

    def _triangulate_new(self, i: int) -> None:
        vi = self.views[i]
        for other, idx in self.adj[i]:
            vo = self.views[other]
            if not vo.registered:
                continue
            pa = self.point_of[i][idx[:, 0]]
            pb = self.point_of[other][idx[:, 1]]
            # Extiende tracks existentes si la reproyección es buena.
            for src_has, a_img, a_col, b_img, b_col, pids in (
                (pb >= 0) & (pa < 0), i, 0, other, 1, pb), ((pa >= 0) & (pb < 0), other, 1, i, 0, pa):
                rows = np.nonzero(src_has)[0]
                if len(rows) == 0:
                    continue
                X = np.array([self.X[p] for p in pids[rows]])
                va = self.views[a_img]
                e, z = _reproj_err(va, X, self.feats[a_img].kps[idx[rows, a_col]])
                for r in rows[(e < REPROJ_PX) & (z > 0)]:
                    kp_a = int(idx[r, a_col])
                    p = int(pids[r])
                    if self.point_of[a_img][kp_a] < 0 and self.alive[p]:
                        self.point_of[a_img][kp_a] = p
                        self.obs[p].append((a_img, kp_a))
            # Triangula los que no tienen punto en ninguna de las dos.
            pa = self.point_of[i][idx[:, 0]]
            pb = self.point_of[other][idx[:, 1]]
            rows = np.nonzero((pa < 0) & (pb < 0))[0]
            if len(rows) == 0:
                continue
            x1 = self.feats[i].kps[idx[rows, 0]].astype(np.float64)
            x2 = self.feats[other].kps[idx[rows, 1]].astype(np.float64)
            X = cv2.triangulatePoints(vi.projection(), vo.projection(), x1.T, x2.T)
            X = (X[:3] / X[3]).T
            e1, z1 = _reproj_err(vi, X, x1)
            e2, z2 = _reproj_err(vo, X, x2)
            ang = _angles_deg(vi.center, vo.center, X)
            ok = (z1 > 0) & (z2 > 0) & (e1 < REPROJ_PX) & (e2 < REPROJ_PX) & (ang > MIN_ANGLE_DEG)
            for r, X_r in zip(rows[ok], X[ok]):
                self._add_point(X_r, [(i, int(idx[r, 0])), (other, int(idx[r, 1]))])

    # -------------------------------------------------------- bundle adjustment
    def bundle_adjust(self, iters: int = 40, with_focal: bool = False) -> None:
        cams = [k for k, v in enumerate(self.views) if v.registered]
        fixed = cams[0]
        free = cams[1:]
        cam_pos = {c: n for n, c in enumerate(free)}
        pids = [p for p, a in enumerate(self.alive) if a and len(self.obs[p]) >= 2]
        if not pids or not free:
            return
        pt_pos = {p: n for n, p in enumerate(pids)}
        ci, pi, xy = [], [], []
        for p in pids:
            for img, kp in self.obs[p]:
                ci.append(img)
                pi.append(pt_pos[p])
                xy.append(self.feats[img].kps[kp])
        ci, pi, xy = np.array(ci), np.array(pi), np.array(xy, np.float64)
        n_obs, n_free, n_pts = len(ci), len(free), len(pids)
        K = self.views[fixed].K
        f0, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        rv_fixed = cv2.Rodrigues(self.views[fixed].R)[0].ravel()
        t_fixed = self.views[fixed].t
        x0 = [np.zeros(1)] if with_focal else []
        for c in free:
            x0.append(cv2.Rodrigues(self.views[c].R)[0].ravel())
            x0.append(self.views[c].t)
        x0.append(np.array([self.X[p] for p in pids]).ravel())
        x0 = np.concatenate(x0)
        off = 1 if with_focal else 0
        is_fixed = ci == fixed
        slot = np.array([cam_pos.get(c, 0) for c in ci])

        def residuals(x):
            lf = x[0] if with_focal else 0.0
            cp = x[off:off + 6 * n_free].reshape(-1, 6)
            P = x[off + 6 * n_free:].reshape(-1, 3)
            rv = np.where(is_fixed[:, None], rv_fixed, cp[slot, :3])
            tv = np.where(is_fixed[:, None], t_fixed, cp[slot, 3:])
            Xc = _rotate(rv, P[pi]) + tv
            f = f0 * np.exp(lf)
            z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
            u = f * Xc[:, 0] / z + cx
            w = f * Xc[:, 1] / z + cy
            return np.concatenate([u - xy[:, 0], w - xy[:, 1]])

        rr, cc = [], []
        rows = np.arange(n_obs)
        for half in (0, n_obs):
            r = rows + half
            if with_focal:
                rr.append(r)
                cc.append(np.zeros(n_obs, np.int64))
            fr = r[~is_fixed]
            for k in range(6):
                rr.append(fr)
                cc.append(off + 6 * slot[~is_fixed] + k)
            for k in range(3):
                rr.append(r)
                cc.append(off + 6 * n_free + 3 * pi + k)
        rr, cc = np.concatenate(rr), np.concatenate(cc)
        A = coo_matrix((np.ones(len(rr), np.uint8), (rr, cc)), shape=(2 * n_obs, len(x0))).tocsr()
        res = least_squares(residuals, x0, jac_sparsity=A, loss="huber", f_scale=2.0,
                            x_scale="jac", method="trf", max_nfev=iters, ftol=1e-6, xtol=1e-8)
        x = res.x
        if with_focal:
            s = float(np.exp(x[0]))
            if 0.5 < s < 2.0:
                for v in self.views:
                    v.K = v.K.copy()
                    v.K[0, 0] *= s
                    v.K[1, 1] *= s
        cp = x[off:off + 6 * n_free].reshape(-1, 6)
        for c, n in cam_pos.items():
            self.views[c].R = cv2.Rodrigues(cp[n, :3])[0]
            self.views[c].t = cp[n, 3:].copy()
        P = x[off + 6 * n_free:].reshape(-1, 3)
        for p, n in pt_pos.items():
            self.X[p] = P[n]
        self._filter_points()

    def _filter_points(self) -> None:
        for p, a in enumerate(self.alive):
            if not a:
                continue
            obs = self.obs[p]
            keep = []
            for img, kp in obs:
                e, z = _reproj_err(self.views[img], self.X[p][None], self.feats[img].kps[kp][None])
                if e[0] < REPROJ_PX and z[0] > 0:
                    keep.append((img, kp))
                elif self.point_of[img][kp] == p:
                    self.point_of[img][kp] = -1
            self.obs[p] = keep
            if len(keep) < 2:
                self._kill_point(p)

    # -------------------------------------------------------------- principal
    def run(self) -> None:
        self.initialize()
        n = len(self.views)
        failed: dict[int, int] = defaultdict(int)
        since_ba = 0
        while True:
            cand = [k for k in range(n) if not self.views[k].registered and failed[k] < 3]
            if not cand:
                break
            scored = sorted(cand, key=lambda k: -len(self._correspondences(k)[0]))
            progressed = False
            for k in scored[:3]:
                if self.register(k):
                    progressed = True
                    since_ba += 1
                    reg = sum(v.registered for v in self.views)
                    self.progress("sfm", reg / n, f"Cámara {self.views[k].name} ubicada ({reg}/{n})")
                    if since_ba >= 4:
                        reg_n = sum(v.registered for v in self.views)
                        self.bundle_adjust(iters=20, with_focal=self.optimize_focal and reg_n >= 8)
                        since_ba = 0
                    break
                failed[k] += 1
            if not progressed and all(failed[k] >= 3 for k in cand):
                break
        self.bundle_adjust(iters=60, with_focal=self.optimize_focal)
        self.bundle_adjust(iters=30)

    def model(self) -> SparseModel:
        pts, cols = [], []
        for p, a in enumerate(self.alive):
            if not a:
                continue
            img, kp = self.obs[p][0]
            x, y = self.feats[img].kps[kp]
            rgb = self.views[img].rgb
            cols.append(rgb[min(int(y), rgb.shape[0] - 1), min(int(x), rgb.shape[1] - 1)])
            pts.append(self.X[p])
        return SparseModel(self.views, np.array(pts).reshape(-1, 3),
                           np.array(cols, np.uint8).reshape(-1, 3), engine="opencv")


def reconstruct(views: list[View], progress: ProgressFn, n_features: int = 6000) -> SparseModel:
    feats = detect(views, n_features, progress)
    matches = match_all(views, feats, progress)
    if not matches:
        raise PipelineError("Las fotos no comparten suficientes puntos en común.")
    sfm = IncrementalSfM(views, feats, matches, progress)
    sfm.run()
    return sfm.model()
