"""Detección SIFT (pirámide de convoluciones gaussianas) y emparejamiento verificado."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .common import ProgressFn, View

# USAC (OpenCV >= 4.5) es mucho más rápido que RANSAC clásico con muchos outliers.
_RANSAC = getattr(cv2, "USAC_FAST", cv2.RANSAC)


@dataclass
class Features:
    kps: np.ndarray    # (N, 2) float32
    desc: np.ndarray   # (N, 128) float32


@dataclass
class PairMatch:
    i: int
    j: int
    idx: np.ndarray    # (M, 2) índices de keypoints (en i, en j) inliers
    E: np.ndarray


def detect(views: list[View], n_features: int, progress: ProgressFn) -> list[Features]:
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    out = []
    for k, v in enumerate(views):
        gray = cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            kp, desc = [], np.zeros((0, 128), np.float32)
        # RootSIFT: mejora el emparejamiento sin coste extra.
        desc = desc.astype(np.float32)
        desc /= desc.sum(axis=1, keepdims=True) + 1e-7
        desc = np.sqrt(desc)
        pts = np.array([p.pt for p in kp], np.float32).reshape(-1, 2)
        out.append(Features(pts, desc))
        progress("features", (k + 1) / len(views), f"SIFT {v.name}: {len(pts)} puntos")
    return out


def candidate_pairs(n: int, window: int = 8, exhaustive_max: int = 40) -> list[tuple[int, int]]:
    if n <= exhaustive_max:
        return list(combinations(range(n), 2))
    # Captura en 360°: vecinos en orden circular + algunos saltos largos.
    pairs = set()
    for i in range(n):
        for d in range(1, window + 1):
            j = (i + d) % n
            pairs.add((min(i, j), max(i, j)))
        for d in (n // 4, n // 2):
            j = (i + d) % n
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def match_all(views: list[View], feats: list[Features], progress: ProgressFn,
              min_inliers: int = 30) -> list[PairMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = candidate_pairs(len(views))
    matches: list[PairMatch] = []
    for k, (i, j) in enumerate(pairs):
        fi, fj = feats[i], feats[j]
        if len(fi.kps) < 20 or len(fj.kps) < 20:
            continue
        knn = matcher.knnMatch(fi.desc, fj.desc, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < min_inliers:
            continue
        idx = np.array([(m.queryIdx, m.trainIdx) for m in good], np.int32)
        # Elimina emparejamientos muchos-a-uno.
        _, uniq = np.unique(idx[:, 1], return_index=True)
        idx = idx[np.sort(uniq)]
        p1 = cv2.undistortPoints(fi.kps[idx[:, 0]].reshape(-1, 1, 2), views[i].K, None)
        p2 = cv2.undistortPoints(fj.kps[idx[:, 1]].reshape(-1, 1, 2), views[j].K, None)
        thr = 1.5 / views[i].K[0, 0]
        E, mask = cv2.findEssentialMat(p1, p2, np.eye(3), _RANSAC, 0.999, thr)
        if E is None or mask is None:
            continue
        if E.shape[0] > 3:
            E = E[:3]
        inl = mask.ravel().astype(bool)
        if inl.sum() >= min_inliers:
            matches.append(PairMatch(i, j, idx[inl], E))
        if k % 10 == 0 or k == len(pairs) - 1:
            progress("matching", (k + 1) / len(pairs), f"Pares {k + 1}/{len(pairs)}")
    return matches
