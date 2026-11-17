"""Dataset sintético: un objeto con textura renderizado (ray casting en NumPy) desde N ángulos.

Sirve para probar el pipeline de punta a punta sin celular:
    python -m scanner3d.synthetic salida/ --views 36
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_rng = np.random.default_rng(7)
_LATTICE = _rng.random((6, 64, 64, 64)).astype(np.float32)


def _value_noise(p: np.ndarray, ch: int) -> np.ndarray:
    g = _LATTICE[ch]
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out = out + w * g[(i[:, 0] + dx) % 64, (i[:, 1] + dy) % 64, (i[:, 2] + dz) % 64]
    return out


def _fbm(p: np.ndarray, ch: int) -> np.ndarray:
    v, amp, tot = 0, 1.0, 0
    for octave in range(4):
        v = v + amp * _value_noise(p * (5.0 * 2 ** octave) + 17.3 * ch, ch)
        tot += amp
        amp *= 0.55
    return v / tot


def _texture(p: np.ndarray, kind: np.ndarray) -> np.ndarray:
    col = np.zeros((len(p), 3), np.float32)
    n = np.stack([_fbm(p, c) for c in range(3)], 1)
    n = np.clip((n - 0.5) * 2.2 + 0.5, 0, 1)
    obj = kind == 1
    col[obj] = n[obj] * np.array([1.0, 0.75, 0.55]) + 0.1
    sph = kind == 2
    col[sph] = n[sph][:, [2, 0, 1]] * np.array([0.6, 0.9, 1.0]) + 0.05
    gnd = kind == 3
    chk = ((np.floor(p[gnd, 0] * 4) + np.floor(p[gnd, 2] * 4)) % 2)[:, None]
    col[gnd] = 0.25 + 0.35 * chk + 0.4 * n[gnd][:, :1] * np.array([0.9, 0.9, 0.8])
    return np.clip(col, 0, 1)


def _intersect(C: np.ndarray, d: np.ndarray):
    n = len(d)
    t_best = np.full(n, np.inf)
    kind = np.zeros(n, np.int8)
    normal = np.zeros((n, 3))
    # Caja (objeto principal)
    lo, hi = np.array([-0.45, 0.0, -0.3]), np.array([0.45, 0.6, 0.3])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - C) / d
        t2 = (hi - C) / d
    tmin = np.nanmax(np.minimum(t1, t2), 1)
    tmax = np.nanmin(np.maximum(t1, t2), 1)
    hit = (tmax >= tmin) & (tmin > 1e-6)
    t_best[hit] = tmin[hit]
    kind[hit] = 1
    axis = np.argmax(np.minimum(t1, t2), 1)
    nb = np.zeros((n, 3))
    nb[np.arange(n), axis] = -np.sign(d[np.arange(n), axis])
    normal[hit] = nb[hit]
    # Esfera encima
    c, r = np.array([0.0, 0.85, 0.0]), 0.3
    oc = C - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    ts = -b - np.sqrt(np.maximum(disc, 0))
    hs = (disc > 0) & (ts > 1e-6) & (ts < t_best)
    t_best[hs] = ts[hs]
    kind[hs] = 2
    normal[hs] = (C + ts[hs, None] * d[hs] - c) / r
    # Plato (disco en y=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = -C[1] / d[:, 1]
    pg = C + tg[:, None] * d
    hg = (d[:, 1] < 0) & (tg < t_best) & (pg[:, 0] ** 2 + pg[:, 2] ** 2 < 1.5 ** 2)
    t_best[hg] = tg[hg]
    kind[hg] = 3
    normal[hg] = [0, 1, 0]
    return t_best, kind, normal


def look_at(C: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # filas: ejes de cámara en mundo (OpenCV: y hacia abajo)


def render(C, R, f, w, h) -> np.ndarray:
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_c = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1).reshape(-1, 3)
    d = dirs_c @ R
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, kind, nrm = _intersect(C, d)
    img = np.zeros((len(d), 3), np.float32)
    sky = kind == 0
    img[sky] = (0.55 + 0.25 * np.clip(-d[sky, 1:2], -1, 1)) * np.array([0.8, 0.85, 0.9])
    hit = ~sky
    p = C + t[hit, None] * d[hit]
    light = np.array([0.4, 0.8, 0.3])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(nrm[hit] @ light, 0, 1)
    img[hit] = _texture(p, kind[hit]) * shade[:, None]
    return (np.clip(img, 0, 1).reshape(h, w, 3) * 255).astype(np.uint8)


def make_dataset(out: Path, views: int = 36, width: int = 960, height: int = 720,
                 focal35: int = 30, radius: float = 2.6, elevation_deg: float = 28.0) -> list[np.ndarray]:
    out.mkdir(parents=True, exist_ok=True)
    f = focal35 / 36.0 * max(width, height)
    target = np.array([0.0, 0.45, 0.0])
    centers = []
    el = np.radians(elevation_deg)
    for k in range(views):
        az = 2 * np.pi * k / views
        el_k = el + np.radians(6) * np.sin(3 * az)   # algo de variación vertical, como a mano
        C = target + radius * np.array([np.cos(el_k) * np.sin(az), np.sin(el_k), np.cos(el_k) * np.cos(az)])
        R = look_at(C, target)
        img = Image.fromarray(render(C, R, f, width, height)).filter(ImageFilter.GaussianBlur(0.6))
        exif = Image.Exif()
        exif[0x8769] = {41989: focal35}
        exif[271] = "scanner3d-synthetic"
        img.save(out / f"view_{k:03d}.jpg", quality=93, exif=exif)
        centers.append(C)
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--views", type=int, default=36)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    make_dataset(a.out, a.views, a.width, a.height)
    print(f"{a.views} imágenes en {a.out}")


if __name__ == "__main__":
    main()
