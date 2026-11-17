"""Carga de imágenes, corrección de orientación EXIF y estimación de la focal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import PipelineError, View

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Tags EXIF
_EXIF_IFD = 0x8769
_FOCAL_35MM = 41989
_FOCAL_MM = 37386

# Focal por defecto (en múltiplos del lado mayor) cuando no hay EXIF:
# ~equivale a 28-30 mm en full frame, típico de un celular.
DEFAULT_FOCAL_FACTOR = 0.85


def focal_from_exif(img: Image.Image) -> float | None:
    """Focal en píxeles relativa al lado mayor (f / max(w,h)) o None."""
    try:
        exif = img.getexif()
        sub = exif.get_ifd(_EXIF_IFD) if exif else {}
        f35 = sub.get(_FOCAL_35MM) or exif.get(_FOCAL_35MM)
        if f35 and float(f35) > 0:
            return float(f35) / 36.0
    except Exception:
        pass
    return None


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_views(folder: Path, max_side: int = 1600) -> list[View]:
    paths = list_images(folder)
    if len(paths) < 3:
        raise PipelineError(f"Se necesitan al menos 3 fotos (hay {len(paths)}).")
    views: list[View] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                ff = focal_from_exif(im)
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                s = min(1.0, max_side / max(w, h))
                if s < 1.0:
                    im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
                rgb = np.asarray(im, dtype=np.uint8).copy()
        except Exception as e:  # imagen corrupta: se omite
            print(f"[preprocess] omitida {p.name}: {e}")
            continue
        h, w = rgb.shape[:2]
        f = (ff or DEFAULT_FOCAL_FACTOR) * max(w, h)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        views.append(View(name=p.name, rgb=rgb, K=K, focal_from_exif=ff is not None))
    if len(views) < 3:
        raise PipelineError("Menos de 3 fotos válidas.")
    sizes = {v.size for v in views}
    if len(sizes) > 1:
        # Normaliza todas al tamaño más común para que la cámara sea única.
        from collections import Counter
        target = Counter(v.size for v in views).most_common(1)[0][0]
        for v in views:
            if v.size != target:
                im = Image.fromarray(v.rgb).resize(target, Image.LANCZOS)
                sx, sy = target[0] / v.size[0], target[1] / v.size[1]
                v.rgb = np.asarray(im)
                v.K = v.K.copy()
                v.K[0] *= sx
                v.K[1] *= sy
    return views
