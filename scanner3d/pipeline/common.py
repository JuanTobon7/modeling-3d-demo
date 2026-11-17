"""Estructuras compartidas por todas las etapas del pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


class PipelineError(RuntimeError):
    """Error controlado: el mensaje se muestra al usuario."""


ProgressFn = Callable[[str, float, str], None]  # (etapa, fraccion 0..1, mensaje)


def no_progress(stage: str, frac: float, msg: str) -> None:  # pragma: no cover
    pass


@dataclass
class View:
    name: str
    rgb: np.ndarray            # (H, W, 3) uint8
    K: np.ndarray              # (3, 3) intrínsecos a la resolución de trabajo
    focal_from_exif: bool
    R: Optional[np.ndarray] = None   # (3, 3) mundo -> cámara
    t: Optional[np.ndarray] = None   # (3,)

    @property
    def registered(self) -> bool:
        return self.R is not None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def axis(self) -> np.ndarray:
        """Dirección óptica en coordenadas de mundo."""
        return self.R[2]

    def projection(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


@dataclass
class SparseModel:
    views: list[View]
    points: np.ndarray                 # (N, 3)
    colors: np.ndarray                 # (N, 3) uint8
    engine: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def registered(self) -> list[View]:
        return [v for v in self.views if v.registered]
