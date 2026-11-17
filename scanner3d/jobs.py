"""Gestor de trabajos: cada reconstrucción corre en un proceso aparte (cancelable, aislado)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

FINAL = {"done", "failed", "cancelled"}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _worker(job_dir: str, engine: str, quality: str) -> None:
    """Se ejecuta en un proceso hijo: escribe progress.json y result.json."""
    from .pipeline import runner
    from .pipeline.common import PipelineError

    d = Path(job_dir)
    prog = d / "progress.json"
    last = {"t": 0.0}

    def progress(stage: str, frac: float, msg: str) -> None:
        now = time.time()
        if now - last["t"] > 0.3 or frac >= 0.999:
            last["t"] = now
            _write_json(prog, {"stage": stage, "progress": round(frac, 4), "message": msg})

    try:
        result = runner.run(d / "images", d / "output", engine, quality, progress)
        _write_json(d / "output" / "result.json", result)
        _write_json(prog, {"stage": "done", "progress": 1.0, "message": "Modelo listo", "status": "done"})
    except PipelineError as e:
        _write_json(prog, {"stage": "error", "progress": 1.0, "message": str(e), "status": "failed"})
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        _write_json(prog, {"stage": "error", "progress": 1.0,
                           "message": f"Error interno: {e}", "status": "failed"})


class JobManager:
    def __init__(self, root: Path, max_concurrent: int = 1):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self._lock = threading.RLock()
        self._procs: dict[str, mp.process.BaseProcess] = {}
        self._queue: list[str] = []
        self._ctx = mp.get_context("spawn")
        self._stop = threading.Event()
        # Trabajos que quedaron "corriendo" tras reiniciar el servidor se marcan como fallidos.
        for d in self.root.iterdir():
            meta = self._read(d.name)
            if meta and meta["status"] in ("queued", "running"):
                meta["status"] = "failed"
                meta["message"] = "El servidor se reinició durante el proceso."
                self._save(meta)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- disco
    def _dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            return self.root / "__invalid__"
        return self.root / job_id

    def _read(self, job_id: str) -> dict | None:
        p = self._dir(job_id) / "job.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save(self, meta: dict) -> None:
        _write_json(self._dir(meta["id"]) / "job.json", meta)

    # ------------------------------------------------------------------ API
    def create(self, name: str, engine: str, quality: str) -> dict:
        job_id = uuid.uuid4().hex[:12]
        d = self._dir(job_id)
        (d / "images").mkdir(parents=True)
        (d / "output").mkdir()
        meta = {"id": job_id, "name": name or f"Escaneo {time.strftime('%Y-%m-%d %H:%M')}",
                "engine": engine, "quality": quality, "status": "created",
                "created": time.time(), "images": 0, "progress": 0.0, "stage": "",
                "message": "Sube las fotos", "result": None}
        self._save(meta)
        return meta

    def get(self, job_id: str) -> dict | None:
        meta = self._read(job_id)
        if meta is None:
            return None
        if meta["status"] == "running":
            p = self._dir(job_id) / "progress.json"
            if p.exists():
                try:
                    meta.update({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                                 if k in ("stage", "progress", "message")})
                except Exception:
                    pass
        meta["files"] = self.files(job_id) if meta["status"] == "done" else []
        return meta

    def list(self) -> list[dict]:
        jobs = [self.get(d.name) for d in self.root.iterdir() if d.is_dir()]
        return sorted([j for j in jobs if j], key=lambda j: -j["created"])

    def image_dir(self, job_id: str) -> Path:
        return self._dir(job_id) / "images"

    def update_image_count(self, job_id: str) -> int:
        with self._lock:
            meta = self._read(job_id)
            n = len([p for p in self.image_dir(job_id).iterdir() if p.is_file()])
            meta["images"] = n
            self._save(meta)
            return n

    def files(self, job_id: str) -> list[str]:
        out = self._dir(job_id) / "output"
        return sorted(p.name for p in out.iterdir() if p.is_file()) if out.exists() else []

    def output_path(self, job_id: str, name: str) -> Path | None:
        if name not in self.files(job_id):
            return None
        return self._dir(job_id) / "output" / name

    def start(self, job_id: str) -> dict:
        with self._lock:
            meta = self._read(job_id)
            meta.update(status="queued", progress=0.0, stage="queued", message="En cola", result=None)
            self._save(meta)
            (self._dir(job_id) / "progress.json").unlink(missing_ok=True)
            self._queue.append(job_id)
            return meta

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            meta = self._read(job_id)
            if job_id in self._queue:
                self._queue.remove(job_id)
            proc = self._procs.pop(job_id, None)
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(5)
            if meta["status"] not in FINAL:
                meta.update(status="cancelled", message="Cancelado")
                self._save(meta)
            return meta

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            for job_id in list(self._procs):
                self.cancel(job_id)

    # --------------------------------------------------------- planificador
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for job_id, proc in list(self._procs.items()):
                    if not proc.is_alive():
                        self._procs.pop(job_id)
                        self._finish(job_id, proc.exitcode)
                while self._queue and len(self._procs) < self.max_concurrent:
                    job_id = self._queue.pop(0)
                    meta = self._read(job_id)
                    if meta is None:
                        continue
                    proc = self._ctx.Process(target=_worker, daemon=True,
                                             args=(str(self._dir(job_id)), meta["engine"], meta["quality"]))
                    proc.start()
                    self._procs[job_id] = proc
                    meta.update(status="running", message="Iniciando", started=time.time())
                    self._save(meta)
            time.sleep(0.25)

    def _finish(self, job_id: str, exitcode: int | None) -> None:
        meta = self._read(job_id)
        if meta is None or meta["status"] in FINAL:
            return
        prog = {}
        p = self._dir(job_id) / "progress.json"
        if p.exists():
            try:
                prog = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        res = self._dir(job_id) / "output" / "result.json"
        if prog.get("status") == "done" and res.exists():
            meta.update(status="done", progress=1.0, stage="done", message="Modelo listo",
                        result=json.loads(res.read_text(encoding="utf-8")))
        else:
            meta.update(status="failed", stage="error",
                        message=prog.get("message") or f"El proceso terminó con código {exitcode}")
        meta["finished"] = time.time()
        self._save(meta)
