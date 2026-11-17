"""API REST (FastAPI) + servidor de la PWA para el celular."""
from __future__ import annotations

import io
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .jobs import JobManager
from .pipeline.runner import QUALITY, engines_available

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MAX_IMAGES = int(os.environ.get("SCANNER_MAX_IMAGES", 300))
MAX_BYTES = int(os.environ.get("SCANNER_MAX_MB", 30)) * 1024 * 1024
ALLOWED = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}


class JobCreate(BaseModel):
    name: str = Field("", max_length=80)
    engine: str = Field("auto", pattern="^(auto|colmap|opencv)$")
    quality: str = Field("normal", pattern="^(fast|normal|high)$")


def create_app(data_dir: Path | None = None, token: str | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("SCANNER_DATA", ROOT / "data"))
    token = token if token is not None else os.environ.get("SCANNER_TOKEN", "")
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["jobs"] = JobManager(data_dir / "jobs")
        yield
        state["jobs"].shutdown()

    app = FastAPI(title="Scanner3D", version="1.0", lifespan=lifespan)

    def auth(x_token: str | None = Header(None), token_q: str | None = Query(None, alias="token")):
        if token and not secrets.compare_digest(x_token or token_q or "", token):
            raise HTTPException(401, "Token inválido")

    def jobs() -> JobManager:
        return state["jobs"]

    def job_or_404(job_id: str) -> dict:
        meta = jobs().get(job_id)
        if meta is None:
            raise HTTPException(404, "Trabajo no encontrado")
        return meta

    api = [Depends(auth)]

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/info", dependencies=api)
    def info():
        return {"engines": engines_available(), "qualities": list(QUALITY),
                "max_images": MAX_IMAGES, "max_mb": MAX_BYTES // (1024 * 1024)}

    @app.get("/api/jobs", dependencies=api)
    def list_jobs():
        return jobs().list()

    @app.post("/api/jobs", dependencies=api, status_code=201)
    def create_job(body: JobCreate):
        return jobs().create(body.name, body.engine, body.quality)

    @app.get("/api/jobs/{job_id}", dependencies=api)
    def get_job(job_id: str):
        return job_or_404(job_id)

    @app.post("/api/jobs/{job_id}/images", dependencies=api)
    async def upload(job_id: str, files: list[UploadFile] = File(...)):
        meta = job_or_404(job_id)
        if meta["status"] not in ("created", "failed", "cancelled"):
            raise HTTPException(409, "El trabajo ya está en proceso")
        folder = jobs().image_dir(job_id)
        existing = len(list(folder.iterdir()))
        accepted, rejected = [], []
        for f in files:
            data = await f.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                rejected.append({"file": f.filename, "error": "demasiado grande"})
                continue
            try:
                with Image.open(io.BytesIO(data)) as im:
                    fmt = im.format
                    im.verify()
            except Exception:
                rejected.append({"file": f.filename, "error": "no es una imagen válida"})
                continue
            if fmt not in ALLOWED:
                rejected.append({"file": f.filename, "error": f"formato {fmt} no soportado"})
                continue
            if existing + len(accepted) >= MAX_IMAGES:
                rejected.append({"file": f.filename, "error": f"máximo {MAX_IMAGES} fotos"})
                continue
            name = f"img_{existing + len(accepted):04d}{ALLOWED[fmt]}"
            (folder / name).write_bytes(data)
            accepted.append(name)
        total = jobs().update_image_count(job_id)
        return {"accepted": len(accepted), "rejected": rejected, "total": total}

    @app.post("/api/jobs/{job_id}/start", dependencies=api)
    def start(job_id: str):
        meta = job_or_404(job_id)
        if meta["status"] in ("queued", "running"):
            raise HTTPException(409, "Ya está en proceso")
        if meta["images"] < 3:
            raise HTTPException(400, "Se necesitan al menos 3 fotos")
        return jobs().start(job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=api)
    def cancel(job_id: str):
        job_or_404(job_id)
        return jobs().cancel(job_id)

    @app.delete("/api/jobs/{job_id}", dependencies=api, status_code=204)
    def delete(job_id: str):
        job_or_404(job_id)
        jobs().delete(job_id)

    @app.get("/api/jobs/{job_id}/files/{name}", dependencies=api)
    def download(job_id: str, name: str):
        job_or_404(job_id)
        path = jobs().output_path(job_id, name)
        if path is None:
            raise HTTPException(404, "Archivo no encontrado")
        media = {".glb": "model/gltf-binary", ".json": "application/json"}.get(path.suffix, "application/octet-stream")
        return FileResponse(path, media_type=media, filename=f"{job_id}_{name}")

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
