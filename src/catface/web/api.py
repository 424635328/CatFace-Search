"""HTTP interface: a thin translation layer over :mod:`catface.web.service`.

Design rules this file follows
------------------------------
* **No business logic here.** Sizing, ranking and readiness live in the service, so the API cannot
  drift from what the tests exercise.
* **The response models are the contract.** They are declared as pydantic classes rather than
  returned as loose dicts, so the OpenAPI schema at ``/docs`` is generated from the same declaration
  the tests assert on, and a field rename cannot silently change the wire format.
* **Refusals are explicit and explain themselves.** An oversized upload, a non-image file, or a
  missing checkpoint each produce a message naming the cause and the limit, because the alternative
  — a generic 500 — leaves the operator guessing whether the service is broken or the input is.
* **Readiness is a separate question from liveness.** ``/healthz`` answers "is the process up" and
  must not depend on model load; ``/api/status`` answers "can it serve a search", which does.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .. import __version__
from ..errors import CatFaceError
from ..logging_utils import configure_utf8_console, get_logger
from .service import ALLOWED_IMAGE_SUFFIXES, MAX_UPLOAD_BYTES, QueryRejectedError, SearchService

LOGGER = get_logger("catface.web.api")

WEB_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"

#: Evidence surfaced in the UI. A retrieval demo that only shows a confident-looking ranking
#: misleads: this system's own error analysis found that every remaining failure sits on a near tie,
#: so the margin and the label-quality caveat are part of the product, not a footnote.
LIMITATIONS: list[dict[str, str]] = [
    {
        "title": "相似度不等于正确率",
        "body": "排名第一不等于答案正确。本项目在 503 个未参与训练的身份上实测 hit@1 = 0.9682，"
        "其余错例的“正确身份相似度 − 最强错误身份相似度”中位数只有 −0.0569，"
        "即全部是贴边的误判。请把 margin 当作可信度，而不是把相似度当作概率。",
    },
    {
        "title": "语料存在同图双标签",
        "body": "评测语料中有 137 对“同一张照片挂在两个身份标签下”。因此同一只猫可能以两个不同"
        "身份名出现在结果里，而两者其实都是对的。详见 docs/diagnostics/LABEL-COLLISIONS.md。",
    },
    {
        "title": "身份训练会削弱通用视觉能力",
        "body": "身份度量学习会压缩通用视觉组织：物种可分间隔下降 64%，品种 1-NN 从 0.9506 降到 "
        "0.7881。不要把这个描述子当作通用特征提取器使用。",
    },
]


# ------------------------------------------------------------------------------------------------
# wire contract
# ------------------------------------------------------------------------------------------------
class MatchModel(BaseModel):
    rank: int = Field(description="1-based rank in the returned list")
    image_id: str
    identity: str
    similarity: float = Field(description="cosine similarity, higher is closer")
    path: str = Field(description="path as recorded in the manifest")
    identity_consensus: float = Field(description="share of the returned matches carrying this identity")


class TimingModel(BaseModel):
    embedding_ms: float
    search_ms: float


class SearchResponse(BaseModel):
    predicted_identity: str | None
    top_similarity: float | None
    margin: float | None = Field(
        description="top-1 similarity minus the best different-identity similarity; the honest "
        "confidence signal, since every measured failure sat on a near tie"
    )
    matches: list[MatchModel]
    descriptor_dim: int
    timing: TimingModel


class StatusResponse(BaseModel):
    ready: bool
    checkpoint: str
    manifest: str
    device: str
    image_size: int
    gallery_images: int
    gallery_identities: int
    descriptor_dim: int
    backend: str | None
    tta: list[str]
    load_seconds: float
    version: str


class IdentityCount(BaseModel):
    identity: str
    images: int


class ErrorResponse(BaseModel):
    detail: str
    kind: str


def _api_key_guard(request: Request) -> None:
    """Enforce an API key when one is configured.

    Unset means open, which is the right default for a laptop demo and the wrong one for anything
    reachable from a network. The check is here rather than in middleware so that ``/healthz``
    stays answerable by a load balancer that has no credentials.
    """
    expected = os.environ.get("CATFACE_API_KEY")
    if not expected:
        return
    provided = request.headers.get("x-api-key") or request.headers.get("authorization", "")
    if provided.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def create_app(service: SearchService | None = None) -> FastAPI:
    """Build the application. Accepting a service makes the app testable without a real model."""
    configure_utf8_console()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Load the model once per process, on startup.

        A lifespan handler rather than the deprecated ``on_event`` hook: the handler form is what
        the installed FastAPI supports without a deprecation warning, and it scopes the load to the
        process instead of to a request.
        """
        instance: SearchService | None = getattr(application.state, "service", None)
        if instance is None:
            LOGGER.warning("no SearchService configured; /api/search will return 503")
        else:
            try:
                instance.load()
            except CatFaceError as error:
                # Report and keep serving: /healthz stays up so an orchestrator can tell "the
                # process is dead" from "the model is not loadable", and /api/status says which.
                LOGGER.error("model failed to load, service is not ready: %s", error)
        yield

    app = FastAPI(
        title="CatFace Search",
        version=__version__,
        lifespan=lifespan,
        description=(
            "猫脸个体识别检索。排名为余弦相似度的精确穷举扫描（非近似索引），因此报告的数字不依赖索引调参。"
        ),
    )
    app.state.service = service
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    api = APIRouter()

    def require_service() -> SearchService:
        instance: SearchService | None = getattr(app.state, "service", None)
        if instance is None:
            raise HTTPException(
                status_code=503,
                detail="no search service is configured; start the app with "
                "`python -m catface.web` or pass a SearchService",
            )
        return instance

    # -- liveness: must not touch the model -------------------------------------------------
    @api.get("/healthz", summary="liveness probe")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.get("/api/status", response_model=StatusResponse, summary="readiness and model facts")
    def status() -> StatusResponse:
        instance = require_service()
        payload = instance.status()
        return StatusResponse(version=__version__, **payload)

    @api.get("/api/identities", response_model=list[IdentityCount], summary="gallery identities")
    def identities(limit: int = 500) -> list[IdentityCount]:
        instance = require_service()
        counts = instance.identity_index()
        return [IdentityCount(identity=name, images=count) for name, count in list(counts.items())[:limit]]

    @api.get("/api/gallery/{image_id}", summary="serve one gallery image")
    def gallery_image(image_id: str) -> FileResponse:
        instance = require_service()
        record = instance.record_for(image_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no gallery entry with id {image_id!r}")
        resolved = instance.resolve_image(record.path)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail="this gallery entry exists but its image file is not available to this deployment",
            )
        return FileResponse(resolved)

    @api.post(
        "/api/search",
        response_model=SearchResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        summary="identify the cat in an uploaded photograph",
    )
    async def search(
        request: Request,
        file: UploadFile = File(..., description="one cat photograph"),
        top_k: int = Form(10, ge=1, le=50),
        identity_aggregation: bool = Form(False),
    ) -> SearchResponse:
        _api_key_guard(request)
        instance = require_service()

        suffix = Path(file.filename or "query").suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file extension {suffix!r}; accepted: "
                f"{', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}",
            )

        # Stream to a bounded temporary file instead of reading the body into memory: the size
        # limit is enforced while writing, so an oversized upload never becomes a large allocation.
        staging = Path(tempfile.mkdtemp(prefix="catface-query-"))
        try:
            target = staging / f"query{suffix}"
            written = 0
            with target.open("wb") as handle:
                while chunk := await file.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                        )
                    handle.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail="the uploaded file is empty")
            await file.close()

            try:
                outcome = instance.search(target, top_k=top_k, identity_aggregation=identity_aggregation)
            except QueryRejectedError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return SearchResponse(**outcome.to_dict())

    # -- the page ---------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        instance = getattr(app.state, "service", None)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "limitations": LIMITATIONS,
                "status": instance.status() if instance else {"ready": False},
            },
        )

    app.include_router(api)

    @app.exception_handler(CatFaceError)
    async def catface_error(_request: Request, exc: CatFaceError) -> JSONResponse:
        """Project errors are the operator's problem, not the client's: 500 with the real reason."""
        LOGGER.error("service error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc), "kind": type(exc).__name__})

    return app


def env_default_checkpoint() -> str:
    return os.environ.get("CATFACE_CHECKPOINT", "artifacts/train/dinov2s-arcface/best.pt")


def env_default_manifest() -> str:
    return os.environ.get("CATFACE_MANIFEST", "data/manifests/cat_individuals_manifest.jsonl")


def env_default_device() -> str:
    return os.environ.get("CATFACE_DEVICE", "cpu")


__all__ = [
    "LIMITATIONS",
    "create_app",
    "env_default_checkpoint",
    "env_default_device",
    "env_default_manifest",
]
