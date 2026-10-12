"""FastAPI application factory.

Run locally:
    uvicorn api.main:app --reload
OpenAPI docs at /docs.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers import analytics, documents, models, monitoring, reviews, system
from api.services import ConflictError, ExtractionFailed
from database.models import RequestMetric
from database.session import get_session
from src.config import Settings, get_settings
from src.ingestion.validation import ValidationError
from src.logging_utils import configure_logging, get_logger, request_id_var
from src.pipeline.model_service import ModelService

log = get_logger("api")

VALIDATION_STATUS = {"file_too_large": 413, "unsupported_extension": 415, "unsupported_content": 415,
                     "extension_mismatch": 415}


def _error(status: int, error: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "message": message,
                                                     "request_id": request_id_var.get(), "details": details})


def create_app(settings: Settings | None = None, model_service: ModelService | None = None,
               session_factory=None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        if app.state.models is None:
            app.state.models = ModelService.load(settings)
        log.info("api_started", extra={"models": app.state.models.status})
        yield

    app = FastAPI(
        title="Document Risk Engine API",
        version="1.0.0",
        description="Enterprise Document Risk Intelligence & Compliance Engine - vendor contracts (V1). "
                    "Authenticate with the `X-API-Key` header.",
        lifespan=lifespan,
    )
    if session_factory is None:
        from database.session import get_sessionmaker

        session_factory = get_sessionmaker()
    app.state.settings = settings
    app.state.models = model_service
    app.state.session_factory = session_factory

    def _session_dep():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_dep

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            latency = round((time.perf_counter() - start) * 1000, 2)
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            log.info("http_request", extra={"method": request.method, "route": path, "status": status,
                                            "latency_ms": latency})
            if request.method != "OPTIONS" and not path.startswith(("/docs", "/openapi")):
                try:
                    with app.state.session_factory() as s:
                        s.add(RequestMetric(method=request.method, route=path, status_code=status,
                                            latency_ms=latency, request_id=rid))
                        s.commit()
                except Exception:  # metrics must never break a request
                    pass
            request_id_var.reset(token)

    @app.exception_handler(ValidationError)
    async def upload_validation(_, exc: ValidationError):
        return _error(VALIDATION_STATUS.get(exc.code, 422), exc.code, exc.message)

    @app.exception_handler(ConflictError)
    async def conflict(_, exc: ConflictError):
        return _error(409, "duplicate_document", str(exc), {"existing_document_id": exc.existing_id})

    @app.exception_handler(ExtractionFailed)
    async def extraction_failed(_, exc: ExtractionFailed):
        return _error(422, "extraction_failed", f"Text could not be extracted: {exc}")

    @app.exception_handler(OperationalError)
    async def db_down(_, exc: OperationalError):
        log.error("database_unavailable", extra={"error": str(exc)[:300]})
        return _error(503, "database_unavailable", "The database is temporarily unavailable. Please retry.")

    @app.exception_handler(SQLAlchemyError)
    async def db_error(_, exc: SQLAlchemyError):
        log.exception("database_error")
        return _error(500, "database_error", "A database error occurred.")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            return _error(exc.status_code, detail.get("error", "error"), detail.get("message", ""), detail.get("details"))
        return JSONResponse(status_code=exc.status_code, headers=getattr(exc, "headers", None),
                            content={"error": "http_error", "message": str(detail), "request_id": request_id_var.get()})

    @app.exception_handler(RequestValidationError)
    async def request_validation(_, exc: RequestValidationError):
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return _error(422, "invalid_request", "Request validation failed", {"errors": errors})

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        log.exception("unhandled_error")
        return _error(500, "internal_error", "An unexpected error occurred. Quote the request_id when reporting it.")

    for r in (system.router, documents.router, reviews.router, analytics.router, models.router, monitoring.router):
        app.include_router(r)
    return app


app = create_app()
