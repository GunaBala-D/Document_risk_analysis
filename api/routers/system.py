from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.auth import ROLE_PERMISSIONS, get_current_user
from api.schemas import UserOut
from database.models import User
from src.llm.ollama_client import OllamaExplainer

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request):
    """Liveness + dependency status. Returns 200 when the core (DB) works, 503 otherwise.
    Model/LLM/OCR problems are reported as degraded, because the API still serves rule-based results."""
    settings, models = request.app.state.settings, request.app.state.models
    checks: dict = {}
    try:
        with request.app.state.session_factory() as s:
            s.execute(text("SELECT 1"))
        checks["database"] = {"status": "up"}
    except Exception as exc:
        checks["database"] = {"status": "down", "error": type(exc).__name__}
    checks["models"] = models.status
    checks["ocr"] = {"status": "up" if (settings.tesseract_cmd or shutil.which("tesseract")) else "down"}
    checks["llm"] = OllamaExplainer(settings.ollama_base_url, settings.ollama_model).health() \
        if settings.llm_enabled else {"status": "disabled"}

    core_ok = checks["database"]["status"] == "up"
    degraded = (models.classifier is None or models.anomaly is None or checks["ocr"]["status"] != "up"
                or checks["llm"]["status"] not in ("up", "disabled"))
    status = "down" if not core_ok else "degraded" if degraded else "ok"
    return JSONResponse({"status": status, "checks": checks}, status_code=200 if core_ok else 503)


@router.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, username=user.username, full_name=user.full_name, role=user.role,
                   permissions=sorted(ROLE_PERMISSIONS.get(user.role, set())))
