from __future__ import annotations

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from database.models import Document, Prediction
from src.config import Settings
from src.pipeline.model_service import ModelService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_models(request: Request) -> ModelService:
    return request.app.state.models


def get_session_factory(request: Request):
    return request.app.state.session_factory


def get_document_or_404(session: Session, document_id: str) -> Document:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, f"Document {document_id} not found")
    return doc


def get_latest_prediction_or_404(session: Session, doc: Document) -> Prediction:
    if doc.processing_status == "unsupported_type":
        raise HTTPException(409, doc.status_message or "Document type is not supported in V1")
    if doc.latest_prediction_id is None:
        raise HTTPException(409, f"Document {doc.id} has not been analyzed yet (status: {doc.processing_status})")
    pred = session.get(Prediction, doc.latest_prediction_id)
    if pred is None:
        raise HTTPException(404, "Prediction not found")
    return pred
