from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
    details: dict | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_format: str
    size_bytes: int
    page_count: int | None
    processing_status: str


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    uploaded_at: datetime
    document_type: str | None
    processing_status: str
    vendor_name: str | None
    review_status: str
    risk_score: int | None = None
    risk_level: str | None = None
    extraction_method: str | None = None


class DocumentList(BaseModel):
    total: int
    items: list[DocumentSummary]


class EntityOut(BaseModel):
    name: str
    value: str | float | None
    source_text: str | None
    confidence: float
    method: str


class DocumentDetail(BaseModel):
    id: str
    filename: str
    file_format: str
    mime_type: str
    file_size: int
    uploaded_at: datetime
    uploaded_by: str | None
    document_type: str | None
    doc_type_confidence: float | None
    page_count: int | None
    extraction_method: str | None
    extraction_status: str
    ocr_pages: int | None
    processing_status: str
    status_message: str | None
    vendor_name: str | None
    review_status: str
    sections: list[dict] | None
    entities: list[EntityOut]
    timings_ms: dict | None
    latest_prediction_id: int | None


class RiskReport(BaseModel):
    """Business risk report. Every value comes from the stored analysis result."""

    document_id: str
    prediction_id: int
    analyzed_at: datetime
    risk_score: int
    risk_level: RiskLevel
    ml_predicted_class: str | None
    ml_probabilities: dict | None
    high_risk_probability: float | None
    anomaly_score: float | None
    score_components: dict
    score_contributions: dict | None
    floors_applied: list | None
    compliance_violations: list[dict]
    anomalies: list[dict]
    top_risk_factors: list[str]
    recommended_actions: list[dict]
    needs_review: bool
    review_reasons: list | None
    degraded: bool
    component_errors: dict | None
    model_version: str | None
    model_versions: dict | None
    timings_ms: dict | None


class ExplanationOut(BaseModel):
    document_id: str
    prediction_id: int
    shap: dict
    llm_status: str
    llm_explanation: dict | None
    llm_model: str | None
    llm_latency_ms: float | None
    llm_error: str | None
    recommended_actions: list[dict]


class FeedbackIn(BaseModel):
    decision: Literal["confirm", "correct"]
    corrected_label: RiskLevel | None = None
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def check(self):
        if self.decision == "correct":
            if self.corrected_label is None:
                raise ValueError("corrected_label is required when decision is 'correct'")
            if not self.comment or len(self.comment.strip()) < 5:
                raise ValueError("A comment (>= 5 characters) explaining the correction is required")
        if self.decision == "confirm" and self.corrected_label is not None:
            raise ValueError("corrected_label must be empty when decision is 'confirm'")
        return self


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    document_id: str
    prediction_id: int
    reviewer_id: int
    created_at: datetime
    model_prediction: str | None
    model_risk_level: str
    model_risk_score: int
    model_version: str | None
    decision: str
    reviewer_label: str
    comment: str | None
    validation_status: str
    validation_flags: list | None
    validated_by: int | None = None
    validated_at: datetime | None = None
    validation_comment: str | None = None


class ValidateIn(BaseModel):
    approve: bool
    comment: str | None = Field(default=None, max_length=2000)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    full_name: str
    role: str
    permissions: list[str] = []
