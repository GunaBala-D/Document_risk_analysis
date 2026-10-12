"""SQLAlchemy ORM models (PostgreSQL in deployment; SQLite supported for local tests).

Raw document text is NOT stored in the database or audit logs - only the uploaded file
(on disk, access-controlled), extracted structured fields and analysis results.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    full_name: Mapped[str] = mapped_column(String(160))
    role: Mapped[str] = mapped_column(String(40))  # see api/auth.py ROLE_PERMISSIONS
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    filename: Mapped[str] = mapped_column(String(200))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_size: Mapped[int] = mapped_column(Integer)
    mime_type: Mapped[str] = mapped_column(String(120))
    file_format: Mapped[str] = mapped_column(String(10))
    storage_path: Mapped[str] = mapped_column(String(400))
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    document_type: Mapped[str | None] = mapped_column(String(30))
    doc_type_confidence: Mapped[float | None] = mapped_column(Float)
    page_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int | None] = mapped_column(Integer)
    extraction_method: Mapped[str | None] = mapped_column(String(20))
    extraction_status: Mapped[str] = mapped_column(String(20), default="pending")
    ocr_pages: Mapped[int | None] = mapped_column(Integer)
    # uploaded | processing | analyzed | unsupported_type | failed
    processing_status: Mapped[str] = mapped_column(String(30), default="uploaded", index=True)
    status_message: Mapped[str | None] = mapped_column(Text)
    sections: Mapped[list | None] = mapped_column(JSONType)
    timings_ms: Mapped[dict | None] = mapped_column(JSONType)

    vendor_name: Mapped[str | None] = mapped_column(String(200), index=True)
    vendor_key: Mapped[str | None] = mapped_column(String(200), index=True)
    # not_required | pending | confirmed | corrected
    review_status: Mapped[str] = mapped_column(String(20), default="not_required", index=True)
    latest_prediction_id: Mapped[int | None] = mapped_column(Integer)

    entities: Mapped[list[DocumentEntity]] = relationship(back_populates="document", cascade="all, delete-orphan")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="document", cascade="all, delete-orphan",
                                                         foreign_keys="Prediction.document_id")


class DocumentEntity(Base):
    __tablename__ = "document_entities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(60))
    value_text: Mapped[str | None] = mapped_column(Text)
    value_num: Mapped[float | None] = mapped_column(Float)
    value_date: Mapped[date | None] = mapped_column(Date)
    source_text: Mapped[str | None] = mapped_column(Text)  # the matched span only, not the document
    confidence: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(60))
    span_start: Mapped[int | None] = mapped_column(Integer)
    span_end: Mapped[int | None] = mapped_column(Integer)
    document: Mapped[Document] = relationship(back_populates="entities")


class DocumentFeatures(Base):
    __tablename__ = "document_features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    feature_version: Mapped[str] = mapped_column(String(20))
    features: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    model_name: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(20))
    anomaly_model_version: Mapped[str | None] = mapped_column(String(20))
    versions: Mapped[dict | None] = mapped_column(JSONType)

    ml_class: Mapped[str | None] = mapped_column(String(10))
    ml_probabilities: Mapped[dict | None] = mapped_column(JSONType)
    ml_component: Mapped[float | None] = mapped_column(Float)
    high_risk_probability: Mapped[float | None] = mapped_column(Float)
    escalate: Mapped[bool | None] = mapped_column(Boolean)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    anomaly_component: Mapped[float | None] = mapped_column(Float)
    compliance_component: Mapped[float | None] = mapped_column(Float)
    business_component: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    contributions: Mapped[dict | None] = mapped_column(JSONType)
    floors_applied: Mapped[list | None] = mapped_column(JSONType)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    component_errors: Mapped[dict | None] = mapped_column(JSONType)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reasons: Mapped[list | None] = mapped_column(JSONType)
    recommended_actions: Mapped[list | None] = mapped_column(JSONType)
    shap_explained_class: Mapped[str | None] = mapped_column(String(10))
    shap_output_space: Mapped[str | None] = mapped_column(String(20))
    shap_base_value: Mapped[float | None] = mapped_column(Float)
    timings_ms: Mapped[dict | None] = mapped_column(JSONType)

    # LLM explanation (optional; filled asynchronously)
    llm_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|ready|unavailable|disabled
    llm_explanation: Mapped[dict | None] = mapped_column(JSONType)
    llm_model: Mapped[str | None] = mapped_column(String(80))
    llm_latency_ms: Mapped[float | None] = mapped_column(Float)
    llm_error: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="predictions", foreign_keys=[document_id])
    risk_factors: Mapped[list[RiskFactor]] = relationship(cascade="all, delete-orphan")
    anomalies: Mapped[list[AnomalyRecord]] = relationship(cascade="all, delete-orphan")
    violations: Mapped[list[ComplianceViolation]] = relationship(cascade="all, delete-orphan")


class RiskFactor(Base):
    __tablename__ = "risk_factors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    feature: Mapped[str] = mapped_column(String(60))
    label: Mapped[str] = mapped_column(String(120))
    feature_value: Mapped[float | None] = mapped_column(Float)
    shap_value: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(10))


class AnomalyRecord(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), index=True)
    anomaly_type: Mapped[str] = mapped_column(String(60))
    score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    description: Mapped[str] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSONType)


class ComplianceViolation(Base):
    __tablename__ = "compliance_violations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), index=True)
    rule_id: Mapped[str] = mapped_column(String(30), index=True)
    rule_name: Mapped[str] = mapped_column(String(120))
    severity: Mapped[str] = mapped_column(String(10))
    rule_type: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(Text)
    remediation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict | None] = mapped_column(JSONType)


class ReviewFeedback(Base):
    __tablename__ = "review_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"))
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    model_prediction: Mapped[str | None] = mapped_column(String(10))  # ML class
    model_risk_level: Mapped[str] = mapped_column(String(10))         # composite level shown to reviewer
    model_risk_score: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str | None] = mapped_column(String(20))
    decision: Mapped[str] = mapped_column(String(10))                 # confirm | correct
    reviewer_label: Mapped[str] = mapped_column(String(10))           # final label proposed by reviewer
    comment: Mapped[str | None] = mapped_column(Text)
    # pending -> validated | rejected (second-person validation before use in training)
    validation_status: Mapped[str] = mapped_column(String(12), default="pending", index=True)
    validation_flags: Mapped[list | None] = mapped_column(JSONType)
    validated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validation_comment: Mapped[str | None] = mapped_column(Text)
    used_in_training_run: Mapped[str | None] = mapped_column(String(64))


class ModelVersion(Base):
    """Operational mirror of the MLflow registry (what was served, when, with which metrics)."""

    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[str] = mapped_column(String(20))
    alias: Mapped[str | None] = mapped_column(String(20))
    family: Mapped[str | None] = mapped_column(String(40))
    run_id: Mapped[str | None] = mapped_column(String(64))
    feature_version: Mapped[str | None] = mapped_column(String(20))
    dataset_version: Mapped[str | None] = mapped_column(String(40))
    metrics: Mapped[dict | None] = mapped_column(JSONType)
    quality_gate: Mapped[dict | None] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(20))  # production | staging | rejected | archived
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(60), index=True)
    entity_type: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str | None] = mapped_column(String(40))
    request_id: Mapped[str | None] = mapped_column(String(40))
    details: Mapped[dict | None] = mapped_column(JSONType)  # identifiers/status only - never document text


class VendorContractRecord(Base):
    """Point-in-time vendor history used for vendor/behaviour features and anomaly checks."""

    __tablename__ = "vendor_contracts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_key: Mapped[str] = mapped_column(String(200), index=True)
    vendor_name: Mapped[str] = mapped_column(String(200))
    contract_date: Mapped[date] = mapped_column(Date, index=True)
    value_inr: Mapped[float | None] = mapped_column(Float)
    payment_days: Mapped[float | None] = mapped_column(Float)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    high_risk: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(20))  # historical_seed | analyzed
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))


class RequestMetric(Base):
    __tablename__ = "request_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    method: Mapped[str] = mapped_column(String(8))
    route: Mapped[str] = mapped_column(String(120), index=True)
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    request_id: Mapped[str | None] = mapped_column(String(40))


class MonitoringReport(Base):
    __tablename__ = "monitoring_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    report_type: Mapped[str] = mapped_column(String(30))  # drift | performance | retraining_decision
    model_version: Mapped[str | None] = mapped_column(String(20))
    summary: Mapped[dict] = mapped_column(JSONType)


Index("ix_vendor_contracts_vendor_date", VendorContractRecord.vendor_key, VendorContractRecord.contract_date)
