"""Structured (JSON-lines) logging.

Every log record is a single JSON object so it can be shipped to any log backend
(CloudWatch, Loki, ELK) and queried by request_id / document_id. Document text is never
logged — only identifiers, timings and status fields.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
    # Third-party libraries are chatty at INFO; keep our own logs readable.
    for noisy in ("httpx", "urllib3", "alembic", "mlflow", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Record the wall-clock duration of a block (milliseconds) into ``timings[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - start) * 1000, 2)
