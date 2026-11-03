"""Minimal environment configuration and structured logging setup."""

import json
import logging
import os
from dataclasses import dataclass

from atlas.security.redaction import SecretRedactor


@dataclass(frozen=True)
class Settings:
    log_level: str = "INFO"

    @classmethod
    def from_environment(cls) -> "Settings":
        level = os.environ.get("ATLAS_LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("ATLAS_LOG_LEVEL must be a standard logging level")
        return cls(log_level=level)


class JsonFormatter(logging.Formatter):
    def __init__(self, redactor: SecretRedactor | None = None) -> None:
        super().__init__()
        self.redactor = redactor or SecretRedactor()

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": self.redactor.redact(record.getMessage()),
            }
        )


def configure_logging(settings: Settings | None = None) -> None:
    resolved = settings or Settings.from_environment()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved.log_level)
