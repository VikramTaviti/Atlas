"""Small dependency-free redactor for logs, traces, and audit summaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


class SecretRedactor:
    _sensitive_key = re.compile(
        r"(secret|password|passwd|token|api[_-]?key|authorization|credential|private[_-]?key)",
        re.IGNORECASE,
    )
    _bearer = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
    _assignment = re.compile(
        r"(?i)(\b(?:password|passwd|token|secret|api[_-]?key|authorization|credential)\b\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
    )

    def __init__(self, secret_values: tuple[str, ...] = ()) -> None:
        self._secret_values = tuple(sorted((s for s in secret_values if s), key=len, reverse=True))

    def with_secrets(self, secret_values: tuple[str, ...]) -> SecretRedactor:
        """Return a redactor that preserves existing rules and adds exact secret values."""
        return SecretRedactor((*self._secret_values, *secret_values))

    def redact(self, value: Any, *, key: str | None = None) -> Any:
        if key is not None and self._sensitive_key.search(key):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(k): self.redact(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if not isinstance(value, str):
            return value

        candidate = value.strip()
        if candidate.startswith(("{", "[")):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                return json.dumps(self.redact(parsed), ensure_ascii=False, separators=(",", ":"))
        redacted = self._bearer.sub("Bearer [REDACTED]", value)
        redacted = self._assignment.sub(r"\1[REDACTED]", redacted)
        for secret in self._secret_values:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
