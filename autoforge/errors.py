"""Base error: every AutoForge error carries a stable code, an optional hint
and optional structured details, so AI clients can branch on them."""

from __future__ import annotations

from typing import Any


class AutoForgeError(Exception):
    code = "AUTOFORGE_ERROR"
    hint: str | None = None

    def __init__(self, message: str, hint: str | None = None, details: Any = None):
        super().__init__(message)
        if hint is not None:
            self.hint = hint
        self.details = details
