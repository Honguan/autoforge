"""Privacy guard (requirement #8): enforced in the core layer, so CLI, MCP,
REST and GUI all go through the same check.

| sensitivity | local | api / subscription |
|-------------|-------|--------------------|
| public      | full  | full               |
| internal    | full  | summary only       |
| sensitive   | full  | denied             |

Training itself always runs locally; this governs what the AI brain may see.
"""

from __future__ import annotations

from enum import Enum

from autoforge.schemas.core import BrainKind, Sensitivity


class Access(str, Enum):
    full = "full"  # raw rows/samples may be shown to the brain
    summary = "summary"  # only aggregate statistics and schema
    denied = "denied"


class PrivacyDenied(Exception):
    code = "PRIVACY_DENIED"

    def __init__(self, brain: BrainKind, sensitivity: Sensitivity):
        self.brain = brain
        self.sensitivity = sensitivity
        super().__init__(
            f"Dataset marked '{sensitivity.value}' cannot be used with a '{brain.value}' brain."
        )

    hint = "Switch to a local model brain (e.g. --brain ollama:<model>) to use sensitive data."


def data_access(brain: BrainKind, sensitivity: Sensitivity) -> Access:
    if brain in (BrainKind.local, BrainKind.human):
        return Access.full
    if sensitivity is Sensitivity.public:
        return Access.full
    if sensitivity is Sensitivity.internal:
        return Access.summary
    return Access.denied


def require_access(brain: BrainKind, sensitivity: Sensitivity) -> Access:
    """Return the allowed access level, or raise PrivacyDenied."""
    access = data_access(brain, sensitivity)
    if access is Access.denied:
        raise PrivacyDenied(brain, sensitivity)
    return access
