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

from autoforge.errors import AutoForgeError
from autoforge.schemas.core import BrainKind, Sensitivity


class Access(str, Enum):
    full = "full"  # raw rows/samples may be shown to the brain
    summary = "summary"  # only aggregate statistics and schema
    denied = "denied"


class PrivacyDenied(AutoForgeError):
    code = "PRIVACY_DENIED"
    hint = (
        "Do not retry. Sensitive data may only be used when a local model is in control "
        "(actor 'ai:local:<model>') or by a human."
    )

    def __init__(self, brain: BrainKind, sensitivity: Sensitivity, path: str | None = None):
        self.brain = brain
        self.sensitivity = sensitivity
        where = f" ({path})" if path else ""
        super().__init__(
            f"Dataset marked '{sensitivity.value}'{where} cannot be used with a '{brain.value}' brain.",
            details={"brain": brain.value, "sensitivity": sensitivity.value, "path": path},
        )


def brain_from_actor(actor: str) -> BrainKind:
    """'human' -> human, 'ai:local' / 'ai:local:<model>' -> local, any other 'ai:...' -> api (cloud)."""
    if actor == "human":
        return BrainKind.human
    if actor == "ai:local" or actor.startswith("ai:local:"):
        return BrainKind.local
    return BrainKind.api


def data_access(brain: BrainKind, sensitivity: Sensitivity) -> Access:
    if brain in (BrainKind.local, BrainKind.human):
        return Access.full
    if sensitivity is Sensitivity.public:
        return Access.full
    if sensitivity is Sensitivity.internal:
        return Access.summary
    return Access.denied


def require_access(brain: BrainKind, sensitivity: Sensitivity, path: str | None = None) -> Access:
    """Return the allowed access level, or raise PrivacyDenied."""
    access = data_access(brain, sensitivity)
    if access is Access.denied:
        raise PrivacyDenied(brain, sensitivity, path)
    return access
