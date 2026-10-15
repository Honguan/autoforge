"""Core data structures shared by CLI, MCP, REST and GUI.

Every structure here is exported as JSON Schema (`autoforge schema <name>`)
so any AI client can read the exact contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Sensitivity(str, Enum):
    public = "public"
    internal = "internal"
    sensitive = "sensitive"


class BrainKind(str, Enum):
    subscription = "subscription"  # external AI client over MCP
    api = "api"  # built-in agent calling a cloud API
    local = "local"  # built-in agent calling a local model
    human = "human"


# ---------- hardware ----------


class GPUInfo(BaseModel):
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None


class CPUInfo(BaseModel):
    model: str
    physical_cores: int | None
    logical_cores: int | None


class HardwareProfile(BaseModel):
    os: str
    python_version: str
    cpu: CPUInfo
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GPUInfo] = Field(default_factory=list)
    cuda_version: str | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    detection_notes: list[str] = Field(default_factory=list)

    @property
    def best_device(self) -> str:
        return "cuda" if self.gpus and self.torch_cuda_available else "cpu"

    @property
    def max_vram_gb(self) -> float:
        return max((g.vram_total_gb for g in self.gpus), default=0.0)


# ---------- data / task / plan ----------


class DatasetProfile(BaseModel):
    path: str
    sensitivity: Sensitivity = Sensitivity.internal
    num_rows: int | None = None
    columns: dict[str, str] = Field(default_factory=dict, description="column -> dtype")
    stats: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    task_type: str = Field(description="e.g. text-classification, tabular-regression, llm-sft")
    input_modality: str
    output_modality: str
    target: str | None = None
    metric: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class TrainingPlan(BaseModel):
    task: TaskSpec
    data_path: str = Field(description="training data file or directory")
    pack: str | None = Field(default=None, description="installed pack name; None when using custom code")
    recipe: str | None = None
    custom_code_path: str | None = Field(
        default=None, description="AI-written trainer 'file.py[:function]' (function defaults to 'train')"
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 0
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    estimated_minutes: float | None = None
    estimated_vram_gb: float | None = None
    rationale: str = ""


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    succeeded = "succeeded"
    failed = "failed"
    stopped = "stopped"


# ---------- uniform result envelope ----------


class ErrorInfo(BaseModel):
    code: str
    message: str
    hint: str | None = None


class Envelope(BaseModel):
    """Every CLI --json / MCP / REST response uses this shape."""

    ok: bool
    data: Any = None
    error: ErrorInfo | None = None


SCHEMAS: dict[str, type[BaseModel]] = {
    "HardwareProfile": HardwareProfile,
    "DatasetProfile": DatasetProfile,
    "TaskSpec": TaskSpec,
    "TrainingPlan": TrainingPlan,
    "Envelope": Envelope,
}
