"""Hardware detection. NVIDIA only for GPUs; falls back to CPU.

Detection order for GPUs: NVML (nvidia-ml-py) -> nvidia-smi -> torch.
Each step is optional, so `autoforge doctor` works on a bare CPU machine.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

import psutil

from autoforge.schemas.core import CPUInfo, GPUInfo, HardwareProfile

_GB = 1024**3


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _gpus_from_nvml(notes: list[str]) -> tuple[list[GPUInfo], str | None]:
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        return [], None
    try:
        pynvml.nvmlInit()
    except Exception as e:  # driver missing, no GPU, etc.
        notes.append(f"NVML init failed: {e}")
        return [], None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        cuda = None
        try:
            v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda = f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            pass
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    vram_total_gb=round(mem.total / _GB, 2),
                    vram_free_gb=round(mem.free / _GB, 2),
                    compute_capability=f"{major}.{minor}",
                    driver_version=driver,
                )
            )
        return gpus, cuda
    finally:
        pynvml.nvmlShutdown()


def parse_nvidia_smi(output: str) -> list[GPUInfo]:
    """Parse `nvidia-smi --query-gpu=index,name,memory.total,memory.free,
    compute_cap,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, free, cc, driver = parts[:6]
        gpus.append(
            GPUInfo(
                index=int(idx),
                name=name,
                vram_total_gb=round(float(total) / 1024, 2),
                vram_free_gb=round(float(free) / 1024, 2),
                compute_capability=cc if cc not in ("", "[N/A]") else None,
                driver_version=driver,
            )
        )
    return gpus


def _gpus_from_smi(notes: list[str]) -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        notes.append("nvidia-smi not found; no NVIDIA driver detected")
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return parse_nvidia_smi(out)
    except (subprocess.SubprocessError, ValueError) as e:
        notes.append(f"nvidia-smi failed: {e}")
        return []


def _torch_info(notes: list[str]) -> tuple[str | None, bool, str | None]:
    try:
        import torch
    except ImportError:
        notes.append('torch not installed; install with `pip install "autoforge-trainer[gpu]"`')
        return None, False, None
    available = bool(torch.cuda.is_available())
    return torch.__version__, available, torch.version.cuda


def detect_hardware() -> HardwareProfile:
    notes: list[str] = []
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())

    gpus, cuda = _gpus_from_nvml(notes)
    if not gpus:
        gpus = _gpus_from_smi(notes)
    torch_version, torch_cuda, torch_cuda_version = _torch_info(notes)
    cuda = cuda or torch_cuda_version

    if gpus and not torch_cuda:
        notes.append("NVIDIA GPU found but torch cannot use CUDA; training will run on CPU")

    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu=CPUInfo(
            model=_cpu_model(),
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
        ),
        ram_total_gb=round(vm.total / _GB, 2),
        ram_available_gb=round(vm.available / _GB, 2),
        disk_free_gb=round(disk.free / _GB, 2),
        gpus=gpus,
        cuda_version=cuda,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda,
        detection_notes=notes,
    )
