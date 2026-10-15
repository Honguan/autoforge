from autoforge.hardware.detect import detect_hardware, parse_nvidia_smi


def test_detect_hardware_runs_without_gpu():
    hw = detect_hardware()
    assert hw.ram_total_gb > 0
    assert hw.best_device in ("cpu", "cuda")


def test_parse_nvidia_smi():
    out = "0, NVIDIA GeForce RTX 3060, 12288, 11000, 8.6, 555.42\n1, NVIDIA A100, 40960, 40000, 8.0, 555.42\n"
    gpus = parse_nvidia_smi(out)
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 3060", "NVIDIA A100"]
    assert gpus[0].vram_total_gb == 12.0
    assert gpus[1].compute_capability == "8.0"


def test_parse_nvidia_smi_skips_malformed_lines():
    assert parse_nvidia_smi("garbage\n") == []
