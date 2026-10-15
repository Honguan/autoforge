import time
from pathlib import Path


def quick(ctx):
    ctx.log_metric("loss", 0.5, step=0)
    model = Path(ctx.output_dir) / "model.txt"
    model.write_text(f"lr={ctx.hyperparameters['lr']} device={ctx.device}")
    return {"metrics": {"accuracy": 0.9}, "model_path": str(model)}


def slow(ctx):
    start = 0
    ckpt = ctx.latest_checkpoint()
    if ckpt:
        start = int(ckpt.read_text()) + 1
        ctx.log(f"resuming from step {start}")
    for step in range(start, ctx.hyperparameters["steps"]):
        ctx.check_stop()
        ctx.log_metric("loss", 1.0 / (step + 1), step=step)
        (ctx.checkpoint_dir / f"step{step:05d}.txt").write_text(str(step))
        time.sleep(0.05)
    return {"metrics": {"loss": 1.0 / ctx.hyperparameters["steps"]}}


def crash(ctx):
    raise RuntimeError("boom")


def stubborn(ctx):
    while True:
        time.sleep(0.1)


def adjustable(ctx):
    seen = []
    for step in range(100000):
        ctx.check_stop()
        if ctx.check_updates():
            seen.append(ctx.hyperparameters["lr"])
            ctx.log_metric("lr", ctx.hyperparameters["lr"], step=step)
        if ctx.hyperparameters.get("finish"):
            return {"metrics": {"final_lr": ctx.hyperparameters["lr"]}}
        time.sleep(0.05)


def score(ctx):
    """Deterministic objective for search tests: best at x=3, lr=0.1 adds a bonus."""
    x = ctx.hyperparameters["x"]
    bonus = 0.5 if ctx.hyperparameters.get("lr") == 0.1 else 0.0
    return {"metrics": {"score": 10 - (x - 3) ** 2 + bonus, "loss": (x - 3) ** 2}}


def needs_missing_pkg(ctx):
    import autoforge_no_such_module  # noqa: F401
