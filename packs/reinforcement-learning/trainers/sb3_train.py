"""Train a Stable-Baselines3 agent (PPO, A2C, DQN or SAC) on a Gymnasium environment.

data_path is the ENVIRONMENT spec, not a dataset:
  * env.yaml / env.json   {"env_id": "CartPole-v1", "env_kwargs": {...}, "max_episode_steps": 500 (optional)}
  * my_env.py             defines make_env() -> gymnasium.Env (your own simulator); the file is copied
                          into the run (output/env.py) and imported from there with importlib.
AutoForge controls: check_stop every env step, live learning_rate (run adjust), episode_reward /
train_loss / eval_reward curves, evaluation + checkpoint every eval_every_steps (checkpoints/state.json,
resumed by `run retry` with the remaining timesteps), best evaluated model kept.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections import deque
from pathlib import Path

ALGOS = ("PPO", "A2C", "DQN", "SAC")
SPEC_EXTS = {".yaml", ".yml", ".json"}


# ---------------------------------------------------------------- environment spec

def _read_spec(path: Path) -> dict:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"environment spec not found: {path}")
    try:
        spec = json.loads(path.read_text()) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{path.name}: cannot parse environment spec: {e}") from e
    if not isinstance(spec, dict) or not isinstance(spec.get("env_id"), str):
        raise ValueError(f"{path.name}: expected a mapping with 'env_id' (e.g. env_id: CartPole-v1) "
                         f"and optional 'env_kwargs', got {spec!r}")
    if not isinstance(spec.get("env_kwargs") or {}, dict):
        raise ValueError(f"{path.name}: 'env_kwargs' must be a mapping")
    return spec


def _load_make_env(py_file: Path):
    spec = importlib.util.spec_from_file_location("autoforge_user_env", py_file)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{py_file.name}: importing the environment file failed: {e!r}") from e
    fn = getattr(mod, "make_env", None)
    if not callable(fn):
        raise ValueError(f"{py_file.name}: must define a function make_env() that returns a gymnasium.Env")
    return fn


def _env_factory(data_path: str, out: Path):
    """Return (make_env callable, spec description). Copies the spec into `out`."""
    import gymnasium as gym

    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix in SPEC_EXTS:
        spec = _read_spec(path)
        shutil.copy(path, out / f"env{suffix}")
        env_id, kwargs = spec["env_id"], dict(spec.get("env_kwargs") or {})
        if spec.get("max_episode_steps"):
            kwargs["max_episode_steps"] = int(spec["max_episode_steps"])
        try:
            gym.spec(env_id)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{path.name}: unknown env_id '{env_id}' ({e}). Use a registered Gymnasium id such as "
                             "CartPole-v1, or a Python file with make_env() for your own simulator") from e

        def make():
            return gym.make(env_id, **kwargs)
        return make, {"kind": "gymnasium", "file": f"env{suffix}", "env_id": env_id}
    if suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"environment file not found: {path}")
        copy = out / "env.py"
        shutil.copy(path, copy)
        user_make = _load_make_env(copy)

        def make():
            env = user_make()
            if not isinstance(env, gym.Env):
                raise ValueError(f"{path.name}: make_env() must return a gymnasium.Env, got {type(env).__name__}")
            return env
        return make, {"kind": "python", "file": "env.py"}
    raise ValueError(f"data_path must be an environment spec (.yaml/.yml/.json with env_id) or a .py file defining "
                     f"make_env(); got '{path.name}'. Reinforcement learning needs a simulator, not a dataset")


def _check_action_space(algo: str, space) -> None:
    from gymnasium import spaces

    if algo == "DQN" and not isinstance(space, spaces.Discrete):
        raise ValueError(f"DQN needs a Discrete action space, this environment has {space}. "
                         "Use algorithm: PPO, A2C or (continuous Box actions) SAC")
    if algo == "SAC" and not isinstance(space, spaces.Box):
        raise ValueError(f"SAC needs a continuous Box action space, this environment has {space}. "
                         "Use algorithm: PPO, A2C or (Discrete actions) DQN")
    if algo in ("PPO", "A2C") and not isinstance(space, (spaces.Discrete, spaces.Box, spaces.MultiDiscrete,
                                                        spaces.MultiBinary)):
        raise ValueError(f"{algo} does not support action space {space}")


def _algo_class(name: str):
    import stable_baselines3 as sb3

    return getattr(sb3, name)


def _loss(values: dict):
    for key in ("train/loss",):
        if key in values:
            return values[key]
    if "train/policy_loss" in values and "train/value_loss" in values:  # A2C
        return values["train/policy_loss"] + values["train/value_loss"]
    if "train/critic_loss" in values:  # SAC
        return values["train/critic_loss"]
    return None


# ---------------------------------------------------------------- training

def train(ctx) -> dict:
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    hp = ctx.hyperparameters
    out, ck_dir = Path(ctx.output_dir), Path(ctx.checkpoint_dir)
    algo = str(hp.get("algorithm", "PPO")).upper()
    if algo not in ALGOS:
        raise ValueError(f"algorithm must be one of {list(ALGOS)}, got '{hp.get('algorithm')}'")
    make, env_desc = _env_factory(ctx.data_path, out)
    total = int(hp.get("total_timesteps", 100_000))
    n_envs = max(1, int(hp.get("n_envs", 4)))
    eval_episodes = max(1, int(hp.get("eval_episodes", 10)))
    eval_every = max(1, int(hp.get("eval_every_steps", 10_000)))
    policy = str(hp.get("policy", "MlpPolicy"))
    torch.manual_seed(ctx.seed)
    np.random.seed(ctx.seed)
    device = "cuda" if ctx.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cpu" and policy == "MlpPolicy":
        torch.set_num_threads(1)  # small MLPs: extra threads only add overhead (several times slower on busy CPUs)

    probe = make()
    _check_action_space(algo, probe.action_space)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    venv = DummyVecEnv([lambda: Monitor(make()) for _ in range(n_envs)])
    venv.seed(ctx.seed)
    eval_env = Monitor(make())
    eval_env.reset(seed=ctx.seed + 10_000)
    cls = _algo_class(algo)

    best = {"reward": None, "path": None}
    state_file = ctx.latest_checkpoint()
    if state_file is not None and state_file.name == "state.json":
        state = json.loads(state_file.read_text())
        model = cls.load(state["path"], env=venv, device=device)
        best = {"reward": state.get("best_reward"), "path": state.get("best_path")}
        ctx.log(f"resuming from {state['timesteps']} timesteps"
                + (" (replay buffer starts empty)" if algo in ("DQN", "SAC") else ""))
    else:
        kwargs = {"learning_rate": float(hp.get("learning_rate", 3e-4)), "gamma": float(hp.get("gamma", 0.99)),
                  "policy_kwargs": {"net_arch": [int(x) for x in hp.get("net_arch", [64, 64])]},
                  "seed": ctx.seed, "device": device, "verbose": 0}
        if algo == "PPO":
            kwargs.update(n_steps=int(hp.get("n_steps", 2048)), batch_size=int(hp.get("batch_size", 64)),
                          ent_coef=float(hp.get("ent_coef", 0.0)))
        elif algo == "A2C":
            kwargs.update(ent_coef=float(hp.get("ent_coef", 0.0)))
        elif algo == "DQN":
            kwargs.update(buffer_size=int(hp.get("buffer_size", 100_000)), batch_size=int(hp.get("batch_size", 64)),
                          exploration_fraction=float(hp.get("exploration_fraction", 0.1)))
        elif algo == "SAC":
            kwargs.update(buffer_size=int(hp.get("buffer_size", 100_000)), batch_size=int(hp.get("batch_size", 64)))
        model = cls(policy, venv, **kwargs)
    done = int(model.num_timesteps)
    remaining = max(0, total - done)
    ctx.log(f"{algo} {policy} on {device} ({env_desc.get('env_id', env_desc['file'])}): obs {obs_space}, "
            f"actions {act_space}, {n_envs} envs, {remaining} of {total} timesteps to go"
            + ("  [tip: MLP policies are usually faster on CPU]" if device == "cuda" and policy == "MlpPolicy" else ""))

    def evaluate(m):
        mean, std = evaluate_policy(m, eval_env, n_eval_episodes=eval_episodes, deterministic=True, warn=False)
        return float(mean), float(std)

    def checkpoint(m, steps: int):
        mean, std = evaluate(m)
        ctx.log_metric("eval_reward", mean, step=steps)
        path = ck_dir / f"step{steps}.zip"
        m.save(path)
        if best["reward"] is None or mean >= best["reward"]:
            best["reward"], best["path"] = mean, str(ck_dir / "best_model.zip")
            m.save(best["path"])
        for old in ck_dir.glob("step*.zip"):
            if old != path:
                old.unlink(missing_ok=True)
        (ck_dir / "state.json").write_text(json.dumps({"timesteps": steps, "path": str(path), "best_reward":
                                                       best["reward"], "best_path": best["path"]}))
        ctx.log(f"step {steps}: eval_reward={mean:.3f} ± {std:.3f} (best {best['reward']:.3f})")

    log_every = max(500, total // 200)

    class Cb(BaseCallback):
        def _on_training_start(self):
            self.episodes = deque(maxlen=100)
            self.last_log = self.last_upd = self.last_eval = self.num_timesteps

        def _on_step(self) -> bool:
            ctx.check_stop()  # raises RunStopped -> the run ends cleanly
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episodes.append(info["episode"]["r"])
            n = self.num_timesteps
            if n - self.last_upd >= 256:
                self.last_upd = n
                changed = ctx.check_updates()
                if "learning_rate" in changed:
                    self.model.learning_rate = float(changed["learning_rate"])
                    self.model._setup_lr_schedule()
            if n - self.last_log >= log_every:
                self.last_log = n
                if self.episodes:
                    ctx.log_metric("episode_reward", float(np.mean(self.episodes)), step=n)
                loss = _loss(self.model.logger.name_to_value)
                if loss is not None:
                    ctx.log_metric("train_loss", float(loss), step=n)
            if n - self.last_eval >= eval_every:
                self.last_eval = n
                checkpoint(self.model, n)
            return True

    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=Cb(), reset_num_timesteps=False, log_interval=None,
                    progress_bar=False)
    steps = int(model.num_timesteps)
    state = json.loads((ck_dir / "state.json").read_text()) if (ck_dir / "state.json").is_file() else {}
    if state.get("timesteps") != steps:
        checkpoint(model, steps)  # evaluate the final policy too
    best_model = cls.load(best["path"], device=device)
    mean, std = evaluate(best_model)
    ctx.log(f"best model: mean_reward={mean:.3f} ± {std:.3f} over {eval_episodes} episodes")
    best_model.save(out / "model.zip")
    (out / "model.json").write_text(json.dumps({
        "algorithm": algo, "policy": policy, "env": env_desc, "timesteps": steps,
        "observation_shape": list(obs_space.shape or ()), "action_space": str(act_space)}))
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "observation", "dtype": "any",
         "description": f"environment observation, list / nested list of shape {list(obs_space.shape or ())}"},
        {"name": "deterministic", "dtype": "any", "required": False,
         "description": "true (default) = best action; false = sample from the policy"}]}))
    venv.close()
    eval_env.close()
    return {"metrics": {"mean_reward": mean, "std_reward": std, "timesteps": steps},
            "model_path": str(out / "model.zip")}


# ---------------------------------------------------------------- serving

def load_model(model_dir):
    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = _algo_class(meta["algorithm"]).load(Path(model_dir) / "model.zip", device="cpu")
    return {"model": model, "meta": meta}


def predict(m, rows: list[dict]) -> list[dict]:
    import numpy as np
    from gymnasium import spaces

    model = m["model"]
    space = model.observation_space
    results = []
    for i, row in enumerate(rows):
        if "observation" not in row:
            raise ValueError(f"row {i}: missing 'observation'")
        obs = np.asarray(row["observation"], dtype=space.dtype)
        if isinstance(space, spaces.Box) and obs.size != int(np.prod(space.shape)):
            raise ValueError(f"row {i}: observation has shape {list(obs.shape)}, expected {list(space.shape)}")
        det = row.get("deterministic", True)
        det = det if isinstance(det, bool) else str(det).lower() not in ("false", "0", "no")
        action, _ = model.predict(obs, deterministic=det)
        action = np.asarray(action)
        results.append({"prediction": action.item() if action.ndim == 0 else action.tolist(), "deterministic": det})
    return results
