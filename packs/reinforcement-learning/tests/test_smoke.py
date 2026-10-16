"""Smoke test on CPU: PPO on CartPole-v1 (YAML spec), A2C on a tiny custom make_env() simulator, and resume."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sb3_train", ROOT / "trainers" / "sb3_train.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

CUSTOM_ENV = '''
import gymnasium as gym
import numpy as np


class GoRight(gym.Env):
    """Agent at position x in [0, 5]; action 1 moves right (+1 reward at the goal), action 0 moves left."""

    observation_space = gym.spaces.Box(0.0, 5.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.t = 0, 0
        return np.array([self.x], dtype=np.float32), {}

    def step(self, action):
        self.x = min(5, max(0, self.x + (1 if action == 1 else -1)))
        self.t += 1
        done = self.x == 5
        return np.array([self.x], dtype=np.float32), (1.0 if done else -0.1), done, self.t >= 20, {}


def make_env():
    return GoRight()
'''


def _ctx(tmp_path, data, hp, name="run", resume=None):
    from autoforge.engine.context import TrainContext

    return TrainContext(tmp_path / name, str(data), None, "reinforcement-learning", hp, "cpu", seed=0,
                        resume_from=resume)


def test_ppo_cartpole_yaml(tmp_path):
    env = tmp_path / "env.yaml"
    env.write_text("env_id: CartPole-v1\nenv_kwargs: {}\n")
    hp = {"algorithm": "PPO", "total_timesteps": 2048, "n_envs": 2, "n_steps": 256, "batch_size": 64,
          "eval_episodes": 3, "eval_every_steps": 1024, "net_arch": [32, 32]}
    res = st.train(_ctx(tmp_path, env, hp))
    m = res["metrics"]
    assert m["mean_reward"] > 0 and m["std_reward"] >= 0 and m["timesteps"] >= 2048
    out = Path(res["model_path"]).parent
    assert (out / "env.yaml").is_file() and json.loads((out / "io.json").read_text())["features"][0]["name"] == "observation"
    state = json.loads((tmp_path / "run" / "checkpoints" / "state.json").read_text())
    assert Path(state["path"]).is_file()
    model = st.load_model(out)
    pred = st.predict(model, [{"observation": [0.0, 0.1, 0.02, -0.1]}, {"observation": [0.0, 0.0, -0.2, 0.0]}])
    assert all(p["prediction"] in (0, 1) and p["deterministic"] is True for p in pred)
    with pytest.raises(ValueError, match="shape"):
        st.predict(model, [{"observation": [1.0, 2.0]}])


def test_custom_make_env_and_resume(tmp_path):
    env = tmp_path / "my_env.py"
    env.write_text(CUSTOM_ENV)
    hp = {"algorithm": "A2C", "total_timesteps": 600, "n_envs": 2, "learning_rate": 0.003,
          "eval_episodes": 2, "eval_every_steps": 300, "net_arch": [16]}
    res = st.train(_ctx(tmp_path, env, hp))
    out = Path(res["model_path"]).parent
    assert (out / "env.py").is_file() and res["metrics"]["timesteps"] >= 600
    assert st.predict(st.load_model(out), [{"observation": [2.0]}])[0]["prediction"] in (0, 1)
    # run retry: resume from the first run's checkpoints and continue to 1000 timesteps
    hp2 = dict(hp, total_timesteps=1000)
    res2 = st.train(_ctx(tmp_path, env, hp2, "run2", resume=str(tmp_path / "run" / "checkpoints")))
    assert 1000 <= res2["metrics"]["timesteps"] < 1400


def test_bad_specs_are_explained(tmp_path):
    pend = tmp_path / "pendulum.json"
    pend.write_text(json.dumps({"env_id": "Pendulum-v1"}))
    with pytest.raises(ValueError, match="DQN needs a Discrete"):
        st.train(_ctx(tmp_path, pend, {"algorithm": "DQN"}))
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="environment spec"):
        st.train(_ctx(tmp_path, csv, {}))
    bad = tmp_path / "bad.yaml"
    bad.write_text("env_id: NoSuchEnv-v9\n")
    with pytest.raises(ValueError, match="unknown env_id"):
        st.train(_ctx(tmp_path, bad, {}))
    nofn = tmp_path / "nofn.py"
    nofn.write_text("x = 1\n")
    with pytest.raises(ValueError, match="make_env"):
        st.train(_ctx(tmp_path, nofn, {}))
