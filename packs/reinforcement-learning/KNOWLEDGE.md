# reinforcement-learning — knowledge for AI planners

## When to use
- An agent must learn to **act** by trial and error in a simulator: control (balancing, robots, HVAC),
  games, scheduling/routing, pricing or inventory policies — anything you can step with `env.step(action)`.
- You have (or can write) a Gymnasium environment. Reward = what the user wants more of.

## When NOT to use
- No simulator, only a log of past decisions and outcomes → supervised learning (`tabular-classic`) on the
  log; offline RL is not supported.
- A one-shot prediction from a table, text or image → the matching supervised pack.
- The reward cannot be computed automatically, or one episode takes minutes of real time.

## Data format
**`data_path` is the environment spec, not a dataset.** No `task.target`. Two forms:

1. A registered Gymnasium environment — `env.yaml` (or `.json`):
   ```yaml
   env_id: CartPole-v1          # any gymnasium id; Atari/MuJoCo/Box2D need their extras installed
   env_kwargs: {}               # optional, passed to gymnasium.make
   max_episode_steps: 500       # optional
   ```
2. Your own simulator — a Python file defining `make_env()` that returns a `gymnasium.Env`:
   ```python
   import gymnasium as gym, numpy as np
   class Shop(gym.Env):
       observation_space = gym.spaces.Box(0, 100, shape=(3,), dtype=np.float32)
       action_space = gym.spaces.Discrete(4)              # e.g. 4 price levels
       def reset(self, seed=None, options=None): ...       # -> obs, info
       def step(self, action): ...                         # -> obs, reward, terminated, truncated, info
   def make_env():
       return Shop()
   ```
   The file is copied into the run (`output/env.py`) and imported from there; keep it self-contained.
   Episodes must end (`terminated` or `truncated`) or no episode rewards are logged.
Anything else (CSV, folder…) fails with "data_path must be an environment spec".

## Recipe(s)
`sb3` — Stable-Baselines3, `algorithm` chooses:
| algorithm | action space | notes |
|---|---|---|
| `PPO` (default) | Discrete, Box, MultiDiscrete | robust first choice |
| `A2C` | same as PPO | faster per step, noisier |
| `DQN` | Discrete only | sample-efficient for small discrete problems |
| `SAC` | Box (continuous) only | best for continuous control; slow per step on CPU |
A wrong combination fails with "DQN needs a Discrete action space" / "SAC needs a continuous Box".
`policy`: `MlpPolicy` (vector observations) or `CnnPolicy` (image observations, HxWxC uint8).
Small MLP policies train **faster on CPU** than on GPU; use a GPU only for `CnnPolicy`.

## Metrics
- `mean_reward` — final evaluation of the best checkpoint over `eval_episodes` deterministic episodes.
  **HPO metric, maximize.** Also `std_reward`, `timesteps`.
- Curves: `episode_reward` (rolling mean of the last 100 training episodes), `train_loss`, `eval_reward`
  (every `eval_every_steps`).

## Tuning guide
- `total_timesteps`: CartPole ~50k, classic control 100k–300k, MuJoCo 1M+, Atari 10M (GPU, CnnPolicy).
- `learning_rate` 1e-4–1e-3 (PPO 3e-4; DQN 1e-4; SAC 3e-4). `gamma` 0.99; 0.995–0.999 for long-horizon rewards.
- PPO: `n_steps` 256–2048 per env (`n_steps * n_envs` divisible by `batch_size`), `ent_coef` 0–0.01 for more exploration.
- DQN: `buffer_size` 50k–1M, `exploration_fraction` 0.1–0.3. `n_envs` 4–16 for PPO/A2C, 1 for DQN/SAC.
- `net_arch` [64, 64] is enough for most vector tasks; [256, 256] for SAC on robotics.
- `eval_episodes` ≥ 10 when the environment is random, or the best-checkpoint choice is noise.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `DQN needs a Discrete action space` | algorithm/action space mismatch | PPO/A2C, or SAC for Box actions |
| `unknown env_id` | typo or missing extra (box2d, mujoco, ale-py) | fix the id or install the extra |
| `episode_reward` never appears | episodes never end | return `truncated=True` after N steps |
| reward flat from the start | reward too sparse, too few timesteps | shape the reward, 5–10× timesteps, `ent_coef: 0.01` |
| reward rises then collapses | learning rate too high | lower `learning_rate` (`run adjust` works live) |
| eval_reward far below episode_reward | stochastic env, few eval episodes | raise `eval_episodes` |
`run retry` resumes from the last checkpoint with the remaining timesteps (DQN/SAC replay buffer restarts empty).

## Serving
Input row: `{"observation": [0.01, -0.2, 0.03, 0.1]}` (list or nested list with the observation shape;
optional `"deterministic": false` to sample). Output: `{"prediction": 1, "deterministic": true}` —
an int for Discrete actions, a list for Box/MultiDiscrete. The model folder holds `model.zip` (SB3,
loadable with `PPO.load`), `model.json` (algorithm) and the environment spec copy.
