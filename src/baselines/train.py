import os
import time
import gymnasium as gym  # Updated import
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless: plots are written to PNG files
import matplotlib.pyplot as plt

from gymnasium import ObservationWrapper  # Updated import
from gymnasium.spaces import Box  # Updated import
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import BaseCallback

from PIL import Image

# all outputs live next to this script: logs/, plots/, demo/, README.md
EXP_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(EXP_DIR, 'logs')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
DEMO_DIR  = os.path.join(EXP_DIR, 'demo')

# ------------------------------------------------------------------------------


class SensorObsWrapper(ObservationWrapper):
    """Pass through the default low-dimensional CartPole state."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        return obs

class ImageObsWrapper(ObservationWrapper):
    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self.width  = width
        self.height = height
        # Observation shape must be (H, W, C)
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    # def observation(self, obs):
    #     img = self.env.render()
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
    #     return resized.reshape((self.height, self.width, 1))  # (H, W, C)
    
    def observation(self, obs):
        img = self.env.render()
        if img is None or not img.size:
            # fallback if rendering fails early
            return np.zeros((self.height, self.width, 1), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.reshape((self.height, self.width, 1))  # channel-last

class FusedKFObsWrapper(ObservationWrapper):
    """
    Fuse sensor and raw-image-derived estimates via a simple Kalman Filter.
    - sensor_dim: number of sensor observations (4 for CartPole)
    - raw image is reduced to sensor_dim via a fixed random projection
    """
    def __init__(self, env, width=32, height=32):
        super().__init__(env)
        self.width  = width
        self.height = height

        # dimensions
        self.sensor_dim = env.observation_space.shape[0]
        self.img_size   = width * height

        # random projection from image -> sensor_dim
        self.proj = np.random.randn(self.sensor_dim, self.img_size).astype(np.float32)

        # Kalman filter matrices
        self.A = np.eye(self.sensor_dim)                         # state transition
        self.Q = np.eye(self.sensor_dim) * 0.01                  # process noise
        # measurement combines two sources => 2*sensor_dim
        self.H = np.vstack([np.eye(self.sensor_dim),            # sensor measurement
                            np.eye(self.sensor_dim)])           # raw-estimate measurement
        self.R = np.eye(2 * self.sensor_dim) * 0.1               # measurement noise

        # state and covariance will be initialized on reset
        self.x = np.zeros(self.sensor_dim, dtype=np.float32)
        self.P = np.eye(self.sensor_dim)

        # fused observation space is just the state_dim for the agent
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.sensor_dim,), dtype=np.float32
        )

    # def reset(self):
    #     obs = self.env.reset()
    #     # initialize state estimate = sensor reading
    #     self.x = obs.astype(np.float32)
    #     self.P = np.eye(self.sensor_dim)
    #     return self.x
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.x = obs.astype(np.float32)
        self.P = np.eye(self.sensor_dim)
        return self.x, info


    def observation(self, obs):
        # 1) sensor measurement
        z1 = obs.astype(np.float32)

        # 2) raw-image -> sensor_dim measurement via projection
        img = self.env.render()
        if img is None or not img.size:
            # Fallback if rendering fails
            flat = np.zeros(self.img_size, dtype=np.float32)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
            flat = (small.astype(np.float32).reshape(-1) / 255.0)

        z2 = self.proj.dot(flat)  # shape: (sensor_dim,)

        # combined measurement
        z = np.concatenate([z1, z2], axis=0)  # shape: (2*sensor_dim,)

        # --- Kalman Filter Predict ---
        x_pred = self.A.dot(self.x)
        P_pred = self.A.dot(self.P).dot(self.A.T) + self.Q

        # --- Kalman Gain ---
        S = self.H.dot(P_pred).dot(self.H.T) + self.R
        K = P_pred.dot(self.H.T).dot(np.linalg.inv(S))

        # --- Update ---
        y = z - self.H.dot(x_pred)
        self.x = x_pred + K.dot(y)
        self.P = (np.eye(self.sensor_dim) - K.dot(self.H)).dot(P_pred)

        return self.x

# ------------------------------------------------------------------------------

def make_env(wrapper_cls, log_dir=None, **kwargs):
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    def _init():
        # env = gym.make("CartPole-v1", render_mode="human")  # Updated to gymnasium
        # rgb_array is required: with render_mode=None, env.render() returns None
        # and the image/fused wrappers silently fall back to all-zero frames
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = wrapper_cls(env, **kwargs)
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))
        return env
    return DummyVecEnv([_init])

# ------------------------------------------------------------------------------


class LivePlotCallback(BaseCallback):
    """Redraw the learning-curve PNG every `plot_freq` steps during training."""
    def __init__(self, log_dirs, plot_freq=2_000):
        super().__init__()
        self.log_dirs  = log_dirs
        self.plot_freq = plot_freq

    def _on_step(self):
        if self.n_calls % self.plot_freq == 0:
            plot_learning_curves(self.log_dirs, os.path.join(PLOTS_DIR, 'live_training.png'))
        return True


def train_sensor_only(log_dir=os.path.join(LOG_DIR, 'sensor'), timesteps=20_000, callback=None):
    env = make_env(SensorObsWrapper, log_dir)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(log_dir, 'model'))
    return model

def train_image_only(log_dir=os.path.join(LOG_DIR, 'image'), timesteps=20_000, callback=None):
    env = make_env(ImageObsWrapper, log_dir, width=64, height=64)
    env = VecTransposeImage(env)  # (C, H, W) ordering for CnnPolicy
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(log_dir, 'model'))
    return model

def train_fused_kf(log_dir=os.path.join(LOG_DIR, 'fused_kf'), timesteps=20_000, callback=None):
    env = make_env(FusedKFObsWrapper, log_dir, width=32, height=32)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(log_dir, 'model'))
    return model
# ------------------------------------------------------------------------------


def plot_learning_curves(log_dirs, out_path, window=20):
    """Plot per-episode returns (faint) and a rolling mean (solid) for each agent."""
    plt.figure(figsize=(8, 5))
    for label, log_dir in log_dirs.items():
        if not os.path.exists(os.path.join(log_dir, 'monitor.csv')):
            continue
        df = load_results(log_dir)
        if df.empty:
            continue
        steps, returns = df['l'].cumsum().values, df['r']
        line, = plt.plot(steps, returns.values, alpha=0.25)
        plt.plot(steps, returns.rolling(window, min_periods=1).mean().values,
                 color=line.get_color(), label=label)
    plt.xlabel('Timestep')
    plt.ylabel('Episode return')
    plt.title('PPO on CartPole: Sensor vs. Image vs. KF-Fused Observations')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def write_results_readme(log_dirs, wall_times, timesteps, window=20):
    rows = []
    for label, log_dir in log_dirs.items():
        df = load_results(log_dir)
        r = df['r']
        rows.append(f"| {label} | {len(df)} | {r.tail(window).mean():.1f} ± {r.tail(window).std():.1f} "
                    f"| {r.max():.0f} | {wall_times[label]/60:.1f} |")
    text = f"""# Baselines: PPO on CartPole-v1

Generated by `src/baselines/train.py` ({timesteps:,} timesteps per agent, 1 seed, default SB3 PPO hyperparameters).

| Agent | Episodes | Return (last {window} eps, mean ± std) | Best return | Wall time (min) |
|---|---|---|---|---|
""" + "\n".join(rows) + """

Max possible return is 500.

![Learning curves](plots/learning_curves.png)

## Demos (greedy policy, 1 episode each)

| Sensor Only | Image Only | KF-Fused |
|---|---|---|
| ![](demo/sensor.gif) | ![](demo/image.gif) | ![](demo/fused_kf.gif) |

Run: `srep-venv/bin/python src/baselines/train.py` (from the repo root). The live plot `plots/live_training.png` is redrawn every 2k steps during training.

- **Sensor Only**: 4D CartPole state.
- **Image Only**: 64x64 grayscale render, `CnnPolicy`. A single frame has no velocity information.
- **KF-Fused**: Kalman filter over the sensor state and a *fixed random projection* of a 32x32 render (fixed Q/R, no learned gain).
"""
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as f:
        f.write(text)

def record_gif(model, env, out_path, max_steps=500, fps=30):
    """Roll out one greedy episode on a VecEnv and save the rendered frames as a GIF."""
    obs = env.reset()
    frames, ret = [], 0.0
    for _ in range(max_steps):
        frames.append(Image.fromarray(env.envs[0].render()))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ret += reward[0]
        if done[0]:
            break
    frames[0].save(out_path, save_all=True, append_images=frames[1:][::2],
                   duration=2 * 1000 // fps, loop=0)
    return ret

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    for d in (PLOTS_DIR, DEMO_DIR):
        os.makedirs(d, exist_ok=True)
    timesteps = 20_000
    agents = [
        ('Sensor Only', train_sensor_only, 'sensor'),
        ('Image Only',  train_image_only,  'image'),
        ('KF-Fused',    train_fused_kf,    'fused_kf'),
    ]
    log_dirs, wall_times, models = {}, {}, {}
    for label, train_fn, name in agents:
        print(f"Training {label} agent...")
        # live plot shows agents finished so far plus the one currently training
        log_dirs[label] = os.path.join(LOG_DIR, name)
        start = time.time()
        models[name] = train_fn(log_dir=log_dirs[label], timesteps=timesteps,
                                callback=LivePlotCallback(dict(log_dirs)))
        wall_times[label] = time.time() - start

    plot_learning_curves(log_dirs, os.path.join(PLOTS_DIR, 'learning_curves.png'))

    # demo GIFs on fresh, unmonitored envs so the training monitor.csv files are untouched
    eval_envs = {
        'sensor':   make_env(SensorObsWrapper, None),
        'image':    VecTransposeImage(make_env(ImageObsWrapper, None,
                                               width=64, height=64)),
        'fused_kf': make_env(FusedKFObsWrapper, None,
                             width=32, height=32),
    }
    for name, env in eval_envs.items():
        ret = record_gif(models[name], env, os.path.join(DEMO_DIR, f'{name}.gif'))
        print(f"Demo {name}: return {ret:.0f}")

    write_results_readme(log_dirs, wall_times, timesteps)
    print(f"Results written to {EXP_DIR}/")
