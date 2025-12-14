import os
import gymnasium as gym  # Updated import
import numpy as np
import cv2
import matplotlib.pyplot as plt

from gymnasium import ObservationWrapper  # Updated import
from gymnasium.spaces import Box  # Updated import
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor, load_results

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

def make_env(wrapper_cls, log_dir, **kwargs):
    os.makedirs(log_dir, exist_ok=True)
    def _init():
        # env = gym.make("CartPole-v1", render_mode="human")  # Updated to gymnasium
        env = gym.make("CartPole-v1", render_mode=None)  # No animation
        env = wrapper_cls(env, **kwargs)
        env = Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))
        return env
    return DummyVecEnv([_init])

# ------------------------------------------------------------------------------


def train_sensor_only(log_dir='logs/sensor', timesteps=20_000):
    env = make_env(SensorObsWrapper, log_dir)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(log_dir, 'model'))
    return log_dir

def train_image_only(log_dir='logs/image', timesteps=20_000):
    env = make_env(ImageObsWrapper, log_dir, width=64, height=64)
    env = VecTransposeImage(env)  # (C, H, W) ordering for CnnPolicy
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(log_dir, 'model'))
    return log_dir

def train_fused_kf(log_dir='logs/fused_kf', timesteps=20_000):
    env = make_env(FusedKFObsWrapper, log_dir, width=32, height=32)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(log_dir, 'model'))
    return log_dir
# ------------------------------------------------------------------------------


def plot_learning_curves(log_dirs):
    plt.figure(figsize=(8, 5))
    for label, log_dir in log_dirs.items():
        df = load_results(log_dir)
        returns = df['r']
        plt.plot(returns.values, label=label)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('PPO on CartPole: Sensor vs. Image vs. KF-Fused Observations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Training sensor-only agent...")
    sensor_logs = train_sensor_only()

    print("Training image-only agent...")
    image_logs = train_image_only()

    print("Training KF-fused agent...")
    fused_logs = train_fused_kf()

    # Plot all three learning curves
    plot_learning_curves({
        'Sensor Only': sensor_logs,
        'Image Only':  image_logs,
        'KF-Fused':     fused_logs
    })
