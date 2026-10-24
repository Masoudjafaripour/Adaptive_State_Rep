import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Actor Network: Dual Outputs (Action + Kalman Gain) with CNN
class ActorNetwork(nn.Module):
    def __init__(self, compact_dim, action_dim, embedding_dim=128):
        super(ActorNetwork, self).__init__()
        # CNN for raw image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # Input channels = 3 (RGB)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),  # Adjust based on input image size
            nn.ReLU()
        )

        # Embedding layer for compact data
        self.compact_embedding = nn.Linear(compact_dim, embedding_dim)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # Combine compact and CNN embeddings
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.action_head = nn.Linear(128, action_dim)  # Action output
        self.kalman_gain_head = nn.Linear(128, 1)  # Kalman Gain output

    def forward(self, compact, raw_image):
        # Process raw image through CNN
        raw_features = self.cnn(raw_image)

        # Embed compact state
        compact_features = self.compact_embedding(compact)

        # Concatenate compact and raw features
        fused_features = torch.cat([compact_features, raw_features], dim=-1)

        # Shared layers
        x = self.shared(fused_features)
        action = torch.tanh(self.action_head(x))  # Action output
        kalman_gain = torch.sigmoid(self.kalman_gain_head(x))  # Kalman Gain
        return action, kalman_gain, compact_features, raw_features

# Environment setup
env = gym.make("Acrobot-v1", render_mode="rgb_array")  # Set render_mode
env = DummyVecEnv([lambda: env])

# Functions to gather compact and raw data
def get_compact_data(env):
    """Get normalized compact state data."""
    state = np.array(env.get_attr("state")[0], dtype=np.float32)
    state[0:2] = state[0:2] / np.pi  # Normalize angles
    state[2:4] = state[2:4] / 10.0   # Normalize angular velocities
    return state

def get_raw_data(env):
    """Get raw pixel observations from the environment."""
    unwrapped_env = env.envs[0]  # Access the unwrapped environment
    if hasattr(unwrapped_env, "render"):
        raw_obs = unwrapped_env.render()  # Render image
        if raw_obs is not None:
            raw_obs_resized = torch.tensor(raw_obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0, 1]
            raw_obs_resized = nn.functional.interpolate(raw_obs_resized, size=(84, 84))  # Resize
            return raw_obs_resized  # Return as tensor
    raise ValueError("Environment does not support rendering or returned None for 'rgb_array' mode.")

# Training loop
def train_actor_network(env, episodes=1000, lr=1e-4):
    compact_dim = 4  # Compact state dimension (angles and velocities)
    action_dim = env.action_space.n

    # Initialize Actor Network
    actor = ActorNetwork(compact_dim, action_dim)
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        step = 0

        while True:
            # Gather compact and raw data
            compact_data = torch.tensor(get_compact_data(env), dtype=torch.float32).unsqueeze(0)  # Add batch dim
            raw_data = get_raw_data(env)  # Already has batch dim and correct shape

            # Actor outputs: action, Kalman Gain, and features
            action, kalman_gain, compact_features, raw_features = actor(compact_data, raw_data)

            # Fuse observations using Kalman Gain
            fused_state = kalman_gain * compact_features + (1 - kalman_gain) * raw_features

            # Execute action in the environment
            action_index = action.argmax().item()
            obs, reward, done, _ = env.step([action_index])
            total_reward += reward
            step += 1

            # Debugging outputs
            if episode % 10 == 0:
                print(f"Step: {step}, Reward: {reward}, Action: {action}, Kalman Gain: {kalman_gain}")

            if done:
                print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")
                break

# Run training
train_actor_network(env)
