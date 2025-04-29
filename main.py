import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym  # Updated to gymnasium

# Actor Network: Dual Outputs (Action + Kalman Gain)
class ActorNetwork(nn.Module):
    def __init__(self, compact_dim, raw_dim, action_dim, embedding_dim=128):
        super(ActorNetwork, self).__init__()
        # Embed compact and raw data into a shared latent space
        self.compact_embedding = nn.Linear(compact_dim, embedding_dim)
        self.raw_embedding = nn.Linear(raw_dim, embedding_dim)

        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.action_head = nn.Linear(128, action_dim)  # Action output
        self.kalman_gain_head = nn.Linear(128, 1)  # Kalman Gain output

    def forward(self, compact, raw):
        # Embed compact and raw inputs
        compact_embedded = self.compact_embedding(compact)
        raw_embedded = self.raw_embedding(raw)

        # Combine embeddings
        fused_embedding = compact_embedded + raw_embedded
        x = self.shared(fused_embedding)
        action = torch.tanh(self.action_head(x))  # Action output
        kalman_gain = torch.sigmoid(self.kalman_gain_head(x))  # Kalman Gain
        return action, kalman_gain

# Environment setup
env = gym.make("Acrobot-v1", render_mode="rgb_array")  # Set render_mode
env = DummyVecEnv([lambda: env])

# Functions to gather compact and raw data
def get_compact_data(env):
    """Get compact state data from the environment (angles and angular velocities)."""
    return np.array(env.get_attr("state")[0], dtype=np.float32)

def get_raw_data(env):
    """Get raw pixel observations from the environment."""
    unwrapped_env = env.envs[0]  # Access the unwrapped environment
    if hasattr(unwrapped_env, "render"):
        raw_obs = unwrapped_env.render()  # No need for `mode` here
        if raw_obs is not None:
            raw_obs_resized = torch.tensor(raw_obs, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
            raw_obs_resized = nn.functional.interpolate(raw_obs_resized.unsqueeze(0), size=(64, 64))
            return raw_obs_resized.flatten()  # Flatten to a vector
    raise ValueError("Environment does not support rendering or returned None for 'rgb_array' mode.")

# Training loop
def train_actor_network(env, episodes=1000, lr=1e-3):
    compact_dim = 4  # Compact state dimension (angles and velocities)
    raw_dim = 3 * 64 * 64  # Raw pixel input dimension (RGB image 64x64)
    action_dim = env.action_space.n

    # Initialize Actor Network
    actor = ActorNetwork(compact_dim, raw_dim, action_dim)
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0

        while True:
            # Gather compact and raw data
            compact_data = torch.tensor(get_compact_data(env), dtype=torch.float32)
            raw_data = get_raw_data(env)

            # Ensure data is batched
            compact_data = compact_data.unsqueeze(0)  # Add batch dimension
            raw_data = raw_data.unsqueeze(0)  # Add batch dimension

            # Actor outputs: action and Kalman Gain
            action, kalman_gain = actor(compact_data, raw_data)

            # Execute action in the environment
            action_index = action.argmax().item()
            obs, reward, done, _ = env.step([action_index])
            total_reward += reward

            if done:
                print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")
                break

# Run training
train_actor_network(env)
