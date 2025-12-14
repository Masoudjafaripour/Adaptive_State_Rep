# Adaptive_State_Rep

A repository for our project: **"Adaptive State Representation in Reinforcement Learning"**, which investigates how to fuse **compact low-dimensional state features** and **rich high-dimensional inputs (e.g., images)** using a **learned Kalman Gain**. The agent dynamically balances these two input sources to achieve robust control under **partial observability** and **sensor noise**.

---

## 🚀 Project Motivation
Reinforcement Learning (RL) agents deployed in real-world environments (e.g., robotics, control) often suffer from:
- **Incomplete observations** (partial observability)
- **Noisy sensors** or **missing modalities**
- Trade-offs between **compact features** (efficient but limited) and **rich raw inputs** (descriptive but noisy and large)

This repo explores an adaptive fusion mechanism to create **robust state representations** via a learnable Kalman Gain.

---

## 📌 Core Idea
We propose learning a Kalman Gain, denoted \( K_t \), that fuses compact state \( \mathbf{o}_t^{\text{compact}} \) and raw image-derived state \( \mathbf{o}_t^{\text{raw}} \) as:

\[ \mathbf{x}_t = K_t \cdot \mathbf{o}_t^{\text{compact}} + (1 - K_t) \cdot \mathbf{h}_\theta(\mathbf{o}_t^{\text{raw}}) \]

Where:
- \( \mathbf{h}_\theta(\cdot) \): A neural network to extract features from raw image
- \( K_t \): Learned dynamically from the actor network

This fused state \( \mathbf{x}_t \) is then used for policy learning with PPO.

---

## 🧠 Features
- PPO training with:
  - `Sensor-only` agent (4D state vector)
  - `Image-only` agent (grayscale image)
  - `Fused` agent (Kalman filter with sensor + image)
- `FusedKFObsWrapper`: Implements Kalman filtering with fixed or learned gain
- `VecTransposeImage` support for CNN input
- Visualization of reward curves for comparison

---

## 🏗️ Project Structure
```bash
KF_RL/
├── 3_all.py                  # Main training script
├── wrappers.py              # Custom observation wrappers (sensor, image, fused)
├── utils.py                 # Plotting and helper utilities
├── custom_cartpole.py       # Optional: custom env definitions
├── logs/                    # Saved model checkpoints and reward logs
└── requirements.txt
```

---

## 📈 Example Results
We compare performance across the three agents (sensor-only, image-only, fused) in `CartPole-v1`:

- **Sensor-only**: fast learning but sensitive to sensor noise
- **Image-only**: robust but slow to learn
- **Fused (KF)**: balances both, showing robustness and faster learning

![Learning Curve Placeholder](learning_curve.png)

---

## 📦 Installation
```bash
# Create and activate a new environment
conda create -n adaptive_state_rep python=3.11
conda activate adaptive_state_rep

# Install dependencies
pip install -r requirements.txt
```

If Box2D is problematic:
```bash
pip install pygame
pip install box2d
```

---

## 🔧 Run Training
```bash
# Train all three modes sequentially
python 3_all.py
```
Or edit the script to run only `sensor`, `image`, or `fused_kf` mode.

---

## 📜 Citation / Credits
This project is based on foundational ideas in:
- Kalman Filtering
- Representation Learning in RL
- PPO with Stable-Baselines3

Contributors: *Masoud Jafaripour*

---

## ✅ To Do
- [x] Train PPO on sensor-only and image-only inputs
- [x] Implement fused Kalman-based observation
- [ ] Log Kalman Gain dynamics over time
- [ ] Add learned Kalman Gain via neural output
- [ ] Test on Acrobot and MountainCar
