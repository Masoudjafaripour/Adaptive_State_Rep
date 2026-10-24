# Experiments

Each folder is a self-contained experiment with its own `README.md`, plots, demos and logs. Run commands from the repo root using the `srep-venv` environment.

| Folder | What it tests | Env / framework | Fusion | Training | Status | Run |
|---|---|---|---|---|---|---|
| [baselines/](baselines/) | Sensor-only vs image-only vs a fixed Kalman-filter fusion | Gymnasium CartPole-v1, SB3 | Kalman filter with fixed Q/R over the sensor and a *random* image projection | PPO, 20k steps | ✅ Done (sensor 293, image 34, KF-fused 28) | `srep-venv/bin/python src/baselines/train.py` |
| [fusion_e2e/](fusion_e2e/) | 8 fusion methods under sensor/image corruption, including the **learned-gain differentiable Kalman filter** | Differentiable CartPole + soft renderer, JAX | sensor/image only, concat, fixed KF, gated, GRU, product of experts, learned KF | Backprop through the simulator (analytic policy gradients), 150 iterations | 🟡 `run01` in progress | `srep-venv/bin/python src/fusion_e2e/train.py` |
| [latent_fusion/](latent_fusion/) | Adaptive fusion **without ground-truth state**: latent-space Kalman filter with self-supervised representation learning, against a privileged oracle and a naive version | Same JAX CartPole (reuses fusion_e2e) | Latent KF; oracle / naive state-space KF; 2 ablations | Backprop through the simulator + innovation NLL + self-prediction + VICReg + random shifts | ⏳ Code ready, not trained yet | `srep-venv/bin/python src/latent_fusion/train.py` |
| [wm_fusion/](wm_fusion/) | **Main contribution on the standard benchmark:** Dreamer-style world model whose posterior is an adaptive Kalman fusion of pixels + proprio | DeepMind Control (`cartpole_swingup`, `cheetah_run`, `walker_walk`), dm_control or MuJoCo Playground (MJX) | Adaptive KF posterior (ours); standard Dreamer concat; fixed-gain product of experts; image-only; proprio-only; no-NLL ablation | World model on replay + actor-critic in imagination (backprop through learned dynamics) | ⏳ Code ready, smoke-tested; needs a GPU for full runs | `srep-venv/bin/python src/wm_fusion/train.py --task cartpole_swingup` |
| [learned_gain_draft/](learned_gain_draft/) | First prototype: actor network that outputs an action and a gain | Gymnasium Acrobot, PyTorch | Gain head (unused) | ❌ No training loop | Archived | — |

## Metrics

- **baselines:** episode return in Gymnasium (max 500). This is the mean of the last 20 training episodes.
- **fusion_e2e:** survival steps out of 500 (pole within 12°, cart within 2.4), reported separately for clean, sensor-corrupt and image-corrupt evaluation. The full history of every run is in [fusion_e2e/runs/records.csv](fusion_e2e/runs/records.csv).

The two experiments aren't directly comparable: they use different action spaces (discrete vs continuous force), observation noise and training algorithms.
