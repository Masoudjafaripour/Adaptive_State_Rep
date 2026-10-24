# Adaptive State Representation

**Adaptive State Representation in Reinforcement Learning.** This project learns *feature-fusion gains* that combine a compact sensor state with rich image observations. The gains are trained end to end through a differentiable simulator, so the agent can shift its trust between the two sources when either one becomes noisy or corrupted.

## Core idea

A Kalman filter whose measurement noise, and therefore its gain, is predicted at every step by networks:

$$\hat x_t = (I - K_{s,t} - K_{i,t})\,\hat x^-_t + K_{s,t}\, z_t + K_{i,t}\, \mu^{img}_t, \qquad K_{s,t} = P_t R_{s,t}^{-1},\quad K_{i,t} = P_t R_{i,t}^{-1}$$

- $\hat x^-_t = A\hat x_{t-1} + Bu_{t-1}$ is the prediction from learned dynamics.
- $z_t$ is the noisy sensor reading.
- $\mu^{img}_t$ is the CNN's state estimate from two stacked frames.
- $R_{s,t} = S^2\,\mathrm{diag}\,\exp \mathrm{NN}_s([z_t - \hat x^-_t,\ z_t - \mu^{img}_t])$ and $R_{i,t} = \mathrm{diag}(v^{img}_t)$ are the predicted measurement noise of each source.

Everything (CNN, noise networks, dynamics and policy $u_t = \tanh\pi_\theta(\hat x_t, P_t)$) is trained jointly by backpropagating the return through the simulator and renderer:

$$\mathcal L = -\tfrac{1}{HN}\textstyle\sum r_t \;+\; \lambda_{aux}\lVert S^{-1}(\hat x_t - s_t)\rVert^2 \;+\; \lambda_{nll}\,\tfrac12\left(\log v_t + e_t^2 / v_t\right)$$

The NLL term teaches each source's variance $v_t$ its actual error $e_t$, which is what makes the gain adapt. Full derivation: [src/fusion_e2e/README.md](src/fusion_e2e/README.md#equations).

## Experiments

| Folder | What it tests | Status |
|---|---|---|
| [src/baselines/](src/baselines/) | PPO (SB3) on Gymnasium CartPole: sensor-only vs image-only vs a fixed-gain Kalman filter fusion | ✅ Sensor 293, image 34, KF-fused 28 (return, max 500). Fixed fusion over a random image projection *hurts*. |
| [src/fusion_e2e/](src/fusion_e2e/) | 8 fusion methods (sensor/image only, concat, fixed KF, gated, GRU, product of experts, **learned KF**) on a differentiable JAX CartPole with sensor or image corruption windows | 🟡 3 seeds running. Seed 0 so far: `concat` 476, `sensor_only` 443, `image_only` 363 (survival, max 500) |
| [src/latent_fusion/](src/latent_fusion/) | Same idea **without ground-truth state**: Kalman fusion in a learned latent space, trained self-supervised (innovation NLL, self-prediction, augmentation) against a privileged oracle | ⏳ Code ready, not trained yet |
| [src/wm_fusion/](src/wm_fusion/) | **Main contribution on the standard benchmark:** Dreamer-style world model with an adaptive Kalman posterior over pixels + proprio, on DeepMind Control (dm_control / MJX) | ⏳ Code ready, smoke-tested; needs a GPU for full runs |
| [src/learned_gain_draft/](src/learned_gain_draft/) | First prototype: actor network that outputs an action and a gain (Acrobot) | Archived, no training loop |

Details and metrics: [src/README.md](src/README.md). Every fusion_e2e run is logged in [runs/records.csv](src/fusion_e2e/runs/records.csv).

## Setup

```bash
uv venv srep-venv --python 3.11
VIRTUAL_ENV=$PWD/srep-venv uv pip install "jax[cpu]" optax flax torch gymnasium stable-baselines3 \
    opencv-python-headless matplotlib pygame pandas
```

## Run

```bash
srep-venv/bin/python src/baselines/train.py                       # PPO baselines
srep-venv/bin/python src/fusion_e2e/train.py --seed 0             # all fusion methods
srep-venv/bin/python src/fusion_e2e/train.py --methods learned_kf # one method
srep-venv/bin/python src/fusion_e2e/analyze.py                    # aggregate seeds -> README + plots
```

Training writes a live plot to `plots/live_training.png` (matplotlib Agg) and demo GIFs to `demo/`.

## To do

- [x] PPO baselines (sensor, image, fixed-gain KF)
- [x] Differentiable CartPole + soft renderer (JAX)
- [x] Learned, input-dependent Kalman gain trained end to end
- [x] Gain dynamics logged and plotted over time
- [ ] Multi-seed comparison: performance and sample efficiency (running)
- [ ] Harder corruption (sensor bias, longer windows), Acrobot / MountainCar

Contributors: *Masoud Jafaripour*
