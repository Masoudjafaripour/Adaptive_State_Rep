"""End-to-end fusion via differentiable simulation (CartPole, JAX).

Each method in fusion.METHODS is trained by backpropagating the return through the
simulator, the soft renderer and the fusion module (short-horizon analytic policy
gradients: episodes are cut into `horizon`-step windows, the gradient is truncated
between windows).

Outputs, next to this file:
    plots/live_training.png   redrawn during training (matplotlib Agg)
    plots/*.png               final learning curves, per-condition results, gain traces
    demo/<method>_<mode>.gif  one evaluation episode per method and corruption mode
    runs/<run_id>/            config, per-method history and params
    runs/records.csv          one row per (run, method); never overwritten, only appended
    README.md                 regenerated from records.csv + notes.md

Usage (from the repo root):
    srep-venv/bin/python src/fusion_e2e/train.py                 # all methods
    srep-venv/bin/python src/fusion_e2e/train.py --methods learned_kf gru --iters 100
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')  # headless: plots are written to PNG files
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
from PIL import Image, ImageDraw

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
import cartpole_jax as env                      # noqa: E402
from fusion import METHODS, mlp, mlp_init       # noqa: E402

PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
DEMO_DIR  = os.path.join(EXP_DIR, 'demo')
RUNS_DIR  = os.path.join(EXP_DIR, 'runs')
RECORDS   = os.path.join(RUNS_DIR, 'records.csv')

MODES = ['clean', 'sensor_corrupt', 'image_corrupt']
RECORD_FIELDS = ['run_id', 'timestamp', 'method', 'iters', 'seed',
                 'clean', 'sensor_corrupt', 'image_corrupt', 'mean', 'est_rmse', 'train_min']

DEFAULT_CFG = dict(
    n_envs=64, episode_len=200, horizon=25, lr=1e-3, aux_weight=0.5, nll_weight=0.1, grad_clip=1.0,
    eval_envs=64, eval_len=500, corrupt_window=(150, 350), eval_every=20,
)

# ------------------------------------------------------------------------------


def build(method, cfg):
    """Jitted train/eval functions for one method."""
    opt = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(cfg['grad_clip']), optax.adam(cfg['lr'])), 20)

    def init_state(key, n):
        s = env.reset(key, n)
        return (s, method.init_carry(n), env.render(s), jnp.zeros(n))

    def rollout(params, c, keys, s_bad, i_bad):
        def body(c, xs):
            s, mc, prev_img, u_prev = c
            key, sb, ib = xs
            z, img = env.observe(key, s, sb, ib)
            frames = jnp.stack([img, prev_img], -1)
            feat, est, mc, info = method.step(params['fusion'], mc, z, frames, u_prev)
            u = jnp.tanh(mlp(params['policy'], feat))[:, 0]
            target = jax.lax.stop_gradient(s)
            aux = jnp.mean(((est['x'] - target) / env.STATE_SCALE) ** 2)
            if 'mu_img' in est:
                aux = aux + jnp.mean(((est['mu_img'] - target) / env.STATE_SCALE) ** 2)
            # heteroscedastic Gaussian NLL: teaches each source's variance head its actual error
            for key_v, meas in (('var_sensor', z), ('var_img', est.get('mu_img'))):
                if key_v in est:
                    v = est[key_v] / env.STATE_SCALE ** 2
                    err2 = ((meas - target) / env.STATE_SCALE) ** 2
                    aux = aux + cfg['nll_weight'] * jnp.mean(0.5 * (jnp.log(v) + err2 / v))
            out = dict(r=env.reward(s, u), aux=aux, s=s, x_est=est['x'], u=u, **info)
            return (env.step(s, u), mc, img, u), out
        return jax.lax.scan(body, c, (keys, s_bad, i_bad))

    def loss_fn(params, c, keys, s_bad, i_bad):
        c2, out = rollout(params, c, keys, s_bad, i_bad)
        rl, aux = -jnp.mean(out['r']), jnp.mean(out['aux'])
        return rl + cfg['aux_weight'] * aux, (c2, rl, aux)

    @jax.jit
    def update(params, opt_state, c, keys, s_bad, i_bad):
        (_, (c2, rl, aux)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, c, keys, s_bad, i_bad)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, jax.lax.stop_gradient(c2), -rl, aux

    @jax.jit
    def reset_failed(c, key):
        """Restart envs whose pole fell (between gradient windows, not differentiated)."""
        mask = env.failed(c[0])
        fresh = init_state(key, mask.shape[0])
        return jax.tree_util.tree_map(
            lambda f, o: jnp.where(mask.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), fresh, c)

    def init_params(key):
        k1, k2 = jax.random.split(key)
        return {'fusion': method.init(k1),
                'policy': mlp_init(k2, [method.feat_dim, 64, 64, 1], last_scale=0.1)}

    return dict(opt=opt, init_state=init_state, update=update, reset_failed=reset_failed,
                eval_rollout=jax.jit(rollout), init_params=init_params)


def evaluate(fns, params, cfg, key, keep_traces=False):
    """Survival steps (max eval_len) under each corruption mode, on a fixed key."""
    n, T = cfg['eval_envs'], cfg['eval_len']
    res = {}
    for mode in MODES:
        k1, k2 = jax.random.split(key)
        s_bad, i_bad = env.fixed_schedule(mode, n, T, *cfg['corrupt_window'])
        _, out = fns['eval_rollout'](params, fns['init_state'](k1, n), jax.random.split(k2, T), s_bad, i_bad)
        alive = jnp.cumprod(~env.failed(out['s']), axis=0)
        err = ((out['x_est'] - out['s']) / env.STATE_SCALE) ** 2
        rmse = jnp.sqrt((err.mean(-1) * alive).sum() / jnp.maximum(alive.sum(), 1))
        res[mode] = {'survival': float(alive.sum(0).mean()), 'est_rmse': float(rmse)}
        if keep_traces:
            res[mode]['out'] = jax.tree_util.tree_map(np.asarray, out)
            res[mode]['alive'] = np.asarray(alive)
            res[mode]['sensor_bad'] = np.asarray(s_bad)
            res[mode]['image_bad'] = np.asarray(i_bad)
    return res


def train_method(name, cfg, iters, seed, live_hist, live_path):
    method = METHODS[name]
    fns = build(method, cfg)
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    eval_key = jax.random.PRNGKey(10_000 + seed)
    params = fns['init_params'](k_init)
    opt_state = fns['opt'].init(params)
    N, T, H = cfg['n_envs'], cfg['episode_len'], cfg['horizon']

    hist = {'iter': [], 'reward': [], 'aux': [], 'eval_iter': [], **{f'eval_{m}': [] for m in MODES}}
    live_hist[name] = hist
    start = time.time()
    for it in range(iters):
        key, k_reset, k_sched = jax.random.split(key, 3)
        c = fns['init_state'](k_reset, N)
        s_bad, i_bad = env.sample_schedule(k_sched, N, T)
        rs, auxs = [], []
        for ch in range(T // H):
            key, k_noise, k_re = jax.random.split(key, 3)
            sl = slice(ch * H, (ch + 1) * H)
            params, opt_state, c, r, a = fns['update'](params, opt_state, c, jax.random.split(k_noise, H),
                                                       s_bad[sl], i_bad[sl])
            c = fns['reset_failed'](c, k_re)
            rs.append(r)
            auxs.append(a)
        hist['iter'].append(it)
        hist['reward'].append(float(jnp.mean(jnp.array(rs))))
        hist['aux'].append(float(jnp.mean(jnp.array(auxs))))

        if (it + 1) % cfg['eval_every'] == 0 or it == iters - 1:
            ev = evaluate(fns, params, cfg, eval_key)
            hist['eval_iter'].append(it + 1)
            for m in MODES:
                hist[f'eval_{m}'].append(ev[m]['survival'])
            print(f"[{name}] iter {it + 1}/{iters}  reward {hist['reward'][-1]:.3f}  aux {hist['aux'][-1]:.3f}  "
                  + '  '.join(f"{m} {ev[m]['survival']:.0f}" for m in MODES), flush=True)
            plot_live(live_hist, live_path)

    train_min = (time.time() - start) / 60
    final = evaluate(fns, params, cfg, eval_key, keep_traces=True)
    return params, hist, final, train_min

# ------------------------------------------------------------------------------
# plots


def _rolling(y, w=10):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


def plot_live(all_hist, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in all_hist.items():
        if not h['iter']:
            continue
        line, = axes[0].plot(h['iter'], _rolling(h['reward']), label=name)
        if h['eval_iter']:
            mean_surv = np.mean([h[f'eval_{m}'] for m in MODES], axis=0)
            axes[1].plot(h['eval_iter'], mean_surv, 'o-', color=line.get_color(), label=name)
    axes[0].set(xlabel='Iteration (episode batch)', ylabel='Mean step reward (rolling 10)', title='Training reward')
    axes[1].set(xlabel='Iteration', ylabel='Survival steps (mean of 3 conditions)', title='Evaluation',
                ylim=(0, 520))
    for ax in axes:
        ax.grid(True)
        ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_by_condition(latest, out_path):
    names = list(latest)
    w = 0.8 / len(MODES)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names)), 4.5))
    for j, m in enumerate(MODES):
        ax.bar(np.arange(len(names)) + (j - 1) * w, [float(latest[n][m]) for n in names], w, label=m)
    ax.set_xticks(np.arange(len(names)), names, rotation=20)
    ax.set(ylabel='Survival steps (max 500)', title='Evaluation by condition (latest run of each method)',
           ylim=(0, 520))
    ax.grid(True, axis='y')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_gain_traces(finals, cfg, out_path):
    """Sensor weight over time for methods that expose one, averaged over envs still alive."""
    names = [n for n, f in finals.items() if 'w_sensor' in f['sensor_corrupt']['out']]
    if not names:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ['sensor_corrupt', 'image_corrupt']):
        for n in names:
            f = finals[n][mode]
            alive = f['alive']
            w = (f['out']['w_sensor'] * alive).sum(1) / np.maximum(alive.sum(1), 1)
            t = np.arange(len(w))
            ax.plot(t[alive.sum(1) > 0], w[alive.sum(1) > 0], label=n)
        ax.axvspan(*cfg['corrupt_window'], color='red', alpha=0.1, label='corruption window')
        ax.set(title=f'{mode}', xlabel='Step', ylim=(-0.02, 1.02))
        ax.grid(True)
    axes[0].set_ylabel('Weight on sensor')
    axes[1].legend(fontsize=8)
    fig.suptitle('Adaptive gain: how much each method trusts the sensor')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True

# ------------------------------------------------------------------------------
# demo GIF


def make_gif(name, trace, out_path, stride=2):
    out, alive = trace['out'], trace['alive'][:, 0]
    bad = trace['sensor_bad'][:, 0] | trace['image_bad'][:, 0]
    bad_label = 'SENSOR CORRUPTED' if trace['sensor_bad'][:, 0].any() else 'IMAGE CORRUPTED'
    T = int(min(len(alive), alive.sum() + 15))
    W, Hh, ground = 360, 200, 140
    ppu = W / 4.8

    def to_px(x, y):
        return (x + 2.4) * ppu, ground - y * ppu

    frames = []
    for t in range(0, T, stride):
        im = Image.new('RGB', (W, Hh), 'white')
        d = ImageDraw.Draw(im)
        d.line([to_px(-2.4, 0), to_px(2.4, 0)], fill=(150, 150, 150), width=2)
        for state, color, width in [(out['x_est'][t, 0], (190, 190, 190), 3), (out['s'][t, 0], (180, 110, 50), 6)]:
            x, th = float(state[0]), float(state[2])
            cx, cy = to_px(x, 0)
            if color[0] == 180:
                d.rectangle([cx - 0.25 * ppu, cy - 0.15 * ppu, cx + 0.25 * ppu, cy + 0.15 * ppu], fill=(60, 60, 60))
            d.line([(cx, cy), to_px(x + np.sin(th), np.cos(th))], fill=color, width=width)
        d.text((6, 4), f'{name}  t={t}', fill='black')
        d.text((6, 18), 'gray = estimated state', fill=(130, 130, 130))
        if bad[t]:
            d.text((W - 120, 4), bad_label, fill='red')
        if not alive[t]:
            d.text((W // 2 - 20, 40), 'FAILED', fill='red')
        if 'w_sensor' in out:
            w = float(np.clip(out['w_sensor'][t, 0], 0, 1))
            d.text((6, Hh - 34), 'trust: sensor | image', fill='black')
            d.rectangle([6, Hh - 18, 206, Hh - 6], outline='black')
            d.rectangle([6, Hh - 18, 6 + 200 * w, Hh - 6], fill=(70, 130, 200))
        frames.append(im)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=40, loop=0)

# ------------------------------------------------------------------------------
# records + README


EQUATIONS = r"""## Equations

**Simulator and observations.** State $s_t = [x, \dot x, \theta, \dot\theta]$, normalized by $S = \mathrm{diag}(1, 1, 0.2, 1)$.

$$s_{t+1} = f(s_t, u_t) \quad \text{(CartPole, Euler, } \Delta t = 0.02\text{)}, \qquad u_t \in [-1, 1]$$

$$z_t = s_t + \sigma_t S \epsilon_t, \quad \epsilon_t \sim \mathcal N(0, I), \quad \sigma_t \in \{0.02 \text{ (nominal)},\ 1.0 \text{ (sensor corrupt)}\}$$

$$o_t = \mathrm{render}(s_t) + \text{noise} \quad \text{(soft, differentiable; 85\% uniform noise when image corrupt)}$$

**Image encoder** (two stacked frames give velocity):

$$(h_t,\ \mu^{img}_t,\ \log v^{img}_t) = \mathrm{CNN}([o_t, o_{t-1}])$$

**Fusion methods** ($\phi_t$ is the policy input):

| Method | Fusion |
|---|---|
| `sensor_only` | $\phi_t = S^{-1} z_t$ |
| `image_only` | $\phi_t = h_t$ |
| `concat` | $\phi_t = [S^{-1} z_t,\ h_t]$ |
| `gated` | $g_t = \sigma(W[e(z_t), h_t]),\quad \phi_t = g_t \odot e(z_t) + (1 - g_t) \odot h_t$ |
| `gru` | $b_t = \mathrm{GRU}(b_{t-1}, [S^{-1} z_t, h_t]),\quad \phi_t = b_t$ |
| `poe` | $\hat x_t = \dfrac{z_t / v^s_t + \mu^{img}_t / v^{img}_t}{1 / v^s_t + 1 / v^{img}_t},\quad v^s_t = S^2 \exp \mathrm{NN}_s([z_t,\ z_t - \mu^{img}_t])$ |
| `fixed_kf`, `learned_kf` | Kalman filter below |

**Kalman filter** with learned linear dynamics $(A, B)$:

$$\hat x^-_t = A \hat x_{t-1} + B u_{t-1}, \qquad P^-_t = A P_{t-1} A^\top + Q$$

$$P_t = \left( (P^-_t)^{-1} + R_{s,t}^{-1} + R_{i,t}^{-1} \right)^{-1}, \qquad \hat x_t = P_t \left( (P^-_t)^{-1} \hat x^-_t + R_{s,t}^{-1} z_t + R_{i,t}^{-1} \mu^{img}_t \right)$$

Equivalently, with gains $K_{s,t} = P_t R_{s,t}^{-1}$ and $K_{i,t} = P_t R_{i,t}^{-1}$:

$$\hat x_t = (I - K_{s,t} - K_{i,t})\, \hat x^-_t + K_{s,t}\, z_t + K_{i,t}\, \mu^{img}_t$$

- `fixed_kf`: $Q, R_s, R_i$ are fixed constants, so the gains converge to constants.
- `learned_kf`: the noise is predicted at every step, so the gains adapt to the inputs:

$$R_{s,t} = S^2 \mathrm{diag}\, \exp \mathrm{NN}_s([z_t - \hat x^-_t,\ z_t - \mu^{img}_t]), \qquad R_{i,t} = \mathrm{diag}(v^{img}_t), \qquad Q = S^2 \mathrm{diag}\, e^{q}$$

The plotted *weight on sensor* is $\overline{\mathrm{diag}}(K_{s,t})$ (for `poe`, $v^{img} / (v^s + v^{img})$; for `gated`, $\bar g_t$). For estimator methods, $\phi_t = [S^{-1} \hat x_t,\ 0.1 \log(\mathrm{diag}\, P_t / S^2)]$.

**Policy, reward, loss.**

$$u_t = \tanh \pi_\theta(\phi_t), \qquad r_t = 1 - \left(\frac{\theta_t}{\theta_{max}}\right)^2 - 0.5 \left(\frac{x_t}{x_{max}}\right)^2 - 0.01 (\dot x_t^2 + \dot\theta_t^2) - 0.01 u_t^2$$

$$\mathcal L = -\frac{1}{HN} \sum_{t,n} r_t \;+\; \lambda_{aux} \sum_t \left( \lVert S^{-1}(\hat x_t - s_t) \rVert^2 + \lVert S^{-1}(\mu^{img}_t - s_t) \rVert^2 \right) \;+\; \lambda_{nll} \sum_t \tfrac12 \left( \log v_t + \frac{e_t^2}{v_t} \right)$$

The NLL term is applied to each source's predicted variance $v_t$ and its actual error $e_t$ (`poe`, `learned_kf`); it teaches the variance heads, and hence the gains, how reliable each source is.

**Gradient.** $\nabla_\theta \mathcal L$ is computed by backpropagating through $f$, $\mathrm{render}$, the fusion module and the policy over $H$-step windows (analytic policy gradient), with the gradient stopped between windows. All parameters (CNN, noise networks, $A$, $B$, $Q$, policy) are trained jointly.
"""


def append_records(rows):
    new = not os.path.exists(RECORDS)
    with open(RECORDS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_records():
    if not os.path.exists(RECORDS):
        return []
    with open(RECORDS) as f:
        return list(csv.DictReader(f))


def write_readme(cfg, has_gain_plot):
    records = read_records()
    latest = {}
    for r in records:
        latest[r['method']] = r
    order = [n for n in METHODS if n in latest]

    lines = ['# Fusion via differentiable simulation (CartPole, JAX)', '',
             'Sensor state and rendered frames are fused by a learned module and trained **end to end**: '
             'the return is backpropagated through the JAX CartPole simulator, the soft renderer and the fusion '
             'module (short-horizon analytic policy gradients). Generated by `src/fusion_e2e/train.py`.', '',
             '## Setup', '',
             f"- **Env**: CartPole dynamics with continuous force, {cfg['episode_len']}-step training episodes, "
             f"{cfg['horizon']}-step gradient windows, {cfg['n_envs']} parallel envs.",
             f"- **Observations**: noisy 4D sensor state (σ = {env.SENSOR_NOISE} × scale) and two stacked "
             f"{env.IMG}×{env.IMG} soft-rendered frames with pixel noise.",
             f"- **Corruption**: during a window the sensor noise jumps to σ = {env.SENSOR_BAD_NOISE} × scale "
             f"(*sensor_corrupt*) or the frame becomes {int(env.IMG_BAD_MIX * 100)}% noise (*image_corrupt*). "
             'Training samples a random mode and window per env.',
             f"- **Eval**: survival steps out of {cfg['eval_len']} (pole within 12°, cart within 2.4), "
             f"{cfg['eval_envs']} envs, corruption window steps {cfg['corrupt_window'][0]}–{cfg['corrupt_window'][1]}.",
             f"- **Loss**: −reward + {cfg['aux_weight']} × state-estimation MSE (true state is free in simulation) "
             f"+ {cfg['nll_weight']} × Gaussian NLL of predicted variances.",
             '', '## Methods', '', '| Method | Description |', '|---|---|']
    lines += [f'| `{m.name}` | {m.description} |' for m in METHODS.values()]
    import analyze
    lines += ['', EQUATIONS] + analyze.readme_section(list(METHODS)) + ['## Results (latest run of each method)', '',
              '| Method | Clean | Sensor corrupt | Image corrupt | Mean | Est. RMSE | Train (min) | Run |',
              '|---|---|---|---|---|---|---|---|']
    for n in order:
        r = latest[n]
        lines.append(f"| `{n}` | {float(r['clean']):.0f} | {float(r['sensor_corrupt']):.0f} | "
                     f"{float(r['image_corrupt']):.0f} | **{float(r['mean']):.0f}** | {float(r['est_rmse']):.3f} | "
                     f"{float(r['train_min']):.1f} | {r['run_id']} |")
    lines += ['', 'Survival steps, max 500. Est. RMSE is the normalized state-estimate error while alive.', '',
              '![By condition](plots/eval_by_condition.png)', '', '![Learning curves](plots/learning_curves.png)', '']
    if has_gain_plot or os.path.exists(os.path.join(PLOTS_DIR, 'gain_traces.png')):
        lines += ['## Adaptive gain', '', 'Weight each method puts on the sensor over an evaluation episode '
                  '(shaded = corruption window).', '', '![Gain traces](plots/gain_traces.png)', '']
    lines += ['## Demos', '', 'Brown = true pole, gray = the method\'s state estimate, bar = trust in the sensor.', '',
              '| Method | Sensor corrupt | Image corrupt |', '|---|---|---|']
    for n in order:
        lines.append(f'| `{n}` | ![](demo/{n}_sensor_corrupt.gif) | ![](demo/{n}_image_corrupt.gif) |')
    notes = os.path.join(EXP_DIR, 'notes.md')
    if os.path.exists(notes):
        lines += ['', '## Notes', '', open(notes).read().strip()]
    lines += ['', '## Run history', '', f'All {len(records)} records: [runs/records.csv](runs/records.csv) '
              '(appended, never overwritten). Per-run config, histories and params: `runs/<run_id>/`.', '',
              '| Run | Method | Iters | Seed | Clean | Sensor | Image | Mean |', '|---|---|---|---|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['run_id']} | `{r['method']}` | {r['iters']} | {r['seed']} | {float(r['clean']):.0f} | "
                     f"{float(r['sensor_corrupt']):.0f} | {float(r['image_corrupt']):.0f} | {float(r['mean']):.0f} |")
    lines += ['', '## Reproduce', '', '```bash', 'srep-venv/bin/python src/fusion_e2e/train.py            # all methods',
              'srep-venv/bin/python src/fusion_e2e/train.py --methods learned_kf --iters 100', '```', '']
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as f:
        f.write('\n'.join(lines))

# ------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run-id', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--eval-every', type=int, default=DEFAULT_CFG['eval_every'])
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG, eval_every=args.eval_every)
    # seed 0 owns the shared demo GIFs, gain plot and live plot; other seeds keep theirs in runs/<run_id>/
    primary = args.seed == 0
    run_dir = os.path.join(RUNS_DIR, args.run_id)
    for d in (PLOTS_DIR, DEMO_DIR, run_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({**cfg, **vars(args)}, f, indent=2)

    live_hist, finals = {}, {}
    for name in args.methods:
        print(f'=== {name} ===', flush=True)
        live_path = os.path.join(PLOTS_DIR if primary else run_dir, 'live_training.png')
        params, hist, final, train_min = train_method(name, cfg, args.iters, args.seed, live_hist, live_path)
        finals[name] = final

        summary = {m: final[m]['survival'] for m in MODES}
        row = {'run_id': args.run_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M'), 'method': name,
               'iters': args.iters, 'seed': args.seed, **{m: f'{v:.1f}' for m, v in summary.items()},
               'mean': f'{np.mean(list(summary.values())):.1f}',
               'est_rmse': f"{np.mean([final[m]['est_rmse'] for m in MODES]):.4f}", 'train_min': f'{train_min:.2f}'}
        append_records([row])
        with open(os.path.join(run_dir, f'{name}.json'), 'w') as f:
            json.dump({'history': hist, 'final': {m: {k: v for k, v in final[m].items() if k in ('survival', 'est_rmse')}
                                                  for m in MODES}}, f)
        with open(os.path.join(run_dir, f'{name}_params.pkl'), 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        for mode in ('sensor_corrupt', 'image_corrupt'):
            make_gif(name, final[mode], os.path.join(DEMO_DIR if primary else run_dir, f'{name}_{mode}.gif'))
        print(f'[{name}] final ' + '  '.join(f'{m} {v:.0f}' for m, v in summary.items())
              + f'  ({train_min:.1f} min)', flush=True)

    out_dir = PLOTS_DIR if primary else run_dir
    plot_live(live_hist, os.path.join(out_dir, 'learning_curves.png'))
    plot_gain_traces(finals, cfg, os.path.join(out_dir, 'gain_traces.png'))
    refresh_outputs()
    print(f'Done. See {EXP_DIR}/README.md', flush=True)


def refresh_outputs():
    """Rebuild the cross-run outputs (condition bar chart, sample-efficiency plot, README) from runs/."""
    latest = {}
    for r in read_records():
        latest[r['method']] = r
    if latest:
        plot_by_condition({n: latest[n] for n in METHODS if n in latest},
                          os.path.join(PLOTS_DIR, 'eval_by_condition.png'))
    write_readme(DEFAULT_CFG, False)


if __name__ == '__main__':
    main()
