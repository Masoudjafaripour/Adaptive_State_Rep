"""Dreamer-style world model with adaptive multimodal fusion on DeepMind Control (pixels + proprio).

World model: Gaussian RSSM trained on replayed real data (reconstruction + reward + KL), with the
posterior fusion from models.FUSIONS. Behaviour: actor-critic trained in imagination, the actor by
backpropagating λ-returns through the learned dynamics. Training episodes get a random sensor or
image corruption window; evaluation runs one episode per condition (clean / sensor / image corrupt).

Outputs, next to this file:
    plots/live_training.png             redrawn after every evaluation (matplotlib Agg)
    plots/sample_efficiency_<task>.png  eval return vs env steps, mean ± std over seeds
    plots/gain_traces_<task>.png        trust in proprio vs image over an eval episode
    demo/<task>_<method>_<mode>.gif     observed frames of the final eval episodes
    runs/<run_id>/                      config, per-method history, params
    runs/records.csv                    one row per (run, task, method); appended, never overwritten
    README.md                           regenerated from runs/ + notes.md

Usage (from the repo root):
    srep-venv/bin/python src/wm_fusion/train.py --task cartpole_swingup --methods kf concat --episodes 100
    srep-venv/bin/python src/wm_fusion/train.py --readme-only
"""
import argparse
import csv
import glob
import json
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')  # headless: plots are written to PNG files
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
from PIL import Image, ImageDraw

import envs
import models as M

EXP_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
DEMO_DIR  = os.path.join(EXP_DIR, 'demo')
RUNS_DIR  = os.path.join(EXP_DIR, 'runs')
RECORDS   = os.path.join(RUNS_DIR, 'records.csv')
MODES = envs.MODES
RECORD_FIELDS = ['run_id', 'timestamp', 'task', 'method', 'episodes', 'seed',
                 'clean', 'sensor_corrupt', 'image_corrupt', 'mean', 'auc', 'train_min']

DEFAULT_CFG = dict(
    prefill=5, updates_per_episode=100, batch=16, seq_len=50, horizon=15,
    gamma=0.99, lam=0.95, kl_balance=0.8, free_nats=1.0, kl_scale=1.0, entropy=1e-4,
    wm_lr=3e-4, actor_lr=8e-5, critic_lr=8e-5, grad_clip=100.0, eval_every=5,
)

# ------------------------------------------------------------------------------
# replay


class Replay:
    def __init__(self, rng):
        self.episodes, self.rng = [], rng

    def add(self, ep):
        self.episodes.append({k: np.asarray(v) for k, v in ep.items()})

    def sample(self, batch, T):
        out = {k: [] for k in self.episodes[0]}
        for _ in range(batch):
            ep = self.episodes[self.rng.integers(len(self.episodes))]
            i = self.rng.integers(0, len(ep['reward']) - T + 1)
            for k in out:
                out[k].append(ep[k][i:i + T])
        return {k: np.swapaxes(np.stack(v), 0, 1) for k, v in out.items()}   # (T, B, ...)

# ------------------------------------------------------------------------------
# agent


def build(fusion, cfg, prop_dim, action_dim):
    nll_w = M.FUSIONS[fusion][3]
    clip = optax.clip_by_global_norm(cfg['grad_clip'])
    opt_wm = optax.chain(clip, optax.adam(cfg['wm_lr'], eps=1e-5))
    opt_actor = optax.chain(clip, optax.adam(cfg['actor_lr'], eps=1e-5))
    opt_critic = optax.chain(clip, optax.adam(cfg['critic_lr'], eps=1e-5))
    sg = jax.lax.stop_gradient

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        params = {'wm': M.wm_init(k1, fusion, prop_dim, action_dim),
                  'actor': M.actor_init(k2, action_dim), 'critic': M.critic_init(k3)}
        opts = {'wm': opt_wm.init(params['wm']), 'actor': opt_actor.init(params['actor']),
                'critic': opt_critic.init(params['critic'])}
        return params, opts

    def wm_loss(wm, batch, key):
        obs = {'image': batch['image'], 'proprio': batch['proprio']}
        hs, zs, (mu_p, std_p, mu, std), info = M.observe(wm, fusion, obs, batch['action'], key)
        feat = jnp.concatenate([hs, zs], -1)
        img_t = batch['image'].astype(jnp.float32) / 255.0 - 0.5
        img_loss = jnp.mean(0.5 * jnp.sum((M.img_dec(wm['dec_img'], feat) - img_t) ** 2, (-3, -2, -1)))
        prop_loss = jnp.mean(0.5 * jnp.sum((M.mlp(wm['dec_prop'], feat) - M.symlog(batch['proprio'])) ** 2, -1))
        rew_loss = jnp.mean(0.5 * (M.mlp(wm['reward'], feat)[..., 0] - batch['reward']) ** 2)
        a = cfg['kl_balance']
        kl = a * M.kl_gauss(sg(mu), sg(std), mu_p, std_p) + (1 - a) * M.kl_gauss(mu, std, sg(mu_p), sg(std_p))
        kl = jnp.mean(jnp.maximum(kl, cfg['free_nats']))
        nll = jnp.mean(info['nll'])
        loss = img_loss + prop_loss + rew_loss + cfg['kl_scale'] * kl + nll_w * nll
        metrics = {'img': img_loss, 'prop': prop_loss, 'rew': rew_loss, 'kl': kl, 'nll': nll}
        return loss, (sg(hs), sg(zs), metrics)

    def imagine(actor_p, wm, h, z, key):
        def body(c, k):
            h, z = c
            k1, k2 = jax.random.split(k)
            a, ent = M.actor(actor_p, jnp.concatenate([h, z], -1), k1)
            h, mu_p, std_p = M.img_step(wm, h, z, a)
            z = mu_p + std_p * jax.random.normal(k2, mu_p.shape)
            return (h, z), (jnp.concatenate([h, z], -1), ent)
        _, (feats, ents) = jax.lax.scan(body, (h, z), jax.random.split(key, cfg['horizon']))
        return feats, ents                                     # (H, N, F), (H, N)

    def lambda_returns(rewards, values):
        """rewards/values: (H, N) for imagined steps 1..H -> returns for steps 0..H-1."""
        def body(nxt, xs):
            r, v = xs
            ret = r + cfg['gamma'] * ((1 - cfg['lam']) * v + cfg['lam'] * nxt)
            return ret, ret
        _, rets = jax.lax.scan(body, values[-1], (rewards, values), reverse=True)
        return rets

    def ac_losses(actor_p, critic_p, wm, h0, z0, key):
        wm = sg(wm)
        feats, ents = imagine(actor_p, wm, h0, z0, key)
        rewards = M.mlp(wm['reward'], feats)[..., 0]
        values = M.critic(sg(critic_p), feats)
        rets = lambda_returns(rewards, values)
        actor_loss = -jnp.mean(rets) - cfg['entropy'] * jnp.mean(ents)
        start = jnp.concatenate([h0, z0], -1)[None]
        feats_v = sg(jnp.concatenate([start, feats[:-1]], 0))
        return actor_loss, (feats_v, sg(rets))

    def critic_loss(critic_p, feats_v, rets):
        return jnp.mean(0.5 * (M.critic(critic_p, feats_v) - rets) ** 2)

    @jax.jit
    def train_step(params, opts, batch, key):
        k1, k2 = jax.random.split(key)
        (wl, (hs, zs, metrics)), g = jax.value_and_grad(wm_loss, has_aux=True)(params['wm'], batch, k1)
        upd, opts_wm = opt_wm.update(g, opts['wm'], params['wm'])
        wm = optax.apply_updates(params['wm'], upd)
        h0, z0 = hs.reshape(-1, M.DETER), zs.reshape(-1, M.STOCH)
        (al, (feats_v, rets)), ga = jax.value_and_grad(ac_losses, has_aux=True)(
            params['actor'], params['critic'], wm, h0, z0, k2)
        upd, opts_a = opt_actor.update(ga, opts['actor'], params['actor'])
        actor_p = optax.apply_updates(params['actor'], upd)
        cl, gc = jax.value_and_grad(critic_loss)(params['critic'], feats_v, rets)
        upd, opts_c = opt_critic.update(gc, opts['critic'], params['critic'])
        critic_p = optax.apply_updates(params['critic'], upd)
        metrics = {**metrics, 'wm': wl, 'actor': al, 'critic': cl, 'ret': jnp.mean(rets)}
        return ({'wm': wm, 'actor': actor_p, 'critic': critic_p},
                {'wm': opts_wm, 'actor': opts_a, 'critic': opts_c}, metrics)

    @jax.jit
    def policy_step(params, state, obs, key, explore):
        """One env step: filter the new observation into the latent, then act. Unbatched obs."""
        h, z, a_prev = state
        wm = params['wm']
        e_img, e_prop = M.embed(wm, {'image': obs['image'][None], 'proprio': obs['proprio'][None]})
        k1, k2 = jax.random.split(key)
        h, z, _, info = M.obs_step(wm, fusion, h, z, a_prev, e_img, e_prop, k1)
        feat = jnp.concatenate([h, z], -1)
        a_explore, _ = M.actor(params['actor'], feat, k2)
        a_mean, _ = M.actor(params['actor'], feat)
        a = jnp.where(explore, a_explore, a_mean)
        return a[0], (h, z, a), {k: v[0] for k, v in info.items() if k in ('w_sensor', 'w_image')}

    def init_state():
        return (jnp.zeros((1, M.DETER)), jnp.zeros((1, M.STOCH)), jnp.zeros((1, action_dim)))

    return dict(init=init, train_step=train_step, policy_step=policy_step, init_state=init_state)


def run_episode(env, agent, params, corruption, key, explore, random_policy=False, rng=None, keep_frames=False):
    """-> (episode dict for replay, true return, trace for plots)."""
    obs = env.reset()
    state = agent['init_state']()
    o, sb, ib = corruption(obs, 0)
    ep = {'image': [o['image']], 'proprio': [o['proprio']], 'action': [np.zeros(env.action_dim, np.float32)],
          'reward': [0.0]}
    trace = {'w_sensor': [], 'sensor_bad': [sb], 'image_bad': [ib], 'frames': [o['image']] if keep_frames else []}
    total = 0.0
    for t in range(env.max_steps):
        key, k = jax.random.split(key)
        if random_policy:
            a = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
        else:
            a, state, info = agent['policy_step'](params, state, o, k, explore)
            a = np.asarray(a)
            if 'w_sensor' in info:
                trace['w_sensor'].append(float(info['w_sensor']))
        obs, r, done = env.step(a)
        total += r
        o, sb, ib = corruption(obs, t + 1)
        for k_, v in (('image', o['image']), ('proprio', o['proprio']), ('action', a), ('reward', r)):
            ep[k_].append(v)
        trace['sensor_bad'].append(sb)
        trace['image_bad'].append(ib)
        if keep_frames:
            trace['frames'].append(o['image'])
        if done:
            break
    ep['reward'] = np.asarray(ep['reward'], np.float32)
    return ep, total, trace


def train_method(task, fusion, cfg, args, live_hist, live_path):
    env = envs.make_env(task, args.backend, seed=args.seed)
    eval_env = envs.make_env(task, args.backend, seed=10_000 + args.seed)
    rng = np.random.default_rng(args.seed)
    obs0 = env.reset()
    agent = build(fusion, cfg, obs0['proprio'].shape[0], env.action_dim)
    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    params, opts = agent['init'](k)
    replay = Replay(rng)
    steps_per_ep = env.max_steps * env.action_repeat

    hist = {'episode': [], 'env_steps': [], 'train_return': [], 'eval_steps': [], **{f'eval_{m}': [] for m in MODES}}
    live_hist[f'{task}/{fusion}'] = hist
    start = time.time()
    for ep_i in range(args.episodes):
        corr = envs.Corruption.sample(rng, env.max_steps)
        key, k = jax.random.split(key)
        ep, ret, _ = run_episode(env, agent, params, corr, k, True, random_policy=ep_i < cfg['prefill'], rng=rng)
        replay.add(ep)
        metrics = {}
        if ep_i >= cfg['prefill'] - 1:
            for _ in range(cfg['updates_per_episode']):
                batch = replay.sample(cfg['batch'], cfg['seq_len'])
                key, k = jax.random.split(key)
                params, opts, metrics = agent['train_step'](params, opts, batch, k)
        hist['episode'].append(ep_i + 1)
        hist['env_steps'].append((ep_i + 1) * steps_per_ep)
        hist['train_return'].append(ret)
        msg = f"[{task}/{fusion}] ep {ep_i + 1}/{args.episodes}  train {ret:.0f}  ({corr.mode})"
        if metrics:
            msg += '  ' + '  '.join(f'{k} {float(v):.2f}' for k, v in metrics.items())
        if (ep_i + 1) % cfg['eval_every'] == 0 or ep_i == args.episodes - 1:
            ev = evaluate(eval_env, agent, params, args.seed)
            hist['eval_steps'].append((ep_i + 1) * steps_per_ep)
            for m in MODES:
                hist[f'eval_{m}'].append(ev[m]['return'])
            msg += '  | eval ' + '  '.join(f"{m} {ev[m]['return']:.0f}" for m in MODES)
            plot_live(live_hist, live_path)
        print(msg, flush=True)

    train_min = (time.time() - start) / 60
    final = evaluate(eval_env, agent, params, args.seed, keep_frames=True)
    return params, hist, final, train_min


def evaluate(env, agent, params, seed, keep_frames=False):
    res = {}
    for i, mode in enumerate(MODES):
        rng = np.random.default_rng(20_000 + seed + i)
        corr = envs.Corruption.fixed(mode, env.max_steps, rng)
        _, ret, trace = run_episode(env, agent, params, corr, jax.random.PRNGKey(seed + i), False,
                                    keep_frames=keep_frames)
        res[mode] = {'return': ret, 'trace': trace}
    return res

# ------------------------------------------------------------------------------
# plots + GIFs


def plot_live(all_hist, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in all_hist.items():
        if not h['episode']:
            continue
        line, = axes[0].plot(np.array(h['env_steps']) / 1e3, h['train_return'], alpha=0.6, label=name)
        if h['eval_steps']:
            mean = np.mean([h[f'eval_{m}'] for m in MODES], axis=0)
            axes[1].plot(np.array(h['eval_steps']) / 1e3, mean, 'o-', color=line.get_color(), label=name)
    axes[0].set(xlabel='Env steps (k)', ylabel='Episode return', title='Training episodes (random corruption)')
    axes[1].set(xlabel='Env steps (k)', ylabel='Return (mean of 3 conditions)', title='Evaluation')
    for ax in axes:
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_gain_traces(task, finals, out_path):
    names = [n for n, f in finals.items() if f['sensor_corrupt']['trace']['w_sensor']]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ['sensor_corrupt', 'image_corrupt']):
        for n in names:
            ax.plot(finals[n][mode]['trace']['w_sensor'], label=n)
        bad = np.array(finals[names[0]][mode]['trace']['sensor_bad' if mode == 'sensor_corrupt' else 'image_bad'])
        idx = np.where(bad)[0]
        if len(idx):
            ax.axvspan(idx[0], idx[-1], color='red', alpha=0.1, label='corruption window')
        ax.set(title=mode, xlabel='Agent step', ylim=(-0.02, 1.02))
        ax.grid(True)
    axes[0].set_ylabel('Posterior gain on proprio (mean over latent dims)')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{task}: adaptive trust in proprioception vs pixels')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def make_gif(name, trace, ret, out_path, stride=2, scale=3):
    frames, w = trace['frames'], trace['w_sensor']
    bad = np.array(trace['sensor_bad']) | np.array(trace['image_bad'])
    label = 'SENSOR CORRUPTED' if any(trace['sensor_bad']) else 'IMAGE CORRUPTED'
    out = []
    for t in range(0, len(frames), stride):
        im = Image.fromarray(frames[t]).resize((64 * scale, 64 * scale), Image.NEAREST)
        canvas = Image.new('RGB', (64 * scale, 64 * scale + 40), 'white')
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, 64 * scale + 2), f'{name} t={t} return={ret:.0f}', fill='black')
        if bad[t]:
            d.text((4, 4), label, fill='red')
        if t > 0 and t - 1 < len(w):
            d.rectangle([4, 64 * scale + 20, 4 + 120, 64 * scale + 32], outline='black')
            d.rectangle([4, 64 * scale + 20, 4 + 120 * float(np.clip(w[t - 1], 0, 1)), 64 * scale + 32],
                        fill=(70, 130, 200))
            d.text((130, 64 * scale + 20), 'proprio trust', fill='black')
        out.append(canvas)
    out[0].save(out_path, save_all=True, append_images=out[1:], duration=50, loop=0)

# ------------------------------------------------------------------------------
# records, aggregation, README


def append_record(row):
    new = not os.path.exists(RECORDS)
    with open(RECORDS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_records():
    if not os.path.exists(RECORDS):
        return []
    with open(RECORDS) as f:
        return list(csv.DictReader(f))


def auc(h):
    """Average eval return (mean of 3 conditions) over training, normalized by the 1000 max."""
    x = np.array(h['eval_steps'], float)
    y = np.mean([h[f'eval_{m}'] for m in MODES], axis=0)
    return float(y[0] / 1000 if len(x) < 2 else np.trapezoid(y, x) / (x[-1] - x[0]) / 1000)


def load_histories():
    """-> {(task, method, episodes): [history, ...]} over all runs."""
    groups = {}
    for path in glob.glob(os.path.join(RUNS_DIR, '*', '*__*.json')):
        task, method = os.path.basename(path)[:-5].split('__')
        d = json.load(open(path))
        groups.setdefault((task, method, d['episodes']), []).append(d)
    return groups


def plot_sample_efficiency(groups, task, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for (t, method, _), runs in sorted(groups.items()):
        if t != task:
            continue
        grid = min((np.array(r['history']['eval_steps']) for r in runs), key=len)
        for ax, field in zip(axes, ['mean', 'eval_sensor_corrupt']):
            ys = []
            for r in runs:
                h = r['history']
                y = np.mean([h[f'eval_{m}'] for m in MODES], 0) if field == 'mean' else np.array(h[field])
                ys.append(np.interp(grid, h['eval_steps'], y))
            ys = np.array(ys)
            line, = ax.plot(grid / 1e3, ys.mean(0), lw=2.5 if method == 'kf' else 1.3,
                            label=f'{method} (n={len(runs)})')
            ax.fill_between(grid / 1e3, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0),
                            color=line.get_color(), alpha=0.15)
    axes[0].set(title='Mean of 3 conditions', xlabel='Env steps (k)', ylabel='Eval return')
    axes[1].set(title='Sensor corrupt', xlabel='Env steps (k)')
    for ax in axes:
        ax.grid(True)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{task}: sample efficiency (mean ± std over seeds)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


EQUATIONS = r"""## Method

**Setting.** DeepMind Control from **pixels and proprioception**: observation $o_t = (x^{img}_t, x^{prop}_t)$, image $64\times64\times3$, action repeat 2. Each training episode gets one random window of *sensor corruption* (proprio noise $\sigma = 1$) or *image corruption* (85% pixel noise). Evaluation runs one episode per condition, with the window at 30–70% of the episode.

**World model** (Dreamer-style Gaussian RSSM):

$$h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}]), \qquad p(z_t \mid h_t) = \mathcal N(\mu^-_t, \mathrm{diag}\,\sigma^{-2}_t)$$

$$e^{img}_t = \mathrm{CNN}(x^{img}_t), \qquad e^{prop}_t = \mathrm{MLP}(\mathrm{symlog}\, x^{prop}_t)$$

**Our posterior: adaptive Kalman fusion.** Each modality $j \in \{img, prop\}$ emits a Gaussian latent observation of $z_t$:

$$w^j_t = f_j(e^j_t), \qquad \log r^j_t = g_j\left([e^j_t,\ w^j_t - \mu^-_t,\ w^j_t - w^{\bar j}_t]\right)$$

Multiplying the prior by both Gaussian observations is exactly a diagonal Kalman update:

$$\sigma^2_t = \left( \frac{1}{\sigma^{-2}_t} + \frac{1}{r^{img}_t} + \frac{1}{r^{prop}_t} \right)^{-1}, \qquad \mu_t = \sigma^2_t \left( \frac{\mu^-_t}{\sigma^{-2}_t} + \frac{w^{img}_t}{r^{img}_t} + \frac{w^{prop}_t}{r^{prop}_t} \right)$$

$$\Longleftrightarrow \quad \mu_t = (1 - k^{img}_t - k^{prop}_t)\, \mu^-_t + k^{img}_t\, w^{img}_t + k^{prop}_t\, w^{prop}_t, \qquad k^j_t = \sigma^2_t / r^j_t$$

The gain $k^j_t$ is the model's per-step, per-dimension trust in modality $j$. It depends on the input through $r^j_t$: an embedding that looks corrupted, a large innovation $w^j_t - \mu^-_t$, or disagreement with the other modality all raise $r^j_t$ and shift trust to the other modality and to the dynamics prior. The standard Dreamer posterior $q(z_t \mid h_t, e^{img}_t, e^{prop}_t) = \mathrm{MLP}(\cdot)$ has no such structure.

**Innovation likelihood** (self-supervised; it is the filter's own marginal likelihood):

$$\mathcal L_{inn} = \sum_j \tfrac12 \left( \log(r^j_t + \sigma^{-2}_t) + \frac{(w^j_t - \mathrm{sg}[\mu^-_t])^2}{r^j_t + \sigma^{-2}_t} \right)$$

**World-model loss** on replayed sequences, with KL balancing ($\alpha = 0.8$) and free nats:

$$\mathcal L_{wm} = \underbrace{\lVert \hat x^{img}_t - x^{img}_t \rVert^2 + \lVert \hat x^{prop}_t - \mathrm{symlog}\, x^{prop}_t \rVert^2 + (\hat r_t - r_t)^2}_{\text{decoders from } [h_t, z_t]} + \beta\, \max\left(1,\ \mathrm{KL}_\alpha[q \,\Vert\, p]\right) + \lambda\, \mathcal L_{inn}$$

**Behaviour in imagination.** From every posterior state, roll the prior forward $H = 15$ steps with $a_\tau \sim \pi_\phi(\cdot \mid h_\tau, z_\tau)$, and compute λ-returns:

$$R^\lambda_\tau = \hat r_{\tau+1} + \gamma\left( (1 - \lambda)\, v_\psi(s_{\tau+1}) + \lambda R^\lambda_{\tau+1} \right)$$

$$\mathcal L_\pi = -\mathbb E\left[ \textstyle\sum_\tau R^\lambda_\tau \right] - \eta\, \mathcal H[\pi], \qquad \mathcal L_v = \mathbb E\left[ \tfrac12 \left( v_\psi(s_\tau) - \mathrm{sg}[R^\lambda_\tau] \right)^2 \right]$$

The actor gradient is backpropagated through the learned dynamics (DreamerV1-style analytic gradients). This is the world-model counterpart of differentiating through a simulator, and needs no differentiable environment.

**Variants.** `concat` is the standard Dreamer posterior. `poe_fixed` uses the same product form with a *constant* learned $r^j$ (fixed gain). `kf_nonll` removes $\mathcal L_{inn}$. `image_only` and `proprio_only` are single-modality baselines.
"""


def write_readme():
    records = read_records()
    groups = load_histories()
    tasks = sorted({k[0] for k in groups})
    lines = ['# World-model fusion: adaptive Kalman posteriors on DeepMind Control', '',
             '**Contribution:** a Dreamer-style world model whose posterior fuses pixels and proprioception with an '
             '*input-dependent Kalman gain*. It trusts the modality that is currently reliable, which gives robustness '
             'to sensor/image corruption. Evaluated on DeepMind Control from pixels + proprio, against the standard '
             'multimodal Dreamer posterior, single-modality agents and a fixed-gain fusion. '
             'Generated by `src/wm_fusion/train.py`.', '',
             '## Variants', '', '| Variant | Posterior |', '|---|---|']
    lines += [f'| `{n}` | {f[2]} |' for n, f in M.FUSIONS.items()]
    lines += ['', EQUATIONS, '## Results', '']
    if not groups:
        lines += ['*No runs yet.*', '']
    for task in tasks:
        plot_sample_efficiency(groups, task, os.path.join(PLOTS_DIR, f'sample_efficiency_{task}.png'))
        lines += [f'### {task}', '', 'Final eval return (max 1000), mean ± std over seeds. **AUC** = average eval '
                  'return during training / 1000 (sample efficiency).', '',
                  '| Method | Seeds | Env steps | Clean | Sensor corrupt | Image corrupt | Mean | AUC |',
                  '|---|---|---|---|---|---|---|---|']
        for (t, method, eps), runs in sorted(groups.items(), key=lambda kv: list(M.FUSIONS).index(kv[0][1])):
            if t != task:
                continue
            fin = {m: np.array([r['final'][m] for r in runs]) for m in MODES}
            mean = np.mean([fin[m] for m in MODES], 0)
            aucs = np.array([auc(r['history']) for r in runs])
            f = lambda a, d=0: f'{a.mean():.{d}f} ± {a.std():.{d}f}'
            name = f'**`{method}`**' if method == 'kf' else f'`{method}`'
            lines.append(f"| {name} | {len(runs)} | {runs[0]['history']['env_steps'][-1] / 1e3:.0f}k | "
                         f"{f(fin['clean'])} | {f(fin['sensor_corrupt'])} | {f(fin['image_corrupt'])} | "
                         f"{f(mean)} | {f(aucs, 2)} |")
        lines += ['', f'![Sample efficiency](plots/sample_efficiency_{task}.png)', '']
        if os.path.exists(os.path.join(PLOTS_DIR, f'gain_traces_{task}.png')):
            lines += [f'![Gain traces](plots/gain_traces_{task}.png)', '']
        gifs = sorted(glob.glob(os.path.join(DEMO_DIR, f'{task}_*_sensor_corrupt.gif')))
        if gifs:
            lines += ['| Method | Sensor corrupt | Image corrupt |', '|---|---|---|']
            for g in gifs:
                method = os.path.basename(g)[len(task) + 1:-len('_sensor_corrupt.gif')]
                lines.append(f'| `{method}` | ![](demo/{task}_{method}_sensor_corrupt.gif) | '
                             f'![](demo/{task}_{method}_image_corrupt.gif) |')
            lines.append('')
    notes = os.path.join(EXP_DIR, 'notes.md')
    if os.path.exists(notes):
        lines += ['## Notes', '', open(notes).read().strip(), '']
    lines += ['## Run history', '', f'All {len(records)} records: [runs/records.csv](runs/records.csv) '
              '(appended, never overwritten).', '']
    if records:
        lines += ['| Run | Task | Method | Episodes | Seed | Clean | Sensor | Image | Mean | AUC |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        lines += [f"| {r['run_id']} | {r['task']} | `{r['method']}` | {r['episodes']} | {r['seed']} | "
                  f"{float(r['clean']):.0f} | {float(r['sensor_corrupt']):.0f} | {float(r['image_corrupt']):.0f} | "
                  f"{float(r['mean']):.0f} | {float(r['auc']):.2f} |" for r in records]
    lines += ['', '## Reproduce', '', '```bash',
              'srep-venv/bin/python src/wm_fusion/train.py --task cartpole_swingup --seed 0   # all variants',
              'srep-venv/bin/python src/wm_fusion/train.py --task walker_walk --methods kf concat --episodes 500',
              'srep-venv/bin/python src/wm_fusion/train.py --backend playground ...           # MJX physics',
              'srep-venv/bin/python src/wm_fusion/train.py --readme-only                      # re-aggregate',
              '```', '']
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as f:
        f.write('\n'.join(lines))

# ------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='cartpole_swingup', choices=list(envs.TASKS))
    ap.add_argument('--methods', nargs='+', default=['kf', 'concat', 'poe_fixed', 'proprio_only', 'image_only',
                                                     'kf_nonll'], choices=list(M.FUSIONS))
    ap.add_argument('--episodes', type=int, default=100, help='1 episode = 1000 env steps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', default='dmc', choices=['dmc', 'playground'])
    ap.add_argument('--run-id', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--readme-only', action='store_true')
    for k, v in DEFAULT_CFG.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    for d in (PLOTS_DIR, DEMO_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)
    if args.readme_only:
        write_readme()
        return
    cfg = {k: getattr(args, k) for k in DEFAULT_CFG}
    run_dir = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    primary = args.seed == 0
    out_dir = PLOTS_DIR if primary else run_dir

    live_hist, finals = {}, {}
    for fusion in args.methods:
        print(f'=== {args.task} / {fusion} ===', flush=True)
        params, hist, final, train_min = train_method(
            args.task, fusion, cfg, args, live_hist, os.path.join(out_dir, 'live_training.png'))
        finals[fusion] = final
        summary = {m: final[m]['return'] for m in MODES}
        with open(os.path.join(run_dir, f'{args.task}__{fusion}.json'), 'w') as f:
            json.dump({'episodes': args.episodes, 'seed': args.seed, 'history': hist, 'final': summary}, f)
        append_record({'run_id': args.run_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M'), 'task': args.task,
                       'method': fusion, 'episodes': args.episodes, 'seed': args.seed,
                       **{m: f'{v:.1f}' for m, v in summary.items()},
                       'mean': f'{np.mean(list(summary.values())):.1f}', 'auc': f'{auc(hist):.3f}',
                       'train_min': f'{train_min:.1f}'})
        with open(os.path.join(run_dir, f'{args.task}__{fusion}_params.pkl'), 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        for mode in ('sensor_corrupt', 'image_corrupt'):
            make_gif(fusion, final[mode]['trace'], final[mode]['return'],
                     os.path.join(DEMO_DIR if primary else run_dir, f'{args.task}_{fusion}_{mode}.gif'))
        print(f'[{args.task}/{fusion}] final ' + '  '.join(f'{m} {v:.0f}' for m, v in summary.items())
              + f'  ({train_min:.1f} min)', flush=True)

    plot_gain_traces(args.task, finals, os.path.join(out_dir, f'gain_traces_{args.task}.png'))
    write_readme()
    print(f'Done. See {EXP_DIR}/README.md', flush=True)


if __name__ == '__main__':
    main()
