"""Representation learning for adaptive fusion without ground-truth state.

Compares, on the same differentiable CartPole as fusion_e2e:
    oracle_kf        fusion_e2e learned_kf, supervised with the true state (upper bound)
    naive_kf         same, with every true-state loss removed (return only)
    latent_kf        Kalman fusion in a learned latent space, trained with return +
                     innovation NLL + multi-step self-prediction + variance regularizer,
                     with random-shift augmentation (proposed)
    latent_kf_noaug  ablation: no augmentation
    latent_kf_nospr  ablation: no self-prediction loss

The latent variants never see the true state; a linear probe on the (stop-gradient) latent
is trained only to report state-estimation error and draw the GIF ghost.

Reuses the simulator, layers, training loop, evaluation, plots and GIFs from fusion_e2e.
Outputs next to this file: plots/, demo/, runs/<run_id>/, runs/records.csv, README.md.

Usage (from the repo root):
    srep-venv/bin/python src/latent_fusion/train.py --seed 0
    srep-venv/bin/python src/latent_fusion/train.py --variants latent_kf naive_kf --iters 100
    srep-venv/bin/python src/latent_fusion/train.py --readme-only
"""
import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(os.path.dirname(EXP_DIR), 'fusion_e2e')

# load fusion_e2e/train.py under its own name (both files are called train.py)
_spec = importlib.util.spec_from_file_location('fusion_e2e_train', os.path.join(FE_DIR, 'train.py'))
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)                    # also puts FE_DIR on sys.path
sys.path.insert(0, EXP_DIR)

import cartpole_jax as env                      # noqa: E402
from fusion import METHODS, dense, dense_init, mlp, mlp_init   # noqa: E402
import methods as latent                        # noqa: E402

PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
DEMO_DIR  = os.path.join(EXP_DIR, 'demo')
RUNS_DIR  = os.path.join(EXP_DIR, 'runs')
RECORDS   = os.path.join(RUNS_DIR, 'records.csv')
MODES = fe.MODES
S = env.STATE_SCALE

DEFAULT_CFG = dict(fe.DEFAULT_CFG, spr_steps=3, spr_weight=1.0, nll_weight=0.1, vic_weight=1.0,
                   probe_weight=1.0, shift_pad=2)

VARIANTS = {
    'oracle_kf':       dict(family='state', state_sup=True, aug=False,
                            description='fusion_e2e `learned_kf`: state-space KF, supervised with the true state '
                                        '(MSE + NLL). Upper bound: needs privileged information.'),
    'naive_kf':        dict(family='state', state_sup=False, aug=False,
                            description='Same model with every true-state loss removed; trained on the return only.'),
    'latent_kf':       dict(family='latent', aug=True, spr=True,
                            description='**Proposed.** Kalman fusion in a learned 16-d latent space; return + '
                                        'innovation NLL + self-prediction + variance regularizer + random shifts. '
                                        'No true state.'),
    'latent_kf_noaug': dict(family='latent', aug=False, spr=True,
                            description='Ablation: `latent_kf` without random-shift augmentation.'),
    'latent_kf_nospr': dict(family='latent', aug=True, spr=False,
                            description='Ablation: `latent_kf` without the self-prediction loss.'),
}

# ------------------------------------------------------------------------------


def random_shift(key, frames, pad):
    """DrQ augmentation: shift each env's frame stack by up to `pad` pixels (edge padding)."""
    n, H, W, C = frames.shape
    padded = jnp.pad(frames, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')
    offs = jax.random.randint(key, (n, 2), 0, 2 * pad + 1)
    return jax.vmap(lambda img, o: jax.lax.dynamic_slice(img, (o[0], o[1], 0), (H, W, C)))(padded, offs)


def _unit(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def build(name, cfg):
    v = VARIANTS[name]
    is_latent = v['family'] == 'latent'
    if is_latent:
        m_init, m_carry, m_step, feat_dim = latent.init, latent.init_carry, latent.step, latent.FEAT_DIM
    else:
        km = METHODS['learned_kf']
        m_init, m_carry, m_step, feat_dim = km.init, km.init_carry, km.step, km.feat_dim

    opt = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(cfg['grad_clip']), optax.adam(cfg['lr'])), 20)

    def init_state(key, n):
        s = env.reset(key, n)
        return (s, m_carry(n), env.render(s), jnp.zeros(n))

    def make_rollout(aug):
        def rollout(params, c, keys, s_bad, i_bad):
            def body(c, xs):
                s, mc, prev_img, u_prev = c
                key, sb, ib = xs
                k_obs, k_aug = jax.random.split(key)
                z, img = env.observe(k_obs, s, sb, ib)
                frames = jnp.stack([img, prev_img], -1)
                if aug:
                    frames = random_shift(k_aug, frames, cfg['shift_pad'])
                feat, est, mc, info = m_step(params['fusion'], mc, z, frames, u_prev)
                u = jnp.tanh(mlp(params['policy'], feat))[:, 0]
                target = jax.lax.stop_gradient(s)
                out = dict(r=env.reward(s, u), s=s, u=u, **info)
                if is_latent:
                    # linear probe on the stop-gradient latent: diagnostics only, never shapes the latent
                    out['x_est'] = dense(params['probe'], jax.lax.stop_gradient(est['m'])) * S
                    out['probe'] = jnp.mean(((out['x_est'] - target) / S) ** 2)
                    out.update({k: est[k] for k in ('m', 'm_prior', 'p_prior', 'w_s', 'r_s', 'w_i', 'r_i')})
                else:
                    out['x_est'] = est['x']
                    sup = jnp.mean(((est['x'] - target) / S) ** 2) + jnp.mean(((est['mu_img'] - target) / S) ** 2)
                    for vk, meas in (('var_sensor', z), ('var_img', est['mu_img'])):
                        var = est[vk] / S ** 2
                        sup = sup + cfg['nll_weight'] * jnp.mean(0.5 * (jnp.log(var) + ((meas - target) / S) ** 2 / var))
                    out['sup'] = sup
                return (env.step(s, u), mc, img, u), out
            return jax.lax.scan(body, c, (keys, s_bad, i_bad))
        return rollout

    train_rollout = make_rollout(v['aug'])

    def loss_fn(params, c, keys, s_bad, i_bad):
        c2, out = train_rollout(params, c, keys, s_bad, i_bad)
        rl = -jnp.mean(out['r'])
        loss, aux = rl, jnp.zeros(())
        if is_latent:
            # innovation NLL: each source's predicted noise must explain its disagreement with the prior
            nll = 0.0
            for w, r in ((out['w_s'], out['r_s']), (out['w_i'], out['r_i'])):
                var = r + out['p_prior']
                nll = nll + jnp.mean(0.5 * (jnp.log(var) + (w - jax.lax.stop_gradient(out['m_prior'])) ** 2 / var))
            # self-prediction: roll the latent dynamics k steps open-loop and match the future posterior
            spr = 0.0
            if v['spr']:
                m, u, H = out['m'], out['u'], out['m'].shape[0]
                for k in range(1, cfg['spr_steps'] + 1):
                    pred = m[:H - k]
                    for j in range(k):
                        pred = latent.predict(params['fusion'], pred, u[j:H - k + j])
                    tgt = jax.lax.stop_gradient(m[k:])
                    spr = spr + jnp.mean(jnp.sum((_unit(pred) - _unit(tgt)) ** 2, -1))
            # variance regularizer (VICReg): keeps the latent from collapsing to a constant
            std = jnp.sqrt(jnp.var(out['m'], axis=1) + 1e-4)
            vic = jnp.mean(jax.nn.relu(1.0 - std))
            aux = cfg['nll_weight'] * nll + cfg['spr_weight'] * spr + cfg['vic_weight'] * vic
            loss = loss + aux + cfg['probe_weight'] * jnp.mean(out['probe'])
        elif v['state_sup']:
            aux = cfg['aux_weight'] * jnp.mean(out['sup'])
            loss = loss + aux
        return loss, (c2, rl, aux)

    @jax.jit
    def update(params, opt_state, c, keys, s_bad, i_bad):
        (_, (c2, rl, aux)), g = jax.value_and_grad(loss_fn, has_aux=True)(params, c, keys, s_bad, i_bad)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, jax.lax.stop_gradient(c2), -rl, aux

    @jax.jit
    def reset_failed(c, key):
        mask = env.failed(c[0])
        fresh = init_state(key, mask.shape[0])
        return jax.tree_util.tree_map(
            lambda f, o: jnp.where(mask.reshape((-1,) + (1,) * (o.ndim - 1)), f, o), fresh, c)

    def init_params(key):
        k1, k2, k3 = jax.random.split(key, 3)
        p = {'fusion': m_init(k1), 'policy': mlp_init(k2, [feat_dim, 64, 64, 1], last_scale=0.1)}
        if is_latent:
            p['probe'] = dense_init(k3, latent.D, 4, 0.1)
        return p

    return dict(opt=opt, init_state=init_state, update=update, reset_failed=reset_failed,
                eval_rollout=jax.jit(make_rollout(False)), init_params=init_params)

# ------------------------------------------------------------------------------
# records + README


def append_records(rows):
    new = not os.path.exists(RECORDS)
    with open(RECORDS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fe.RECORD_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def read_records():
    if not os.path.exists(RECORDS):
        return []
    with open(RECORDS) as f:
        return list(csv.DictReader(f))


EQUATIONS = r"""## Equations

Same simulator, observations and corruption as [fusion_e2e](../fusion_e2e/README.md#equations): sensor $z_t$,
frames $o_t$, true state $s_t$, actions $u_t$. The difference is that **no loss uses $s_t$** (except the oracle).

**Encoders.** Each source maps to a $D = 16$ latent observation:

$$w^s_t = e_s(S^{-1} z_t), \qquad (w^i_t,\ \ell^i_t) = e_i(\mathrm{aug}([o_t, o_{t-1}]))$$

where $\mathrm{aug}$ is a random shift of up to 2 px (DrQ). Latents are soft-bounded, $w \leftarrow 3\tanh(w/3)$, so the innovation NLL stays finite.

**Latent prior** from learned residual dynamics, with diagonal variance:

$$m^-_t = 3\tanh\left(\left(m_{t-1} + g_\psi([m_{t-1}, u_{t-1}])\right)/3\right), \qquad p^-_t = p_{t-1} + q$$

**Per-step noise of each source**, predicted from its innovation and its disagreement with the other source:

$$r^s_t = \exp \mathrm{NN}_s([w^s_t - m^-_t,\ w^s_t - w^i_t]), \qquad r^i_t = \exp\left(\ell^i_t + \mathrm{NN}_i([w^i_t - m^-_t,\ w^i_t - w^s_t])\right)$$

**Kalman update** in latent space (elementwise):

$$p_t = \left(\frac{1}{p^-_t} + \frac{1}{r^s_t} + \frac{1}{r^i_t}\right)^{-1}, \qquad m_t = p_t\left(\frac{m^-_t}{p^-_t} + \frac{w^s_t}{r^s_t} + \frac{w^i_t}{r^i_t}\right)$$

Equivalently, $m_t = (1 - k^s_t - k^i_t)\, m^-_t + k^s_t\, w^s_t + k^i_t\, w^i_t$ with gains $k^s_t = p_t / r^s_t$ and $k^i_t = p_t / r^i_t$. The plotted *weight on sensor* is $\overline{k^s_t}$.

**Policy.** $u_t = \tanh \pi_\theta([m_t,\ 0.1 \log p_t])$.

**Self-supervised losses** (no $s_t$):

1. Innovation NLL. Each source's noise must explain its deviation from the prior; this is the Kalman filter's own marginal likelihood:

$$\mathcal L_{nll} = \sum_{j \in \{s, i\}} \tfrac12 \left( \log(r^j_t + p^-_t) + \frac{(w^j_t - \mathrm{sg}[m^-_t])^2}{r^j_t + p^-_t} \right)$$

2. Multi-step self-prediction (SPR-style). Roll the dynamics forward $k$ steps open-loop and match the future posterior:

$$\hat m_{t+k} = g^{(k)}(m_t, u_{t:t+k-1}), \qquad \mathcal L_{spr} = \sum_{k=1}^{K} \left\lVert \frac{\hat m_{t+k}}{\lVert \hat m_{t+k} \rVert} - \mathrm{sg}\left[\frac{m_{t+k}}{\lVert m_{t+k} \rVert}\right] \right\rVert^2, \qquad K = 3$$

3. Variance regularizer (VICReg). Prevents the latent from collapsing to a constant, which would trivially satisfy 1 and 2:

$$\mathcal L_{var} = \frac1D \sum_d \max\left(0,\ 1 - \sqrt{\mathrm{Var}_{batch}(m_{t,d}) + \epsilon}\right)$$

**Total loss**, backpropagated through simulator, renderer, encoders, filter and policy over $H$-step windows:

$$\mathcal L = -\frac{1}{HN}\sum r_t + \lambda_{nll}\mathcal L_{nll} + \lambda_{spr}\mathcal L_{spr} + \lambda_{var}\mathcal L_{var}$$

**Probe (diagnostic only).** $\hat x_t = W\,\mathrm{sg}[m_t]$ is fit to $s_t$ to report state-estimation error. The stop-gradient means it cannot influence the representation.

**Comparison.** `oracle_kf` adds $\lVert S^{-1}(\hat x_t - s_t)\rVert^2$ and a true-error NLL (fusion_e2e's loss). `naive_kf` keeps only $-\sum r_t$. If `latent_kf` gets close to the oracle and clearly beats the naive version, the self-supervised representation is doing the work the privileged state was doing.
"""


def write_readme(cfg):
    sys.path.insert(0, FE_DIR)
    import analyze
    records = read_records()
    latest = {}
    for r in records:
        latest[r['method']] = r
    order = [n for n in VARIANTS if n in latest]
    lines = ['# Latent fusion: adaptive gains without ground-truth state', '',
             'The fusion_e2e methods learn their state estimates and noise models from the **true simulator state**, '
             'which a real system would not have. This experiment learns the representation **self-supervised** and '
             'fuses in latent space, then compares against the privileged oracle and a naive no-supervision version. '
             'Generated by `src/latent_fusion/train.py`.', '',
             '## Variants', '', '| Variant | Description |', '|---|---|']
    lines += [f'| `{n}` | {v["description"]} |' for n, v in VARIANTS.items()]
    lines += ['', EQUATIONS]
    lines += analyze.readme_section(list(VARIANTS), runs_dir=RUNS_DIR, plots_dir=PLOTS_DIR)
    lines += ['## Results (latest run of each variant)', '']
    if order:
        lines += ['| Variant | Clean | Sensor corrupt | Image corrupt | Mean | Probe RMSE | Train (min) | Run |',
                  '|---|---|---|---|---|---|---|---|']
        for n in order:
            r = latest[n]
            lines.append(f"| `{n}` | {float(r['clean']):.0f} | {float(r['sensor_corrupt']):.0f} | "
                         f"{float(r['image_corrupt']):.0f} | **{float(r['mean']):.0f}** | {float(r['est_rmse']):.3f} | "
                         f"{float(r['train_min']):.1f} | {r['run_id']} |")
        lines += ['', 'Survival steps, max 500. RMSE is the normalized state error of the estimate '
                  '(linear probe for latent variants).', '',
                  '![By condition](plots/eval_by_condition.png)', '', '![Learning curves](plots/learning_curves.png)',
                  '', '## Adaptive gain', '', '![Gain traces](plots/gain_traces.png)', '',
                  '## Demos', '', '| Variant | Sensor corrupt | Image corrupt |', '|---|---|---|']
        lines += [f'| `{n}` | ![](demo/{n}_sensor_corrupt.gif) | ![](demo/{n}_image_corrupt.gif) |' for n in order]
    else:
        lines += ['*No runs yet.* Start with `srep-venv/bin/python src/latent_fusion/train.py --seed 0`.']
    notes = os.path.join(EXP_DIR, 'notes.md')
    if os.path.exists(notes):
        lines += ['', '## Notes', '', open(notes).read().strip()]
    lines += ['', '## Run history', '', f'All {len(records)} records: [runs/records.csv](runs/records.csv) '
              '(appended, never overwritten).', '']
    if records:
        lines += ['| Run | Variant | Iters | Seed | Clean | Sensor | Image | Mean |', '|---|---|---|---|---|---|---|---|']
        lines += [f"| {r['run_id']} | `{r['method']}` | {r['iters']} | {r['seed']} | {float(r['clean']):.0f} | "
                  f"{float(r['sensor_corrupt']):.0f} | {float(r['image_corrupt']):.0f} | {float(r['mean']):.0f} |"
                  for r in records]
    lines += ['', '## Reproduce', '', '```bash',
              'for s in 0 1 2; do srep-venv/bin/python src/latent_fusion/train.py --seed $s --run-id seed$s; done',
              'srep-venv/bin/python src/latent_fusion/train.py --readme-only   # re-aggregate', '```', '']
    with open(os.path.join(EXP_DIR, 'README.md'), 'w') as f:
        f.write('\n'.join(lines))


def refresh_outputs():
    latest = {}
    for r in read_records():
        latest[r['method']] = r
    if latest:
        fe.plot_by_condition({n: latest[n] for n in VARIANTS if n in latest},
                             os.path.join(PLOTS_DIR, 'eval_by_condition.png'))
    write_readme(DEFAULT_CFG)

# ------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--run-id', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--readme-only', action='store_true', help='only regenerate README/plots from runs/')
    args = ap.parse_args()

    for d in (PLOTS_DIR, DEMO_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)
    if args.readme_only:
        refresh_outputs()
        return

    cfg = dict(DEFAULT_CFG, eval_every=args.eval_every)
    run_dir = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({**cfg, **vars(args)}, f, indent=2)
    primary = args.seed == 0
    out_dir = PLOTS_DIR if primary else run_dir

    live_hist, finals = {}, {}
    for name in args.variants:
        print(f'=== {name} ===', flush=True)
        fns = build(name, cfg)
        params, hist, final, train_min = fe.train_method(
            name, cfg, args.iters, args.seed, live_hist, os.path.join(out_dir, 'live_training.png'), fns=fns)
        finals[name] = final
        summary = {m: final[m]['survival'] for m in MODES}
        append_records([{'run_id': args.run_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M'), 'method': name,
                         'iters': args.iters, 'seed': args.seed, **{m: f'{v:.1f}' for m, v in summary.items()},
                         'mean': f'{np.mean(list(summary.values())):.1f}',
                         'est_rmse': f"{np.mean([final[m]['est_rmse'] for m in MODES]):.4f}",
                         'train_min': f'{train_min:.2f}'}])
        with open(os.path.join(run_dir, f'{name}.json'), 'w') as f:
            json.dump({'history': hist, 'final': {m: {k: final[m][k] for k in ('survival', 'est_rmse')}
                                                  for m in MODES}}, f)
        with open(os.path.join(run_dir, f'{name}_params.pkl'), 'wb') as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        for mode in ('sensor_corrupt', 'image_corrupt'):
            fe.make_gif(name, final[mode], os.path.join(DEMO_DIR if primary else run_dir, f'{name}_{mode}.gif'))
        print(f'[{name}] final ' + '  '.join(f'{m} {v:.0f}' for m, v in summary.items())
              + f'  ({train_min:.1f} min)', flush=True)

    fe.plot_live(live_hist, os.path.join(out_dir, 'learning_curves.png'))
    fe.plot_gain_traces(finals, cfg, os.path.join(out_dir, 'gain_traces.png'))
    refresh_outputs()
    print(f'Done. See {EXP_DIR}/README.md', flush=True)


if __name__ == '__main__':
    main()
