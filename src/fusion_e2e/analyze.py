"""Aggregate all runs: final performance and sample efficiency, mean ± std over seeds.

Reads runs/<run_id>/config.json and runs/<run_id>/<method>.json (eval histories), groups
by (method, iters), and writes:
    plots/sample_efficiency.png   eval survival vs environment steps (mean ± std over seeds)
    the "Performance and sample efficiency" section of README.md (via train.write_readme)

Usage (from the repo root): srep-venv/bin/python src/fusion_e2e/analyze.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXP_DIR, 'runs')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
MODES = ['clean', 'sensor_corrupt', 'image_corrupt']
THRESHOLD = 450   # survival steps (mean of 3 conditions) counted as "solved"


def load_runs():
    """-> {(method, iters): [ {seed, steps, curve, sensor_curve, final}, ... ]}"""
    groups = {}
    for cfg_path in sorted(glob.glob(os.path.join(RUNS_DIR, '*', 'config.json'))):
        run_dir = os.path.dirname(cfg_path)
        cfg = json.load(open(cfg_path))
        steps_per_iter = cfg['n_envs'] * cfg['episode_len']
        for path in glob.glob(os.path.join(run_dir, '*.json')):
            method = os.path.basename(path)[:-5]
            if method == 'config':
                continue
            d = json.load(open(path))
            h = d['history']
            if not h['eval_iter']:
                continue
            groups.setdefault((method, cfg['iters']), []).append({
                'run_id': os.path.basename(run_dir), 'seed': cfg['seed'],
                'steps': np.array(h['eval_iter']) * steps_per_iter,
                'curve': np.mean([h[f'eval_{m}'] for m in MODES], axis=0),
                'sensor_curve': np.array(h['eval_sensor_corrupt']),
                'final': {m: d['final'][m]['survival'] for m in MODES},
            })
    return groups


def summarize(groups):
    """Per (method, iters): final mean / sensor-corrupt, AUC, steps-to-threshold; mean ± std over seeds."""
    rows = {}
    for key, runs in groups.items():
        finals = np.array([np.mean(list(r['final'].values())) for r in runs])
        sensor = np.array([r['final']['sensor_corrupt'] for r in runs])
        # AUC: average eval survival over training, normalized to [0, 1]
        auc = np.array([np.trapezoid(r['curve'], r['steps']) / (r['steps'][-1] - r['steps'][0]) / 500
                        if len(r['steps']) > 1 else r['curve'][0] / 500 for r in runs])
        hit = [r['steps'][np.argmax(r['curve'] >= THRESHOLD)] if (r['curve'] >= THRESHOLD).any() else np.nan
               for r in runs]
        rows[key] = {'n_seeds': len(runs), 'seeds': sorted(r['seed'] for r in runs),
                     'final': (finals.mean(), finals.std()), 'sensor': (sensor.mean(), sensor.std()),
                     'auc': (auc.mean(), auc.std()), 'steps_to_threshold': hit}
    return rows


def plot_sample_efficiency(groups, iters, out_path, methods_order):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for name in methods_order:
        runs = groups.get((name, iters))
        if not runs:
            continue
        # runs may use different eval_every: interpolate onto the coarsest grid
        grid = min((r['steps'] for r in runs), key=len)
        steps = grid / 1e6
        for ax, field in zip(axes, ['curve', 'sensor_curve']):
            ys = np.array([np.interp(grid, r['steps'], r[field]) for r in runs])
            mu, sd = ys.mean(0), ys.std(0)
            lw = 2.5 if name == 'learned_kf' else 1.3
            line, = ax.plot(steps, mu, lw=lw, label=f'{name} (n={len(runs)})')
            ax.fill_between(steps, mu - sd, mu + sd, color=line.get_color(), alpha=0.15)
    axes[0].axhline(THRESHOLD, color='gray', ls='--', lw=1)
    axes[0].set(title='Mean of 3 conditions', xlabel='Environment steps (M)', ylabel='Survival steps (max 500)')
    axes[1].set(title='Sensor corrupt', xlabel='Environment steps (M)')
    for ax in axes:
        ax.grid(True)
        ax.set_ylim(0, 520)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Sample efficiency ({iters} iterations, mean ± std over seeds)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def readme_section(methods_order):
    """Markdown for README; also refreshes plots/sample_efficiency.png."""
    groups = load_runs()
    if not groups:
        return []
    rows = summarize(groups)
    iters = max({k[1] for k in groups}, key=lambda i: sum(len(v) for k, v in groups.items() if k[1] == i))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_sample_efficiency(groups, iters, os.path.join(PLOTS_DIR, 'sample_efficiency.png'), methods_order)

    def fmt(ms, scale=1.0, digits=0):
        m, s = ms
        return f'{m * scale:.{digits}f} ± {s * scale:.{digits}f}'

    lines = ['## Performance and sample efficiency', '',
             f'Aggregated over all runs with {iters} iterations; mean ± std over seeds. '
             f'**AUC** = average evaluation survival during training / 500 (higher = learns faster). '
             f'**Steps to {THRESHOLD}** = environment steps until mean survival first reaches {THRESHOLD} '
             '(— = never).', '',
             '| Method | Seeds | Final mean | Final sensor corrupt | AUC | Steps to threshold (M) |',
             '|---|---|---|---|---|---|']
    for name in methods_order:
        r = rows.get((name, iters))
        if r is None:
            continue
        hits = r['steps_to_threshold']
        reached = [h for h in hits if not np.isnan(h)]
        hit_txt = (f"{np.mean(reached) / 1e6:.2f} ({len(reached)}/{len(hits)} seeds)" if reached else '—')
        lines.append(f"| `{name}` | {r['n_seeds']} | {fmt(r['final'])} | {fmt(r['sensor'])} | "
                     f"{fmt(r['auc'], digits=2)} | {hit_txt} |")
    lines += ['', '![Sample efficiency](plots/sample_efficiency.png)', '']
    return lines


if __name__ == '__main__':
    sys.path.insert(0, EXP_DIR)
    import train
    train.refresh_outputs()
    print(f'Updated {EXP_DIR}/README.md and plots/')
