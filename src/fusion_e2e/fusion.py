"""Fusion methods: noisy sensor state + rendered frames -> policy features.

Every method has the same interface (`Method`), so train.py runs them side by side:

    step(params, carry, z, frames, u_prev) -> (feat, est, carry, info)

    z       (N, 4)          noisy sensor reading
    frames  (N, IMG, IMG, 2) current and previous frame (two frames give velocity)
    u_prev  (N,)            previous action (used by the Kalman predict step)
    feat    (N, feat_dim)   input to the policy MLP
    est     {'x': (N, 4), optional 'mu_img': (N, 4)}  state estimates, trained with
            an auxiliary loss against the true simulator state; optional 'var_sensor' /
            'var_img' (N, 4) predicted noise variances, trained with a Gaussian NLL
    info    {'w_sensor': (N,)} for methods with an explicit sensor weight (gain plots)

Add new methods to METHODS. Never remove or rename old ones: runs/records.csv
refers to them by name.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from cartpole_jax import IMG, SENSOR_NOISE, STATE_SCALE, TAU

S = STATE_SCALE

# ------------------------------------------------------------------------------
# minimal layers (plain pytrees, no NN library)


def dense_init(key, n_in, n_out, scale=1.0):
    w = jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in) * scale
    return {'w': w, 'b': jnp.zeros(n_out)}


def dense(p, x):
    return x @ p['w'] + p['b']


def mlp_init(key, sizes, last_scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    return [dense_init(k, i, o, last_scale if j == len(sizes) - 2 else 1.0)
            for j, (k, i, o) in enumerate(zip(keys, sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for p in ps[:-1]:
        x = jax.nn.relu(dense(p, x))
    return dense(ps[-1], x)


def conv_init(key, kh, kw, c_in, c_out):
    w = jax.random.normal(key, (kh, kw, c_in, c_out)) * jnp.sqrt(2.0 / (kh * kw * c_in))
    return {'w': w, 'b': jnp.zeros(c_out)}


def conv(p, x, stride=2):
    y = jax.lax.conv_general_dilated(x, p['w'], (stride, stride), 'SAME',
                                     dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + p['b']


def image_encoder_init(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    flat = (IMG // 4) * (IMG // 4) * 32
    return {'c1': conv_init(k1, 3, 3, 2, 16), 'c2': conv_init(k2, 3, 3, 16, 32),
            'fc': dense_init(k3, flat, 64),
            'head': with_logvar_bias(dense_init(k4, 64, 8, scale=0.01), 2 * jnp.log(0.1), start=4)}


def with_logvar_bias(p_head, logv, start=0):
    """Start a variance head at a given log-variance (normalized units)."""
    b = p_head['b'].at[start:start + 4].set(logv)
    return {**p_head, 'b': b}


def image_encoder(p, frames):
    """frames -> latent h (N, 64), state estimate mu (N, 4), log-variance (N, 4, normalized units)."""
    x = jax.nn.relu(conv(p['c1'], frames))
    x = jax.nn.relu(conv(p['c2'], x))
    h = jax.nn.relu(dense(p['fc'], x.reshape(x.shape[0], -1)))
    out = dense(p['head'], h)
    return h, out[:, :4] * S, jnp.clip(out[:, 4:], -10.0, 5.0)


def gru_init(key, n_in, n_hid):
    k1, k2 = jax.random.split(key)
    return {'x': dense_init(k1, n_in, 3 * n_hid, 0.5), 'h': dense_init(k2, n_hid, 3 * n_hid, 0.5)}


def gru(p, h, x):
    xz, xr, xn = jnp.split(dense(p['x'], x), 3, -1)
    hz, hr, hn = jnp.split(dense(p['h'], h), 3, -1)
    z = jax.nn.sigmoid(xz + hz)
    r = jax.nn.sigmoid(xr + hr)
    n = jnp.tanh(xn + r * hn)
    return (1 - z) * n + z * h


def uncertainty_feat(x, var):
    """Policy features for estimator-style methods: normalized estimate + scaled log-variance."""
    return jnp.concatenate([x / S, 0.1 * jnp.log(var / S ** 2)], -1)

# ------------------------------------------------------------------------------


class Method(NamedTuple):
    name: str
    description: str
    feat_dim: int
    init: Callable
    init_carry: Callable
    step: Callable


def _no_carry(n):
    return ()

# --- baselines -----------------------------------------------------------------


def _sensor_only():
    def step(p, carry, z, frames, u_prev):
        return z / S, {'x': z}, carry, {}
    return Method('sensor_only', 'Noisy sensor state only, no image.', 4,
                  lambda key: {}, _no_carry, step)


def _image_only():
    def step(p, carry, z, frames, u_prev):
        h, mu, _ = image_encoder(p['enc'], frames)
        return h, {'x': mu}, carry, {}
    return Method('image_only', 'CNN on 2 stacked frames only, no sensor.', 64,
                  lambda key: {'enc': image_encoder_init(key)}, _no_carry, step)


def _concat():
    def init(key):
        k1, k2 = jax.random.split(key)
        return {'enc': image_encoder_init(k1), 'dec': dense_init(k2, 68, 4, 0.1)}

    def step(p, carry, z, frames, u_prev):
        h, mu, _ = image_encoder(p['enc'], frames)
        feat = jnp.concatenate([z / S, h], -1)
        return feat, {'x': dense(p['dec'], feat) * S, 'mu_img': mu}, carry, {}
    return Method('concat', 'Concatenate sensor state and CNN latent; the policy MLP does the fusion.', 68,
                  init, _no_carry, step)

# --- option 3: gated latent fusion ---------------------------------------------


def _gated():
    def init(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {'enc': image_encoder_init(k1), 'sens': mlp_init(k2, [4, 64]),
                'gate': dense_init(k3, 128, 64, 0.1), 'dec': dense_init(k4, 64, 4, 0.1)}

    def step(p, carry, z, frames, u_prev):
        h_i, mu, _ = image_encoder(p['enc'], frames)
        h_s = jax.nn.relu(mlp(p['sens'], z / S))
        g = jax.nn.sigmoid(dense(p['gate'], jnp.concatenate([h_s, h_i], -1)))
        feat = g * h_s + (1 - g) * h_i
        return feat, {'x': dense(p['dec'], feat) * S, 'mu_img': mu}, carry, {'w_sensor': g.mean(-1)}
    return Method('gated', 'Gated latent fusion: g * enc(sensor) + (1 - g) * enc(image), g learned per feature.', 64,
                  init, _no_carry, step)

# --- option 4: recurrent belief ------------------------------------------------


def _gru():
    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {'enc': image_encoder_init(k1), 'gru': gru_init(k2, 68, 64), 'dec': dense_init(k3, 64, 4, 0.1)}

    def step(p, h, z, frames, u_prev):
        h_i, mu, _ = image_encoder(p['enc'], frames)
        h = gru(p['gru'], h, jnp.concatenate([z / S, h_i], -1))
        return h, {'x': dense(p['dec'], h) * S, 'mu_img': mu}, h, {}
    return Method('gru', 'Recurrent belief: GRU over [sensor, CNN latent].', 64,
                  init, lambda n: jnp.zeros((n, 64)), step)

# --- option 1: product of experts (static precision weighting) -----------------


def _poe():
    def init(key):
        k1, k2 = jax.random.split(key)
        sens = mlp_init(k2, [8, 32, 4], 0.1)
        sens[-1] = with_logvar_bias(sens[-1], 2 * jnp.log(SENSOR_NOISE))
        return {'enc': image_encoder_init(k1), 'sens': sens}

    def step(p, carry, z, frames, u_prev):
        _, mu, logv_i = image_encoder(p['enc'], frames)
        # sensor variance is predicted from the reading and its disagreement with the image
        logv_s = jnp.clip(mlp(p['sens'], jnp.concatenate([z / S, (z - mu) / S], -1)), -10.0, 5.0)
        v_s, v_i = jnp.exp(logv_s) * S ** 2, jnp.exp(logv_i) * S ** 2
        var = 1.0 / (1.0 / v_s + 1.0 / v_i)
        x = var * (z / v_s + mu / v_i)
        return (uncertainty_feat(x, var), {'x': x, 'mu_img': mu, 'var_sensor': v_s, 'var_img': v_i}, carry,
                {'w_sensor': (var / v_s).mean(-1)})
    return Method('poe', 'Product of experts: precision-weighted average of sensor and image estimates, '
                  'both variances predicted by networks (no dynamics).', 8, init, _no_carry, step)

# --- option 2: differentiable Kalman filter (recommended) ----------------------

_A0 = jnp.eye(4) + TAU * (jnp.eye(4, k=1) * jnp.array([1.0, 0.0, 1.0, 0.0])[:, None])
_B0 = jnp.array([0.0, 0.18, 0.0, -0.27])   # ~ TAU * d(x_ddot, theta_ddot)/du at upright


def _kf(learned):
    def init(key):
        k1, k2 = jax.random.split(key)
        p = {'enc': image_encoder_init(k1), 'A': _A0, 'B': _B0}
        if learned:
            p['sens'] = mlp_init(k2, [8, 32, 4], 0.1)
            p['sens'][-1] = with_logvar_bias(p['sens'][-1], 2 * jnp.log(SENSOR_NOISE))
            p['logq'] = jnp.full(4, 2 * jnp.log(0.05))
        return p

    def init_carry(n):
        return jnp.zeros((n, 4)), jnp.broadcast_to(jnp.diag(0.1 * S ** 2), (n, 4, 4))

    def step(p, carry, z, frames, u_prev):
        x, P = carry
        _, mu, logv_i = image_encoder(p['enc'], frames)
        A = p['A']
        # predict with a learned linear model
        x_pred = x @ A.T + u_prev[:, None] * p['B']
        q = (jnp.exp(p['logq']) if learned else jnp.full(4, 0.05 ** 2)) * S ** 2
        P_pred = jnp.einsum('ij,njk,lk->nil', A, P, A) + jnp.diag(q)
        innov = z - x_pred
        if learned:
            # measurement noise of each source is predicted at every step
            logv_s = jnp.clip(mlp(p['sens'], jnp.concatenate([innov / S, (z - mu) / S], -1)), -10.0, 5.0)
            r_s, r_i = jnp.exp(logv_s) * S ** 2, jnp.exp(logv_i) * S ** 2
        else:
            r_s = jnp.broadcast_to(SENSOR_NOISE ** 2 * S ** 2, z.shape)
            r_i = jnp.broadcast_to(0.1 ** 2 * S ** 2, z.shape)
        # update (information form; both measurements observe the full state)
        eye = jnp.eye(4)
        P_inv = jnp.linalg.inv(P_pred + 1e-6 * eye)
        P_new = jnp.linalg.inv(P_inv + (1.0 / r_s + 1.0 / r_i)[:, :, None] * eye)
        P_new = 0.5 * (P_new + jnp.swapaxes(P_new, 1, 2))
        x_new = jnp.einsum('nij,nj->ni', P_new, jnp.einsum('nij,nj->ni', P_inv, x_pred) + z / r_s + mu / r_i)
        var = jnp.diagonal(P_new, axis1=1, axis2=2)
        # diag of the sensor Kalman gain K_s = P_new R_s^-1
        w_sensor = (var / r_s).mean(-1)
        est = {'x': x_new, 'mu_img': mu}
        if learned:
            est.update(var_sensor=r_s, var_img=r_i)
        return (uncertainty_feat(x_new, var), est, (x_new, P_new),
                {'w_sensor': w_sensor})

    if learned:
        desc = ('Differentiable Kalman filter, learned linear dynamics; sensor and image measurement noise '
                'predicted per step by networks, so the gain adapts (recommended method).')
    else:
        desc = 'Kalman filter with fixed hand-set Q/R (constant gain), learned linear dynamics + CNN measurement.'
    return Method('learned_kf' if learned else 'fixed_kf', desc, 8, init, init_carry, step)

# ------------------------------------------------------------------------------

METHODS = {m.name: m for m in [
    _sensor_only(), _image_only(), _concat(), _kf(False), _gated(), _gru(), _poe(), _kf(True),
]}
