"""Latent-space Kalman fusion (no ground-truth state needed).

Each source is encoded to a D-dim latent observation w with a predicted noise r; a
learned latent dynamics model gives the prior; a diagonal Kalman update fuses them
(RKN-style). Same step() interface as fusion_e2e/fusion.py:

    step(params, carry, z, frames, u_prev) -> (feat, est, carry, info)

est carries the latent terms the self-supervised losses in train.py need.
"""
import jax
import jax.numpy as jnp

from cartpole_jax import STATE_SCALE as S
from fusion import dense, dense_init, image_encoder, image_encoder_init, mlp, mlp_init

D = 16      # latent dimension
BOUND = 3.0  # soft bound on latents: keeps the innovation NLL finite if the latent drifts
LOG_R_MIN = -5.0


def _bounded(x):
    return BOUND * jnp.tanh(x / BOUND)


def init(key):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    head_i = dense_init(k2, 64, 2 * D, 0.5)
    head_i['b'] = head_i['b'].at[D:].set(jnp.log(0.1))
    return {
        'enc_i': image_encoder_init(k1),              # CNN trunk (its 4-d state head is unused here)
        'head_i': head_i,                             # -> w_i, own log-noise
        'enc_s': mlp_init(k3, [4, 64, D]),           # sensor -> w_s
        'noise_s': mlp_init(k4, [2 * D, 32, D], 0.1),
        'noise_i': mlp_init(k5, [2 * D, 32, D], 0.1),
        'dyn': mlp_init(k6, [D + 1, 64, D], 0.1),    # residual latent dynamics
        'logq': jnp.full(D, jnp.log(1e-2)),
    }


def init_carry(n):
    return jnp.zeros((n, D)), jnp.ones((n, D))


def predict(p, m, u):
    """Latent dynamics m_{t+1} = m_t + g([m_t, u_t]); works for any leading dims."""
    return _bounded(m + mlp(p['dyn'], jnp.concatenate([m, u[..., None]], -1)))


def step(p, carry, z, frames, u_prev):
    m, var = carry
    m_prior = predict(p, m, u_prev)
    p_prior = var + jnp.exp(p['logq'])

    w_s = _bounded(mlp(p['enc_s'], z / S))
    h, _, _ = image_encoder(p['enc_i'], frames)
    out_i = dense(p['head_i'], h)
    w_i, own_lr_i = _bounded(out_i[:, :D]), out_i[:, D:]

    # noise of each source from its innovation and its disagreement with the other source
    lr_s = jnp.clip(mlp(p['noise_s'], jnp.concatenate([w_s - m_prior, w_s - w_i], -1)), LOG_R_MIN, 5.0)
    lr_i = jnp.clip(own_lr_i + mlp(p['noise_i'], jnp.concatenate([w_i - m_prior, w_i - w_s], -1)), LOG_R_MIN, 5.0)
    r_s, r_i = jnp.exp(lr_s), jnp.exp(lr_i)

    # diagonal Kalman update with two identity measurements in latent space
    var_new = 1.0 / (1.0 / p_prior + 1.0 / r_s + 1.0 / r_i)
    m_new = var_new * (m_prior / p_prior + w_s / r_s + w_i / r_i)

    feat = jnp.concatenate([m_new, 0.1 * jnp.log(var_new)], -1)
    est = {'m': m_new, 'm_prior': m_prior, 'p_prior': p_prior, 'w_s': w_s, 'r_s': r_s, 'w_i': w_i, 'r_i': r_i}
    return feat, est, (m_new, var_new), {'w_sensor': (var_new / r_s).mean(-1)}


FEAT_DIM = 2 * D
