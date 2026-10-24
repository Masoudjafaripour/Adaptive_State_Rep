"""Differentiable CartPole in JAX: dynamics, soft renderer, and sensor/image corruption.

Dynamics follow gymnasium's CartPole-v1 (Euler, tau=0.02) but with a continuous
force u in [-1, 1] * FORCE_MAG, so the whole rollout is differentiable.
"""
import jax
import jax.numpy as jnp

GRAVITY   = 9.8
M_CART    = 1.0
M_POLE    = 0.1
TOTAL_M   = M_CART + M_POLE
HALF_LEN  = 0.5
POLE_ML   = M_POLE * HALF_LEN
FORCE_MAG = 10.0
TAU       = 0.02

THETA_LIMIT = 12 * 2 * jnp.pi / 360   # same failure thresholds as gymnasium
X_LIMIT     = 2.4

# rough per-dimension scale of [x, x_dot, theta, theta_dot]; used to normalize
STATE_SCALE = jnp.array([1.0, 1.0, 0.2, 1.0])

# observation noise (in units of STATE_SCALE)
SENSOR_NOISE     = 0.02   # nominal sensor
SENSOR_BAD_NOISE = 1.0    # sensor during a corruption window
IMG_NOISE        = 0.05   # nominal pixel noise
IMG_BAD_MIX      = 0.85   # image during corruption: 15% signal, 85% uniform noise

IMG = 32
_XS = jnp.linspace(-2.4, 2.4, IMG)
_YS = jnp.linspace(1.2, -0.6, IMG)
PX, PY = jnp.meshgrid(_XS, _YS)   # (IMG, IMG), row 0 is the top of the image

# ------------------------------------------------------------------------------


def step(s, u):
    """s: (N, 4) [x, x_dot, theta, theta_dot], u: (N,) in [-1, 1]."""
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    f = FORCE_MAG * u
    c, sn = jnp.cos(th), jnp.sin(th)
    temp  = (f + POLE_ML * thd ** 2 * sn) / TOTAL_M
    thacc = (GRAVITY * sn - c * temp) / (HALF_LEN * (4.0 / 3.0 - M_POLE * c ** 2 / TOTAL_M))
    xacc  = temp - POLE_ML * thacc * c / TOTAL_M
    return jnp.stack([x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc], -1)


def reward(s, u):
    """Smooth surrogate for 'keep the pole up and the cart centred', scaled by the failure limits
    (cos(theta) alone is ~0.98 at the 12 degree limit, which gave almost no gradient signal)."""
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    return (1.0 - (th / THETA_LIMIT) ** 2 - 0.5 * (x / X_LIMIT) ** 2
            - 0.01 * (xd ** 2 + thd ** 2) - 0.01 * u ** 2)


def failed(s):
    return (jnp.abs(s[..., 2]) > THETA_LIMIT) | (jnp.abs(s[..., 0]) > X_LIMIT)


def reset(key, n):
    return jax.random.uniform(key, (n, 4), minval=-0.05, maxval=0.05)


def render(s):
    """Soft (differentiable) grayscale render: (N, 4) -> (N, IMG, IMG) in [0, 1]."""
    x  = s[:, 0, None, None]
    th = s[:, 2, None, None]
    soft = 0.04
    cart = jax.nn.sigmoid((0.25 - jnp.abs(PX - x)) / soft) * jax.nn.sigmoid((0.12 - jnp.abs(PY)) / soft)
    # pole: distance from each pixel to the segment (x, 0) -> tip
    dx, dy = 2 * HALF_LEN * jnp.sin(th), 2 * HALF_LEN * jnp.cos(th)
    t = jnp.clip(((PX - x) * dx + PY * dy) / (dx ** 2 + dy ** 2), 0.0, 1.0)
    d = jnp.sqrt((PX - x - t * dx) ** 2 + (PY - t * dy) ** 2 + 1e-8)
    pole = jax.nn.sigmoid((0.08 - d) / soft)
    return 1.0 - (1.0 - cart) * (1.0 - pole)


def observe(key, s, sensor_bad, image_bad):
    """Noisy sensor reading and noisy frame. sensor_bad / image_bad: (N,) bool."""
    k1, k2, k3 = jax.random.split(key, 3)
    sig = jnp.where(sensor_bad[:, None], SENSOR_BAD_NOISE, SENSOR_NOISE) * STATE_SCALE
    z = s + sig * jax.random.normal(k1, s.shape)
    img = render(s)
    clean = img + IMG_NOISE * jax.random.normal(k2, img.shape)
    bad = (1 - IMG_BAD_MIX) * img + IMG_BAD_MIX * jax.random.uniform(k3, img.shape)
    return z, jnp.where(image_bad[:, None, None], bad, clean)

# ------------------------------------------------------------------------------
# corruption schedules: (T, N) bool masks for sensor / image corruption


def sample_schedule(key, n, T):
    """Training: each env is clean, sensor-corrupted or image-corrupted for one random window."""
    k1, k2, k3 = jax.random.split(key, 3)
    mode   = jax.random.randint(k1, (n,), 0, 3)
    start  = jax.random.randint(k2, (n,), 0, T // 2)
    length = jax.random.randint(k3, (n,), T // 4, T // 2)
    t = jnp.arange(T)[:, None]
    win = (t >= start) & (t < start + length)
    return win & (mode == 1), win & (mode == 2)


def fixed_schedule(mode, n, T, start, end):
    """Evaluation: mode in {'clean', 'sensor_corrupt', 'image_corrupt'}, window [start, end)."""
    t = jnp.arange(T)[:, None]
    win = jnp.broadcast_to((t >= start) & (t < end), (T, n))
    off = jnp.zeros((T, n), bool)
    return (win if mode == 'sensor_corrupt' else off), (win if mode == 'image_corrupt' else off)
