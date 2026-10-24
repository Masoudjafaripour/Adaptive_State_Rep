"""Dreamer-style world model (Gaussian RSSM) with pluggable multimodal posterior fusion.

The contribution is the `kf` posterior: with a Gaussian RSSM, fusing the prior with one
Gaussian latent observation per modality is exactly a (diagonal) Kalman update, so the
posterior is

    var_t = (1/var^-_t + sum_j 1/r^j_t)^-1,   mu_t = var_t (mu^-_t/var^-_t + sum_j w^j_t / r^j_t)

with per-step modality noise r^j_t predicted from the modality's embedding, its innovation
w^j_t - mu^-_t and its disagreement with the other modality. The gain k^j_t = var_t / r^j_t
is how much the model trusts modality j at time t.

All other posteriors are baselines / ablations (see FUSIONS). Plain pytrees, no NN library.
"""
import jax
import jax.numpy as jnp

DETER, STOCH, HID, EMBED = 200, 30, 200, 256
DEPTH = 16          # CNN width (Dreamer uses 32; halved for CPU)
MIN_STD = 0.1
LOG_R_MIN, LOG_R_MAX = -5.0, 5.0

# ------------------------------------------------------------------------------
# layers


def dense_init(key, n_in, n_out, scale=1.0):
    return {'w': jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(1.0 / n_in) * scale, 'b': jnp.zeros(n_out)}


def dense(p, x):
    return x @ p['w'] + p['b']


def mlp_init(key, sizes, last_scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    return [dense_init(k, i, o, last_scale if j == len(sizes) - 2 else 1.0)
            for j, (k, i, o) in enumerate(zip(keys, sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for p in ps[:-1]:
        x = jax.nn.elu(dense(p, x))
    return dense(ps[-1], x)


def conv_init(key, c_in, c_out, k=4):
    return {'w': jax.random.normal(key, (k, k, c_in, c_out)) * jnp.sqrt(1.0 / (k * k * c_in)), 'b': jnp.zeros(c_out)}


def conv(p, x):
    return jax.lax.conv_general_dilated(x, p['w'], (2, 2), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + p['b']


def deconv(p, x):
    return jax.lax.conv_transpose(x, p['w'], (2, 2), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + p['b']


def gru_init(key, n_in, n_hid):
    k1, k2 = jax.random.split(key)
    return {'x': dense_init(k1, n_in, 3 * n_hid), 'h': dense_init(k2, n_hid, 3 * n_hid)}


def gru(p, h, x):
    xz, xr, xn = jnp.split(dense(p['x'], x), 3, -1)
    hz, hr, hn = jnp.split(dense(p['h'], h), 3, -1)
    z, r = jax.nn.sigmoid(xz + hz), jax.nn.sigmoid(xr + hr)
    return (1 - z) * jnp.tanh(xn + r * hn) + z * h


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _std(x):
    return jax.nn.softplus(x) + MIN_STD

# ------------------------------------------------------------------------------
# encoders / decoders


def img_enc_init(key):
    ks = jax.random.split(key, 5)
    chans = [3, DEPTH, 2 * DEPTH, 4 * DEPTH, 8 * DEPTH]
    return {'convs': [conv_init(k, a, b) for k, a, b in zip(ks[:4], chans[:-1], chans[1:])],
            'out': dense_init(ks[4], 4 * 4 * 8 * DEPTH, EMBED)}


def img_enc(p, img):
    """img: (..., 64, 64, 3) uint8 -> (..., EMBED)."""
    lead = img.shape[:-3]
    x = img.reshape((-1,) + img.shape[-3:]).astype(jnp.float32) / 255.0 - 0.5
    for c in p['convs']:
        x = jax.nn.elu(conv(c, x))
    return jax.nn.elu(dense(p['out'], x.reshape(x.shape[0], -1))).reshape(lead + (EMBED,))


def img_dec_init(key):
    ks = jax.random.split(key, 5)
    chans = [8 * DEPTH, 4 * DEPTH, 2 * DEPTH, DEPTH, 3]
    return {'inp': dense_init(ks[0], DETER + STOCH, 4 * 4 * 8 * DEPTH),
            'deconvs': [conv_init(k, a, b) for k, a, b in zip(ks[1:], chans[:-1], chans[1:])]}


def img_dec(p, feat):
    """feat: (..., DETER+STOCH) -> image mean in [-0.5, 0.5] space, (..., 64, 64, 3)."""
    lead = feat.shape[:-1]
    x = dense(p['inp'], feat.reshape(-1, feat.shape[-1])).reshape(-1, 4, 4, 8 * DEPTH)
    for i, d in enumerate(p['deconvs']):
        x = deconv(d, x)
        if i < len(p['deconvs']) - 1:
            x = jax.nn.elu(x)
    return x.reshape(lead + (64, 64, 3))

# ------------------------------------------------------------------------------
# posterior fusion variants
#   init(key, prop_dim) -> params
#   apply(p, h, mu_p, std_p, e_img, e_prop) -> (mu, std, info)
#   info: 'w_sensor' (trust in proprio, for plots), 'nll' (innovation NLL, 0 if unused)


def _gauss_product(mu_p, std_p, ws, rs):
    """Prior N(mu_p, std_p^2) times one N(w, r) per modality (diagonal Kalman update)."""
    prec = 1.0 / std_p ** 2 + sum(1.0 / r for r in rs)
    var = 1.0 / prec
    mu = var * (mu_p / std_p ** 2 + sum(w / r for w, r in zip(ws, rs)))
    gains = [var / r for r in rs]
    return mu, jnp.sqrt(var), gains


def _innovation_nll(ws, rs, mu_p, std_p):
    mu_p = jax.lax.stop_gradient(mu_p)
    return sum(jnp.mean(0.5 * (jnp.log(r + std_p ** 2) + (w - mu_p) ** 2 / (r + std_p ** 2)), -1)
               for w, r in zip(ws, rs))


def _mlp_post(inputs):
    """Standard Dreamer posterior q(z | h, embeddings) over the selected inputs."""
    sizes = {'h': DETER, 'img': EMBED, 'prop': EMBED}

    def init(key, prop_dim):
        return mlp_init(key, [sum(sizes[i] for i in inputs), HID, 2 * STOCH])

    def apply(p, h, mu_p, std_p, e_img, e_prop):
        x = jnp.concatenate([{'h': h, 'img': e_img, 'prop': e_prop}[i] for i in inputs], -1)
        mu, s = jnp.split(mlp(p, x), 2, -1)
        return mu, _std(s), {'nll': jnp.zeros(h.shape[:-1])}
    return init, apply


def _kf_post(adaptive):
    """Kalman/product-of-experts posterior. adaptive=False: per-modality noise is a learned constant."""
    def init(key, prop_dim):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p = {'w_img': mlp_init(k1, [EMBED, HID, STOCH]), 'w_prop': mlp_init(k2, [EMBED, HID, STOCH])}
        if adaptive:
            p['r_img'] = mlp_init(k3, [EMBED + 2 * STOCH, HID, STOCH], 0.1)
            p['r_prop'] = mlp_init(k4, [EMBED + 2 * STOCH, HID, STOCH], 0.1)
        else:
            p['logr_img'], p['logr_prop'] = jnp.zeros(STOCH), jnp.zeros(STOCH)
        return p

    def apply(p, h, mu_p, std_p, e_img, e_prop):
        w_i, w_s = mlp(p['w_img'], e_img), mlp(p['w_prop'], e_prop)
        if adaptive:
            lr_i = mlp(p['r_img'], jnp.concatenate([e_img, w_i - mu_p, w_i - w_s], -1))
            lr_s = mlp(p['r_prop'], jnp.concatenate([e_prop, w_s - mu_p, w_s - w_i], -1))
        else:
            lr_i = jnp.broadcast_to(p['logr_img'], w_i.shape)
            lr_s = jnp.broadcast_to(p['logr_prop'], w_s.shape)
        r_i = jnp.exp(jnp.clip(lr_i, LOG_R_MIN, LOG_R_MAX))
        r_s = jnp.exp(jnp.clip(lr_s, LOG_R_MIN, LOG_R_MAX))
        mu, std, (k_i, k_s) = _gauss_product(mu_p, std_p, [w_i, w_s], [r_i, r_s])
        return mu, std, {'w_sensor': k_s.mean(-1), 'w_image': k_i.mean(-1),
                         'nll': _innovation_nll([w_i, w_s], [r_i, r_s], mu_p, std_p)}
    return init, apply


# name -> (init, apply, description, innovation-NLL weight)
FUSIONS = {
    'concat':       (*_mlp_post(['h', 'img', 'prop']), 'Standard multimodal Dreamer: q(z | h, enc(img), enc(proprio)).', 0.0),
    'image_only':   (*_mlp_post(['h', 'img']), 'Dreamer from pixels only.', 0.0),
    'proprio_only': (*_mlp_post(['h', 'prop']), 'Dreamer from proprioception only.', 0.0),
    'poe_fixed':    (*_kf_post(False), 'Kalman/product-of-experts posterior with a learned *constant* noise per modality '
                                       '(fixed gain).', 0.0),
    'kf_nonll':     (*_kf_post(True), 'Ablation: adaptive Kalman posterior without the innovation NLL.', 0.0),
    'kf':           (*_kf_post(True), '**Ours.** Adaptive Kalman posterior: per-step modality noise from embedding, '
                                      'innovation and cross-modal disagreement; + innovation NLL.', 1.0),
}

# ------------------------------------------------------------------------------
# world model


def wm_init(key, fusion, prop_dim, action_dim):
    ks = jax.random.split(key, 9)
    return {
        'enc_img': img_enc_init(ks[0]),
        'enc_prop': mlp_init(ks[1], [prop_dim, HID, EMBED]),
        'rssm_in': dense_init(ks[2], STOCH + action_dim, HID),
        'gru': gru_init(ks[3], HID, DETER),
        'prior': mlp_init(ks[4], [DETER, HID, 2 * STOCH]),
        'post': FUSIONS[fusion][0](ks[5], prop_dim),
        'dec_img': img_dec_init(ks[6]),
        'dec_prop': mlp_init(ks[7], [DETER + STOCH, HID, HID, prop_dim]),
        'reward': mlp_init(ks[8], [DETER + STOCH, HID, HID, 1]),
    }


def embed(p, obs):
    return img_enc(p['enc_img'], obs['image']), jax.nn.elu(mlp(p['enc_prop'], symlog(obs['proprio'])))


def img_step(p, h, z, a):
    """Prior transition: (h, z, a) -> h', prior mean/std."""
    h = gru(p['gru'], h, jax.nn.elu(dense(p['rssm_in'], jnp.concatenate([z, a], -1))))
    mu_p, s = jnp.split(mlp(p['prior'], h), 2, -1)
    return h, mu_p, _std(s)


def obs_step(p, fusion, h, z, a, e_img, e_prop, key):
    h, mu_p, std_p = img_step(p, h, z, a)
    mu, std, info = FUSIONS[fusion][1](p['post'], h, mu_p, std_p, e_img, e_prop)
    z = mu + std * jax.random.normal(key, mu.shape)
    return h, z, (mu_p, std_p, mu, std), info


def observe(p, fusion, obs, actions, key):
    """obs: dict of (T, B, ...), actions: (T, B, A) with actions[t] = action taken *before* obs[t]."""
    e_img, e_prop = embed(p, obs)
    B = actions.shape[1]
    h0, z0 = jnp.zeros((B, DETER)), jnp.zeros((B, STOCH))

    def body(c, xs):
        h, z = c
        a, ei, ep, k = xs
        h, z, dists, info = obs_step(p, fusion, h, z, a, ei, ep, k)
        return (h, z), (h, z, dists, info)
    keys = jax.random.split(key, actions.shape[0])
    _, (hs, zs, dists, info) = jax.lax.scan(body, (h0, z0), (actions, e_img, e_prop, keys))
    return hs, zs, dists, info


def kl_gauss(mu_q, std_q, mu_p, std_p):
    return jnp.sum(jnp.log(std_p / std_q) + (std_q ** 2 + (mu_q - mu_p) ** 2) / (2 * std_p ** 2) - 0.5, -1)

# ------------------------------------------------------------------------------
# actor / critic


def actor_init(key, action_dim):
    return mlp_init(key, [DETER + STOCH, HID, HID, 2 * action_dim], 0.1)


def actor(p, feat, key=None):
    """Tanh-Gaussian policy. key=None -> deterministic mean action. Returns (action, entropy proxy)."""
    mu, s = jnp.split(mlp(p, feat), 2, -1)
    std = jax.nn.softplus(s) + 1e-2
    if key is None:
        return jnp.tanh(mu), jnp.zeros(mu.shape[:-1])
    a = jnp.tanh(mu + std * jax.random.normal(key, mu.shape))
    return a, jnp.sum(jnp.log(std), -1)


def critic_init(key):
    return mlp_init(key, [DETER + STOCH, HID, HID, 1], 0.1)


def critic(p, feat):
    return mlp(p, feat)[..., 0]
