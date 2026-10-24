"""DeepMind Control tasks with pixels + proprioception, and sensor/image corruption windows.

Backends (same tasks, same observation format):
    dmc         dm_control (C MuJoCo + OpenGL renderer). Standard pixel benchmark; runs on Mac.
    playground  MuJoCo Playground (MJX, JAX physics); frames rendered on the host from the MJX state.
                Physics is jit-compiled; for batched GPU rendering use Playground's Madrona renderer (CUDA).

obs = {'image': (64, 64, 3) uint8, 'proprio': (P,) float32}; action repeat 2, 1000 env steps per episode.
"""
import numpy as np

TASKS = {   # name -> (dm_control domain, task, Playground env name)
    'cartpole_swingup': ('cartpole', 'swingup', 'CartpoleSwingup'),
    'cheetah_run':      ('cheetah', 'run', 'CheetahRun'),
    'walker_walk':      ('walker', 'walk', 'WalkerWalk'),
}
MODES = ['clean', 'sensor_corrupt', 'image_corrupt']

SENSOR_NOISE     = 0.01   # nominal proprio noise (raw units)
SENSOR_BAD_NOISE = 1.0    # proprio noise during a sensor-corruption window
IMG_BAD_MIX      = 0.85   # image during corruption: 15% signal, 85% uniform noise


class DMCEnv:
    def __init__(self, task, seed=0, size=64, action_repeat=2):
        from dm_control import suite
        domain, name, _ = TASKS[task]
        self._env = suite.load(domain, name, task_kwargs={'random': seed})
        self.size, self.action_repeat = size, action_repeat
        spec = self._env.action_spec()
        self.action_dim = spec.shape[0]
        self.max_steps = 1000 // action_repeat
        self.camera = 2 if domain == 'quadruped' else 0

    def _obs(self, ts):
        prop = np.concatenate([np.asarray(v, np.float32).ravel() for v in ts.observation.values()])
        img = self._env.physics.render(self.size, self.size, camera_id=self.camera)
        return {'image': img, 'proprio': prop}

    def reset(self):
        return self._obs(self._env.reset())

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            ts = self._env.step(action)
            reward += ts.reward or 0.0
            if ts.last():
                break
        return self._obs(ts), reward, ts.last()


class PlaygroundEnv:
    """MJX physics (jit), host rendering of the same MuJoCo model."""

    def __init__(self, task, seed=0, size=64, action_repeat=2):
        import jax
        import mujoco
        from mujoco_playground import registry
        self._env = registry.load(TASKS[task][2])
        self._reset, self._step = jax.jit(self._env.reset), jax.jit(self._env.step)
        self._key = jax.random.PRNGKey(seed)
        self._jax = jax
        self._model = self._env.mj_model
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, size, size)
        self._mujoco = mujoco
        self.action_dim, self.size, self.action_repeat = self._env.action_size, size, action_repeat
        self.max_steps = 1000 // action_repeat

    def _obs(self):
        d = self._state.data
        self._data.qpos[:], self._data.qvel[:] = np.asarray(d.qpos), np.asarray(d.qvel)
        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=0)
        return {'image': self._renderer.render(), 'proprio': np.asarray(self._state.obs, np.float32).ravel()}

    def reset(self):
        self._key, k = self._jax.random.split(self._key)
        self._state, self._t = self._reset(k), 0
        return self._obs()

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            self._state = self._step(self._state, np.asarray(action, np.float32))
            reward += float(self._state.reward)
        self._t += 1
        return self._obs(), reward, self._t >= self.max_steps


def make_env(task, backend='dmc', seed=0):
    return {'dmc': DMCEnv, 'playground': PlaygroundEnv}[backend](task, seed)


class Corruption:
    """Per-episode corruption schedule over agent steps: one window of sensor or image corruption."""

    def __init__(self, mode, start, end, rng):
        self.mode, self.start, self.end, self.rng = mode, start, end, rng

    @classmethod
    def sample(cls, rng, T):
        mode = MODES[rng.integers(3)]
        start = int(rng.integers(0, T // 2))
        return cls(mode, start, start + int(rng.integers(T // 4, T // 2)), rng)

    @classmethod
    def fixed(cls, mode, T, rng):
        return cls(mode, int(0.3 * T), int(0.7 * T), rng)

    def active(self, t):
        return self.start <= t < self.end

    def __call__(self, obs, t):
        """-> corrupted copy of obs, plus (sensor_bad, image_bad) flags."""
        on = self.active(t)
        s_bad, i_bad = on and self.mode == 'sensor_corrupt', on and self.mode == 'image_corrupt'
        sig = SENSOR_BAD_NOISE if s_bad else SENSOR_NOISE
        prop = obs['proprio'] + sig * self.rng.standard_normal(obs['proprio'].shape).astype(np.float32)
        img = obs['image']
        if i_bad:
            noise = self.rng.integers(0, 256, img.shape)
            img = ((1 - IMG_BAD_MIX) * img + IMG_BAD_MIX * noise).astype(np.uint8)
        return {'image': img, 'proprio': prop}, s_bad, i_bad
