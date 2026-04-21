"""
POPGym environment wrapper for TD-MPC2.

Bridges POPGym's Gymnasium POMDP environments to the interface
expected by TD-MPC2's training loop.

Recommended invocation (POPGym rewards live in ~[-1,1] after symlog,
so tighten the value bins and reduce total steps):

    python train.py task=popgym-PositionOnlyPendulumEasy-v0 \
        episodic=true vmin=-2 vmax=2 steps=500000

All cfg fields set here can be overridden on the CLI.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.wrappers.timeout import Timeout


# ──────────────────────────────────────────────────────────────────
# Task registry
#
# Maps TD-MPC2 task string → dict with:
#   gym_id:     Gymnasium registered id (from `import popgym`)
#   max_steps:  fallback max episode steps (used only when the env
#               does not expose max_episode_length)
#   action:     'continuous' | 'discrete'  (for logging / documentation)
#
# Episode length is auto-detected from the env when possible;
# the registry value is only a fallback.
# ──────────────────────────────────────────────────────────────────
POPGYM_TASKS = {
    # ── Continuous-action (Pendulum-based) ─────────────────────────
    "popgym-PositionOnlyPendulumEasy-v0": dict(
        gym_id="popgym-PositionOnlyPendulumEasy-v0",
        max_steps=200, action="continuous",
    ),
    "popgym-PositionOnlyPendulumMedium-v0": dict(
        gym_id="popgym-PositionOnlyPendulumMedium-v0",
        max_steps=150, action="continuous",
    ),
    "popgym-PositionOnlyPendulumHard-v0": dict(
        gym_id="popgym-PositionOnlyPendulumHard-v0",
        max_steps=100, action="continuous",
    ),
    "popgym-NoisyPositionOnlyPendulumEasy-v0": dict(
        gym_id="popgym-NoisyPositionOnlyPendulumEasy-v0",
        max_steps=200, action="continuous",
    ),
    "popgym-NoisyPositionOnlyPendulumMedium-v0": dict(
        gym_id="popgym-NoisyPositionOnlyPendulumMedium-v0",
        max_steps=150, action="continuous",
    ),
    "popgym-NoisyPositionOnlyPendulumHard-v0": dict(
        gym_id="popgym-NoisyPositionOnlyPendulumHard-v0",
        max_steps=100, action="continuous",
    ),
    # ── Discrete-action: CartPole variants ─────────────────────────
    "popgym-PositionOnlyCartPoleEasy-v0": dict(
        gym_id="popgym-PositionOnlyCartPoleEasy-v0",
        max_steps=200, action="discrete",
    ),
    "popgym-PositionOnlyCartPoleMedium-v0": dict(
        gym_id="popgym-PositionOnlyCartPoleMedium-v0",
        max_steps=400, action="discrete",
    ),
    "popgym-PositionOnlyCartPoleHard-v0": dict(
        gym_id="popgym-PositionOnlyCartPoleHard-v0",
        max_steps=600, action="discrete",
    ),
    "popgym-VelocityOnlyCartpoleEasy-v0": dict(
        gym_id="popgym-VelocityOnlyCartpoleEasy-v0",
        max_steps=200, action="discrete",
    ),
    "popgym-VelocityOnlyCartpoleMedium-v0": dict(
        gym_id="popgym-VelocityOnlyCartpoleMedium-v0",
        max_steps=400, action="discrete",
    ),
    "popgym-VelocityOnlyCartpoleHard-v0": dict(
        gym_id="popgym-VelocityOnlyCartpoleHard-v0",
        max_steps=600, action="discrete",
    ),
    "popgym-NoisyPositionOnlyCartPoleEasy-v0": dict(
        gym_id="popgym-NoisyPositionOnlyCartPoleEasy-v0",
        max_steps=200, action="discrete",
    ),
    "popgym-NoisyPositionOnlyCartPoleMedium-v0": dict(
        gym_id="popgym-NoisyPositionOnlyCartPoleMedium-v0",
        max_steps=400, action="discrete",
    ),
    "popgym-NoisyPositionOnlyCartPoleHard-v0": dict(
        gym_id="popgym-NoisyPositionOnlyCartPoleHard-v0",
        max_steps=600, action="discrete",
    ),
    # ── Discrete-action: Game envs ─────────────────────────────────
    "popgym-HigherLowerEasy-v0": dict(
        gym_id="popgym-HigherLowerEasy-v0",
        max_steps=51, action="discrete",
    ),
    "popgym-HigherLowerMedium-v0": dict(
        gym_id="popgym-HigherLowerMedium-v0",
        max_steps=103, action="discrete",
    ),
    "popgym-HigherLowerHard-v0": dict(
        gym_id="popgym-HigherLowerHard-v0",
        max_steps=155, action="discrete",
    ),
    "popgym-BattleshipEasy-v0": dict(
        gym_id="popgym-BattleshipEasy-v0",
        max_steps=36, action="discrete",
    ),
    "popgym-BattleshipMedium-v0": dict(
        gym_id="popgym-BattleshipMedium-v0",
        max_steps=64, action="discrete",
    ),
    "popgym-BattleshipHard-v0": dict(
        gym_id="popgym-BattleshipHard-v0",
        max_steps=100, action="discrete",
    ),
    "popgym-ConcentrationEasy-v0": dict(
        gym_id="popgym-ConcentrationEasy-v0",
        max_steps=100, action="discrete",
    ),
    "popgym-ConcentrationMedium-v0": dict(
        gym_id="popgym-ConcentrationMedium-v0",
        max_steps=200, action="discrete",
    ),
    "popgym-ConcentrationHard-v0": dict(
        gym_id="popgym-ConcentrationHard-v0",
        max_steps=400, action="discrete",
    ),
    "popgym-MineSweeperEasy-v0": dict(
        gym_id="popgym-MineSweeperEasy-v0",
        max_steps=50, action="discrete",
    ),
    "popgym-MineSweeperMedium-v0": dict(
        gym_id="popgym-MineSweeperMedium-v0",
        max_steps=100, action="discrete",
    ),
    "popgym-MineSweeperHard-v0": dict(
        gym_id="popgym-MineSweeperHard-v0",
        max_steps=200, action="discrete",
    ),
    # ── Discrete-action: Diagnostic envs ───────────────────────────
    "popgym-RepeatFirstEasy-v0": dict(
        gym_id="popgym-RepeatFirstEasy-v0",
        max_steps=51, action="discrete",
    ),
    "popgym-RepeatFirstMedium-v0": dict(
        gym_id="popgym-RepeatFirstMedium-v0",
        max_steps=415, action="discrete",
    ),
    "popgym-RepeatFirstHard-v0": dict(
        gym_id="popgym-RepeatFirstHard-v0",
        max_steps=831, action="discrete",
    ),
    "popgym-RepeatPreviousEasy-v0": dict(
        gym_id="popgym-RepeatPreviousEasy-v0",
        max_steps=51, action="discrete",
    ),
    "popgym-RepeatPreviousMedium-v0": dict(
        gym_id="popgym-RepeatPreviousMedium-v0",
        max_steps=415, action="discrete",
    ),
    "popgym-RepeatPreviousHard-v0": dict(
        gym_id="popgym-RepeatPreviousHard-v0",
        max_steps=831, action="discrete",
    ),
    "popgym-AutoencodeEasy-v0": dict(
        gym_id="popgym-AutoencodeEasy-v0",
        max_steps=104, action="discrete",
    ),
    "popgym-AutoencodeMedium-v0": dict(
        gym_id="popgym-AutoencodeMedium-v0",
        max_steps=832, action="discrete",
    ),
    "popgym-AutoencodeHard-v0": dict(
        gym_id="popgym-AutoencodeHard-v0",
        max_steps=1664, action="discrete",
    ),
    "popgym-CountRecallEasy-v0": dict(
        gym_id="popgym-CountRecallEasy-v0",
        max_steps=51, action="discrete",
    ),
    "popgym-CountRecallMedium-v0": dict(
        gym_id="popgym-CountRecallMedium-v0",
        max_steps=415, action="discrete",
    ),
    "popgym-CountRecallHard-v0": dict(
        gym_id="popgym-CountRecallHard-v0",
        max_steps=831, action="discrete",
    ),
    "popgym-MultiarmedBanditEasy-v0": dict(
        gym_id="popgym-MultiarmedBanditEasy-v0",
        max_steps=100, action="discrete",
    ),
    "popgym-MultiarmedBanditMedium-v0": dict(
        gym_id="popgym-MultiarmedBanditMedium-v0",
        max_steps=200, action="discrete",
    ),
    "popgym-MultiarmedBanditHard-v0": dict(
        gym_id="popgym-MultiarmedBanditHard-v0",
        max_steps=400, action="discrete",
    ),
}


# ──────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────

class DiscreteToBoxWrapper(gym.Wrapper):
    """Convert a Discrete/MultiDiscrete action space to a continuous Box.

    The continuous action is interpreted as logits; argmax selects the
    discrete action.  This is the simplest relaxation that lets
    MPPI / Gaussian planning work over discrete environments.

    Discrete(n)         →  Box(-1, 1, shape=(n,))   →  argmax  →  int
    MultiDiscrete(nvec) →  Box(-1, 1, shape=(Σn,))  →  per-group argmax
    """

    def __init__(self, env):
        super().__init__(env)
        self._orig_action_space = env.action_space

        if isinstance(env.action_space, spaces.Discrete):
            n = env.action_space.n
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n,), dtype=np.float32
            )
            self._decode = self._decode_discrete

        elif isinstance(env.action_space, spaces.MultiDiscrete):
            total = int(np.sum(env.action_space.nvec))
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(total,), dtype=np.float32
            )
            self._nvec = env.action_space.nvec
            self._decode = self._decode_multidiscrete

        else:
            # Already continuous — passthrough
            self._decode = lambda a: a

    def _decode_discrete(self, action):
        return int(np.argmax(action))

    def _decode_multidiscrete(self, action):
        idx = 0
        acts = []
        for n in self._nvec:
            acts.append(int(np.argmax(action[idx : idx + n])))
            idx += n
        return np.array(acts)

    def step(self, action):
        return self.env.step(self._decode(action))


class DiscreteObsToBoxWrapper(gym.Wrapper):
    """Convert a Discrete observation space to a one-hot float32 Box.

    Many POPGym game/diagnostic envs return a single int as the
    observation.  TD-MPC2's encoder expects a float tensor.
    """

    def __init__(self, env):
        super().__init__(env)
        self._needs_convert = False

        if isinstance(env.observation_space, spaces.Discrete):
            self._needs_convert = True
            self._n = env.observation_space.n
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._n,), dtype=np.float32
            )

    def _convert_obs(self, obs):
        if not self._needs_convert:
            return obs
        one_hot = np.zeros(self._n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info


class POPGymWrapper(gym.Wrapper):
    """Adapts a POPGym env (Gymnasium 5-tuple) to TD-MPC2's 4-tuple
    interface and injects the required 'success' / 'terminated' info keys.

    Reads cfg.seed for the first reset, then clears it so subsequent
    resets use the env's own RNG stream.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self._initial_seed = getattr(cfg, "seed", None)

    def reset(self):
        kwargs = {}
        if self._initial_seed is not None:
            kwargs["seed"] = self._initial_seed
            self._initial_seed = None      # only seed the very first reset
        obs, _info = self.env.reset(**kwargs)
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        action_np = action.copy() if isinstance(action, np.ndarray) else action
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        done = terminated or truncated
        info["success"] = float(terminated and reward > 0)
        info["terminated"] = terminated
        return obs, float(reward), done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _detect_max_episode_steps(env, fallback):
    """Try to read the episode length from the unwrapped POPGym env.

    POPGym envs inconsistently name this attribute:
      - most use  .max_episode_length
      - Concentration uses  .episode_length
      - HigherLower computes it implicitly (deck_size - 1)

    Falls back to the registry value if nothing is found.
    """
    inner = env.unwrapped
    for attr in ("max_episode_length", "episode_length"):
        val = getattr(inner, attr, None)
        if val is not None:
            return int(val)
    # HigherLower: infer from deck_size
    if hasattr(inner, "deck_size"):
        return int(inner.deck_size - 1)
    return fallback


# ──────────────────────────────────────────────────────────────────
# Factory  (called by envs/__init__.py)
# ──────────────────────────────────────────────────────────────────

def make_env(cfg):
    """
    Make a POPGym environment for TD-MPC2.

    Reads from cfg
    ~~~~~~~~~~~~~~~
    cfg.task   – must match a key in POPGYM_TASKS
    cfg.obs    – must be 'state' (POPGym has no pixel mode)
    cfg.seed   – passed to the first env.reset() for reproducibility

    Writes to cfg (only when user has not overridden via CLI)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg.episodic      = True   POPGym envs terminate.
    cfg.discount_max  = 0.99   Episodic default (matches mujoco.py).
    cfg.rho           = 0.7    Episodic default (matches mujoco.py).

    Cannot be set here (consumed by parse_cfg before make_env runs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg.vmin / cfg.vmax    →  override on CLI:  vmin=-2 vmax=2
    cfg.steps              →  override on CLI:  steps=500000
    cfg.num_bins           →  leave at default (101) unless experimenting

    These are set automatically by envs/__init__.py after this returns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg.obs_shape, cfg.action_dim, cfg.episode_length, cfg.seed_steps
    """
    if cfg.task not in POPGYM_TASKS:
        raise ValueError(f"Unknown POPGym task: {cfg.task}")
    assert cfg.obs == "state", "POPGym tasks only support state observations."

    task_cfg = POPGYM_TASKS[cfg.task]

    # ── Create env ──────────────────────────────────────────────
    import popgym  # noqa: F401  (triggers gymnasium.register)

    env = gym.make(task_cfg["gym_id"])

    # Auto-detect episode length; fall back to registry value
    max_steps = _detect_max_episode_steps(env, fallback=task_cfg["max_steps"])

    # ── Space adapters ──────────────────────────────────────────
    if isinstance(env.observation_space, spaces.Discrete):
        env = DiscreteObsToBoxWrapper(env)

    if isinstance(env.observation_space, (spaces.Tuple, spaces.Dict)):
        env = gym.wrappers.FlattenObservation(env)

    if isinstance(env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
        env = DiscreteToBoxWrapper(env)

    # ── TD-MPC2 interface adapter ───────────────────────────────
    env = POPGymWrapper(env, cfg)
    env = Timeout(env, max_episode_steps=max_steps)

    # ── Episodic cfg defaults ───────────────────────────────────
    # Only overwrite when the user left config.yaml defaults.
    # Hydra CLI overrides are already baked into cfg by this point,
    # so an explicit `episodic=false` on the CLI will survive.
    if not cfg.episodic:                  # default: false
        cfg.episodic = True
    if cfg.discount_max > 0.99:           # default: 0.995
        cfg.discount_max = 0.99
    if cfg.rho < 0.6:                     # default: 0.5
        cfg.rho = 0.7

    return env