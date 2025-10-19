from lerobot.envs.gymnasium import GymnasiumEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def zero_action(space: gym.Space):
    if isinstance(space, spaces.Discrete):
        return 0  # or np.int64(0)
    if isinstance(space, spaces.MultiDiscrete):
        return np.zeros_like(space.nvec, dtype=np.int64)
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.n, dtype=np.int64)
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    # fallback
    return space.sample()

env = GymnasiumEnv("ALE/BattleZone-v5")
# env = GymnasiumEnv("FetchPickAndPlace-v4")
obs, info = env.reset()
print({k: type(v) for k, v in obs.items()})
print({k: v.shape for k, v in obs["images"].items()})
print("state shape:", obs["state"].shape)
print("goal in obs:", "goal" in obs)
print(env.action_space)
print(env.action_space.shape)

done = False
while not done:
    action = zero_action(env.action_space)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print("rollout ok")
env.close()
