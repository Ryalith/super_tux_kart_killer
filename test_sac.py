import torch
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import SAC
from datetime import datetime

env = gym.make(
    "supertuxkart/flattened_continuous_actions-v0",
    render_mode="human",
    agent=AgentSpec(use_ai=False),
)


class ContinuousSubsetWrapper(gym.ObservationWrapper):
    """
    Keep only a subset of entries from obs['continuous'] while leaving
    the rest of the Dict observation unchanged.
    """

    def __init__(self, env: gym.Env, keep_indices: list[int]):
        super().__init__(env)
        self.keep_indices = list(keep_indices)

        assert isinstance(self.observation_space, gym.spaces.Dict), (
            "Expected Dict observation space."
        )
        assert "continuous" in self.observation_space.spaces, (
            "Expected 'continuous' key in observation."
        )

        old_box = self.observation_space["continuous"]
        low = old_box.low[self.keep_indices]
        high = old_box.high[self.keep_indices]
        new_box = gym.spaces.Box(low=low, high=high, dtype=old_box.dtype)

        spaces = dict(self.observation_space.spaces)
        spaces["continuous"] = new_box
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        obs = dict(obs)
        obs["continuous"] = obs["continuous"][self.keep_indices]
        return obs


keep_indices = (
    [
        2,
        3,
        4,  # center_path[0:3]
        5,  # center_path_distance[0]
        6,  # distance_down_track[0]
        8,
        9,
        10,  # front[0:3]
    ]
    + list(range(42, 52))  # paths_distance[0:5, 0:2]
    + list(range(52, 67))  # paths_end[0:5, 0:3]
    + list(range(67, 82))  # paths_start[0:5, 0:3]
    + list(range(82, 87))  # paths_width[0:5, 0:1]
    + [89, 90, 91]  # velocity[0:3]
)
# env = ContinuousSubsetWrapper(env, keep_indices=keep_indices)
<<<<<<< HEAD
=======

>>>>>>> 6f8ee2e1fdf92ee3831dfda4d092578fade723f5

# Drop the discrete part: make the observation a pure Box by using only 'continuous'
class ContinuousOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict), (
            "Expected Dict observation space."
        )
        assert "continuous" in self.observation_space.spaces, (
            "Expected 'continuous' key in observation."
        )
        self.observation_space = self.observation_space["continuous"]

    def observation(self, obs):
        # Return only the continuous vector
        return obs["continuous"]

<<<<<<< HEAD
=======

>>>>>>> 6f8ee2e1fdf92ee3831dfda4d092578fade723f5
# env = ContinuousOnlyWrapper(env)

# Load model WITH the environment so SB3 can properly reconstruct observation/action spaces
print("Loading model...")
<<<<<<< HEAD
model = SAC.load("/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer/sac_stk-600000-fullobs.zip", env=env)
=======
# model = SAC.load("sac_stk-600000.zip", env=env)
model = SAC.load("sac_stk_ent_coeff_0.0325_bis.zip", env=env)
# model = SAC.load("sac_stk_ent_coeff_0.0325.zip", env=env)
>>>>>>> 6f8ee2e1fdf92ee3831dfda4d092578fade723f5

# print("Testing")
obs, info = env.reset()
# print(f"Observation shape: {obs.shape}, Observation space: {env.observation_space}")
# print(f"Model observation space: {model.observation_space}")
# print(f"Action space: {env.action_space}")
# print(f"Model action space: {model.action_space}")
<<<<<<< HEAD

# # Verify spaces match
# assert model.observation_space == env.observation_space, \
#     f"Observation space mismatch! Model: {model.observation_space}, Env: {env.observation_space}"
# assert model.action_space == env.action_space, \
#     f"Action space mismatch! Model: {model.action_space}, Env: {env.action_space}"
=======
#
# # Verify spaces match
# assert model.observation_space == env.observation_space, (
#     f"Observation space mismatch! Model: {model.observation_space}, Env: {env.observation_space}"
# )
# assert model.action_space == env.action_space, (
#     f"Action space mismatch! Model: {model.action_space}, Env: {env.action_space}"
# )
>>>>>>> 6f8ee2e1fdf92ee3831dfda4d092578fade723f5
# print("✓ Observation and action spaces match!")

total_reward = 0
step = 0
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    # print(f"Step {step}: action = {action}, obs[0:5] = {obs[:5]}")
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    print(f"  reward = {reward:.3f}, total = {total_reward:.3f}, info: {info}")
    # print(obs['discrete'])
    if terminated or truncated:
        print(f"Episode ended after {step} steps with total reward {total_reward:.3f}")
        # done = True
        obs, info = env.reset()
        
        total_reward = 0
        step = 0

    if step > 150:
        obs, info = env.reset()
        total_reward = 0
        step = 0


env.close()
