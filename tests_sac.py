import torch
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import SAC
from datetime import datetime

env = gym.make(
    "supertuxkart/flattened_continuous_actions-v0",
    render_mode=None,
    agent=AgentSpec(use_ai=False),
)

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

model = SAC(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=10000,
    batch_size=1024,
    tau=0.005,
    gamma=0.995,
    train_freq=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    ent_coef=0.0325,  # Dichotomy 0.0325 0.01
    tensorboard_log=f"runs/{current_time}-SAC",
)

# model = SAC.load(
#     "sac_stk_ent_coeff_0.055", env=env, tensorboard_log=f"runs/{current_time}-SAC"
# )

model.learn(total_timesteps=450_000, log_interval=1)

print("Saving")
model.save("sac_stk")

del model  # remove to demonstrate saving and loading

env.close()
# print("Loading")
# model = SAC.load("sac_pendulum")

# print("Testing")
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()
