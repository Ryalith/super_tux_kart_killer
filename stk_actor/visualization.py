# Todo import policies and load weights
# load the env for render
# create a loop that steps the env for a certain number of frames
# the visuals should update

from .env import make_discrete_env_for_render
from .algorithms import PPOAlgo


def visualize_PPO_policy(run_name: str, max_steps=10_000):
    env = make_discrete_env_for_render()

    algo = PPOAlgo(env=env)
    algo.load_checkpoint(f"/home/rya/.cache/stk_killer/runs/{run_name}")
    actor = algo.ppo_actor

    done = False
    it = 0
    obs_td = env.reset()

    # Start render loop
    env.render()
    while not done and it < max_steps:
        act_td = actor(obs_td)
        env.step(act_td)
        it += 1
