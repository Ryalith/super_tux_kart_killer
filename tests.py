import stk_actor
from pathlib import Path

if __name__ == "__main__":

    stk_actor.data.set_data_folder(Path("/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer"))
    env = stk_actor.env.make_discrete_env()
    ppo = stk_actor.algorithms.PPOAlgo(env)
    ppo.train(128*100, 128, 32, "cpu")