import stk_actor
from pathlib import Path

if __name__ == "__main__":

    stk_actor.data.set_data_folder(Path("/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer"))
    env = stk_actor.env.make_discrete_env(render_mode = "human", use_ai = False) # render_mode : None or "human" to get a display. use_ai: False to use our agent's actions, True to use the native bot's actions
    env.reset()
    ppo = stk_actor.algorithms.PPOAlgo(env)
    ppo.load_checkpoint("ppo_pretrained-100000.ckpt", True)
    ppo.train(128*1000, 128, 32, "cpu")
    env.close()