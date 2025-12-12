import gymnasium as gym
from pystk2_gymnasium import AgentSpec



def extract_states(n_steps):
    env = gym.make("supertuxkart/flattened_multidiscrete-v0", render_mode="human", agent=AgentSpec(use_ai=True))
    state, *_ = env.reset()
    action = env.action_space.sample()
    states = []
    i = 0
    while i < n_steps:
        try:
            state, _, terminated, truncated, _ = env.step(action)
        except:
            print(f"Restarting the environment, step {i+1}/{n_steps}")
            env = gym.make("supertuxkart/flattened_multidiscrete-v0", render_mode="human", agent=AgentSpec(use_ai=True))
            state, *_ = env.reset()
            continue
        if terminated or truncated:
            continue
        states.append(state)
        i += 1
    env.close()
    return states





if __name__ == '__main__':
    
    num_steps = 100000
    
    print("Extracting training states")
    training_states = extract_states(num_steps)
    # val_states = extract_states(num_steps // 5)
    # Save the extracted training states to disk for future use
    import pickle
    with open(f'/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer/pretrain_training_states_{num_steps}.pkl', 'wb') as f:
        pickle.dump(training_states, f)

    
    