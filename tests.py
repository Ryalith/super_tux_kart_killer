import stk_actor
from pathlib import Path
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import torch
from tensordict import TensorDict
import torch.nn.functional as F
import numpy as np

def state_to_obs_tensordict(state):
    discrete_raw = torch.as_tensor(state["discrete"]).long()
    discrete_nvec = torch.as_tensor([10,  7,  7,  7,  7,  7,  2,  4, 11]).long()
    discrete_one_hot = torch.cat(
        [F.one_hot(val, num_classes=int(n)).float() for val, n in zip(discrete_raw, discrete_nvec)]
    )
    obs_td = TensorDict(
        {
            "continuous": torch.as_tensor(state["continuous"]).float(),
            "discrete": discrete_one_hot,
        },
        batch_size=[],
    )

    return obs_td


def states_to_obs_tensordict(states):
    """Batchified variant that accepts a list of states."""
    discrete_nvec = torch.as_tensor([10, 7, 7, 7, 7, 7, 2, 4, 11]).long()
    batch_discrete = torch.as_tensor(np.array([s["discrete"] for s in states])).long()

    discrete_one_hot = torch.cat(
        [
            F.one_hot(batch_discrete[:, idx], num_classes=int(n)).float()
            for idx, n in enumerate(discrete_nvec)
        ],
        dim=-1,
    )

    obs_td = TensorDict(
        {
            "continuous": torch.stack(
                [torch.as_tensor(s["continuous"]).float() for s in states]
            ),
            "discrete": discrete_one_hot,
        },
        batch_size=[len(states)],
    )

    return obs_td


if __name__ == "__main__":
    ppo_actor = stk_actor.agents.PPODiscreteProbaActor(
            None, [128, 128, 128]
        )  

    env = gym.make("supertuxkart/flattened_multidiscrete-v0", render_mode="human", agent=AgentSpec(use_ai=False))
    state1, *_ = env.reset()
    print(state1)
    action = env.action_space.sample()
    print(action)
    state2, *_ = env.step(action)
    print(state2)
    obs_td1 = state_to_obs_tensordict(state1)
    logits1 = ppo_actor(obs_td1)['logits']
    obs_td2 = state_to_obs_tensordict(state2)
    logits2 = ppo_actor(obs_td2)['logits']
    print(logits1)
    print(logits2)
    states = [state1, state2]
    obs_td = states_to_obs_tensordict(states)
    logits = ppo_actor(obs_td)['logits']
    print(logits)
    env.close()