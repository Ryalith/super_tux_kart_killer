import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import stk_actor
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import pickle

action_dims = [5, 2, 2, 2, 2, 2, 7]
cum = [0]
for d in action_dims:
    cum.append(cum[-1] + d)   # [0,5,7,9,11,13,15,22]

ce = nn.CrossEntropyLoss(reduction = "none")

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



def bc_loss(targets, logits):
    """
    states: states of the environment, containing obs and actions of the bots
    logits: [B, 22]  (ints in the correct ranges)
    """
    per_head_losses = []
    correct = []

    for i, d in enumerate(action_dims):
        start, end = cum[i], cum[i+1]
        logits_i = logits[..., start:end]
        target_i = targets[..., i]
        loss_i = ce(logits_i, target_i)
        acc_i = (logits_i.argmax(-1) == target_i).sum()
        per_head_losses.append(loss_i)  # list of (B)
        correct.append(acc_i)

    per_head_losses = torch.stack(per_head_losses, dim=0).T # (B, 7)
    loss = per_head_losses.mean()

    return loss, correct, targets.shape[0] 

class StateDataset(Dataset):
    def __init__(self, states):
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

def collate_fn(states):
    actions = torch.tensor(np.array([s['action'] for s in states]))

    return states_to_obs_tensordict(states), actions

if __name__ == '__main__':

    states_path = "/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer/pretrain_training_states_100000.pkl"


    # Load the states from the given path
    with open(states_path, "rb") as f:
        training_states = pickle.load(f)

    num_steps = len(training_states)

    random.shuffle(training_states)

    # Split training_states into train and val
    split_ratio = 0.8
    split_idx = int(num_steps * split_ratio)
    train_states = training_states[:split_idx]
    val_states = training_states[split_idx:]

    train_dataset = StateDataset(train_states)
    val_dataset = StateDataset(val_states)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    ppo_actor = stk_actor.agents.PPODiscreteProbaActor(
            None, [128, 128, 128]
        )    

    ppo_actor = ppo_actor

    print("Starting training")
    num_epochs = 5
    optim = torch.optim.Adam(ppo_actor.parameters(), lr = 1e-3)

    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        for obs_td, actions in train_loader:
            optim.zero_grad()
            logits = ppo_actor(obs_td)['logits']
            loss, correct, total = bc_loss(actions, logits)
            acc = np.array(correct) / total
            loss.backward()
            optim.step()
            global_step += 1

        with torch.no_grad():
            correct_pred = np.zeros(7)
            total_pred = 0
            for obs_td, actions in train_loader:
                logits = ppo_actor(obs_td)['logits']
                loss, correct, total = bc_loss(actions, logits)
                correct_pred += np.array(correct)
                total_pred += total
            acc = correct_pred / total_pred
            print(f"Accuracy: {acc}")
            
        

    checkpoint_path = f"/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer/ppo_pretrained-{num_steps}.ckpt"
    checkpoint = {
            "ppo_actor_state_dict": ppo_actor.state_dict(),
        }
    torch.save(checkpoint, checkpoint_path)
    