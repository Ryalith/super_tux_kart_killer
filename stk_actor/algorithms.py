# TODO: create SAC class
# Take an env and a config
# Create the actors and critics
# implement training loop:
# reset the env, store actions sampled from the actor into the memory
# train by sampling random batches from the replay memory
# backprop the SAC loss on the batch

# TODO: add tensorboard loggers to check the status of training

import torch
import gymnasium as gym
from .actors import ReplayBuffer, Transition
from .utils import (
    stk_state_dict_to_tensor,
    stk_action_tensor_to_dict,
)


class SACAlgo:
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.actor = ...
        self.q_critic_1 = ...
        self.q_critic_2 = ...
        self.q_tgt_critic_1 = ...
        self.q_tgt_critic_2 = ...
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    def _fill_replay_buffer(self, max_iter: int):
        ix = 0
        done = False
        state, *_ = self.env.reset()

        while not done:
            ix += 1
            state_tensor = stk_state_dict_to_tensor(state)
            action_tensor = self.actor(state_tensor)
            action = stk_action_tensor_to_dict(action_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state_tensor = stk_state_dict_to_tensor(next_state)
            done = truncated or terminated
            self.replay_buffer.push(
                Transition(
                    state_tensor,
                    action_tensor,
                    next_state_tensor,
                    torch.tensor(reward),
                    torch.tensor(terminated),
                    torch.tensor(truncated),
                )
            )
            state = next_state
            if ix >= max_iter:
                break

    def _train_on_replay_buffer(self):
        ...
        # TODO: define SAC loss compute it and back propagate the loss
        # There might be a need to change the way the replay buffer is implemented to avoid resampling
        # Maybe define the deplay buffer as a dataset and create a data loader ?

    def train(self, epochs):
        for epoch in range(epochs):
            self._fill_replay_buffer(...)
            self._train_on_replay_buffer(...)
