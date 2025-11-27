# TODO: create SAC class
# Take an env and a config
# Create the actors and critics
# implement training loop:
# reset the env, store actions sampled from the actor into the memory
# train by sampling random batches from the replay memory
# backprop the SAC loss on the batch

# TODO: Look into Hybrid models than can handle a mix of discrete continuous actions

# TODO: add tensorboard loggers to check the status of training

import copy

import gymnasium as gym
import torch
import torchrl

from .agents import ReplayBuffer, Transition
from .utils import (
    stk_action_tensor_to_dict,
    stk_state_dict_to_tensor,
)


class PPOAlgo:
    default_config = {
        "buffer_capactiy": 1024,
        "batch_size": 128,
        "discount_factor": 0.99,
        "critic_coef": 1.0,
    }

    def __init__(self, env, config: dict | None = None):
        """
        Instantiate a PPOAlgorithm to be used with discreet actions
        """
        self.config = config if config is not None else self.default_config
        self.v_critic = ...
        self.old_v_critic = copy.deepcopy(self.v_critic)
        self.kl_agent = ...
        self.ppo_actor = ...
        self.old_ppo_actor = copy.deepcopy(self.ppo_actor)
        self.replay_buffer = ReplayBuffer(self.config.buffer_capacity)

    def _fill_replay_buffer(self, max_iter: int):
        # TODO: torch rl might have a better replay buffer implementation for parallel env exploration
        state, *_ = self.env.reset()

        with torch.no_grad():
            for ix in range(max_iter):
                ix += 1
                state_tensor = stk_state_dict_to_tensor(state)
                action_tensor = self.ppo_actor(state_tensor)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action_tensor.cpu().numpy()
                )
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
                if done:
                    break

    def _train_one_step(self):
        # TODO: i'm not sure if we train on transitions or normaly ordered batches
        transitions = self.replay_buffer.sample(self.config.batch_size)

        terminated = torch.cat(transitions.terminated)
        reward = torch.cat(transitions.reward)

        # Compute V Values and resulting advantage
        v_value = ...  # use v_agent
        old_v_value = ...  # use old_v_agent
        advantage = ...  # see bbrl utils gae

        # Compute Critic loss with TD(0)
        target = reward + self.config.discount_factor * old_v_value * (
            1 - terminated.int()
        )
        critic_loss = (
            torch.nn.functional.mse_loss(v_value, target) * self.config.critic_coef
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_agent.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()

        # Compute Actor loss

    def train(self):
        pass


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
