import random
from collections import deque, namedtuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import (
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from .utils import build_mlp

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated", "truncated")
)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> Transition:
        transitions = random.sample(self.memory, batch_size)
        # See torch DQN tutorial for why we do this
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim, min_std=1e-4):
        """Creates a new Squashed Gaussian actor for use in a SAC Algo

        TODO: Make sure this approach works for discreet outputs which isn't obvious

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-4
        """
        super().__init__()
        self.min_std = min_std
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.softplus = nn.Softplus()

        # cache_size avoids numerical infinites or NaNs when
        # computing log probabilities
        self.tanh_transform = TanhTransform(cache_size=1)

    def normal_dist(self, state: torch.Tensor):
        """Compute normal distribution given observation(s)"""

        backbone_output = self.backbone(state)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = self.softplus(std_out) + self.min_std
        # Independent ensures that we have a multivariate
        # Gaussian with a diagonal covariance matrix (given as
        # a vector `std`)
        return Independent(Normal(mean, std), 1)

    def log_prob(self, state, action):
        normal_dist = self.normal_dist(state)
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        log_prob = action_dist.log_prob(action)
        return log_prob

    def forward(self, state, stochastic=True):
        """Computes the action a_t and its log-probability p(a_t| s_t)

        :param stochastic: True when sampling
        """
        normal_dist = self.normal_dist(state)
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        if stochastic:
            # Uses the re-parametrization trick
            action = action_dist.rsample()
        else:
            # Directly uses the mode of the distribution
            action = self.tanh_transform(normal_dist.mode)

        # This line allows to deepcopy the actor...
        self.tanh_transform._cached_x_y = [None, None]
        return action


class ContinuousQCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: list[int], action_dim: int):
        """Creates a new critic agent $Q(s, a)$ for use in a SAC Algo

        :param state_dim: The number of dimensions for the observations
        :param hidden_layers: The list of hidden layers for the NN
        :param action_dim: The numer of dimensions for actions
        """
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, transition: Transition):
        obs_act = torch.cat((transition.state, transition.action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        return q_value
