from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import MultiDiscrete

from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator

from tensordict.nn import TensorDictModule

from .env import get_observation_vector_dim, get_action_vector_dims
#
# PPO Agents
#


class PPODiscretePolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        action_dims: Sequence[int],
        Act=nn.ReLU,
    ):
        """
        nn.Module accepting a tensor of observations as input
        returns the parameters to a torch multi categorical distribution
        Intended to be used wrapped in a TensorDictModule with a ProbabilisticActor
        """
        super().__init__()
        # TODO: write the net,
        # since output is meant to be used by multi categorical:
        # return a Sequence of n outputs
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(Act())

        for i in range(1, len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(Act())

        embed_dim = hidden_dims[-1]
        total_categories = sum(action_dims)
        layers.append(nn.Linear(embed_dim, total_categories))

        self.embed = nn.Sequential(*layers)

        self.heads = [nn.Softmax(mctg_dim) for mctg_dim in action_dims]

        self.action = action_dims
        self.offsets = [0]
        for i, mctg_dim in enumerate(action_dims[:-1]):
            self.offsets.append(mctg_dim + self.offsets[i])

    def forward(self, obs) -> Sequence[torch.Tensor]:
        embedding = self.embed(obs)
        logits = [
            # Apply softmax
            head(embedding[..., offset : offset + act_dim])
            for head, offset, act_dim in zip(self.heads, self.offsets, self.action_dims)
        ]
        return {"logits": logits}


class DiscreteProbaActor(ProbabilisticActor):
    def __init__(
        self,
        env_specs,
        hidden_net_dims: Sequence[int],
        Act=nn.ReLU,
    ):
        input_dim = get_observation_vector_dim(env_specs)
        action_dims = get_action_vector_dims(env_specs)

        policy_net = PPODiscretePolicyNet(input_dim, hidden_net_dims, action_dims, Act)

        policy_module = TensorDictModule(
            policy_net, in_keys=["observation"], out_keys=["logits"]
        )

        super().__init__(
            module=policy_module,
            spec=env_specs["input_spec"]["full_action_spec"]["action"],
            in_keys=["logits"],
            distribution_class=MultiDiscrete,
            distribution_kwargs={
                "nvec": torch.tensor(action_dims),
            },
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
            log_prob_key="sample_log_prob",
        )


class PPOValueNet(nn.Module):
    # Same idea as policy net but for use with a torchrl ValueOperator
    pass


#
# SAC Agents: Not Working for now
#


# class SACContinuousActor(nn.Module):
#     def __init__(self, state_dim, hidden_layers, action_dim, min_std=1e-4):
#         """Creates a new Squashed Gaussian actor for use in a SAC Algo
#
#         TODO: Make sure this approach works for discreet outputs which isn't obvious
#
#         :param state_dim: The dimension of the state space
#         :param hidden_layers: Hidden layer sizes
#         :param action_dim: The dimension of the action space
#         :param min_std: The minimum standard deviation, defaults to 1e-4
#         """
#         super().__init__()
#         self.min_std = min_std
#         backbone_dim = [state_dim] + list(hidden_layers)
#         self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
#         self.backbone = nn.Sequential(*self.layers)
#         self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
#         self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
#         self.softplus = nn.Softplus()
#
#         # cache_size avoids numerical infinites or NaNs when
#         # computing log probabilities
#         self.tanh_transform = TanhTransform(cache_size=1)
#
#     def normal_dist(self, state: torch.Tensor):
#         """Compute normal distribution given observation(s)"""
#
#         backbone_output = self.backbone(state)
#         mean = self.last_mean_layer(backbone_output)
#         std_out = self.last_std_layer(backbone_output)
#         std = self.softplus(std_out) + self.min_std
#         # Independent ensures that we have a multivariate
#         # Gaussian with a diagonal covariance matrix (given as
#         # a vector `std`)
#         return Independent(Normal(mean, std), 1)
#
#     def log_prob(self, state, action):
#         normal_dist = self.normal_dist(state)
#         action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
#         log_prob = action_dist.log_prob(action)
#         return log_prob
#
#     def forward(self, state, stochastic=True):
#         """Computes the action a_t and its log-probability p(a_t| s_t)
#
#         :param stochastic: True when sampling
#         """
#         normal_dist = self.normal_dist(state)
#         action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
#         if stochastic:
#             # Uses the re-parametrization trick
#             action = action_dist.rsample()
#         else:
#             # Directly uses the mode of the distribution
#             action = self.tanh_transform(normal_dist.mode)
#
#         # This line allows to deepcopy the actor...
#         self.tanh_transform._cached_x_y = [None, None]
#         return action


# class SACContinuousQCritic(nn.Module):
#     def __init__(self, state_dim: int, hidden_layers: list[int], action_dim: int):
#         """Creates a new critic agent $Q(s, a)$ for use in a SAC Algo
#
#         :param state_dim: The number of dimensions for the observations
#         :param hidden_layers: The list of hidden layers for the NN
#         :param action_dim: The numer of dimensions for actions
#         """
#         super().__init__()
#         self.is_q_function = True
#         self.model = build_mlp(
#             [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
#         )
#
#     def forward(self, transition: Transition):
#         obs_act = torch.cat((transition.state, transition.action), dim=1)
#         q_value = self.model(obs_act).squeeze(-1)
#         return q_value
