from typing import Sequence

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torch import distributions as d
from torch.distributions import constraints
from torch.distributions.constraints import Constraint
from torch.distributions.utils import lazy_property, probs_to_logits
from torch.nn import functional as F
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator

from .env import get_action_vector_dims, get_observation_vector_dim


#
# PPO Agents
#
def multi_logits_to_probs(logits, n_categories: Sequence[int]):
    """
    Converts a tensor of logits into probabilities for a sequence of multinomial variables
    whose number of possible outcomes is stored in n_categories.
    """
    probs = torch.empty_like(logits)
    start = 0
    for i, n_cat in enumerate(n_categories):
        probs[..., start:start+n_cat] = F.softmax(logits[..., start:start+n_cat], dim=-1)
        start+=n_cat
    return probs

class _MultiIntegerInterval(Constraint):
    """
    Constrain to multiple integer intervals along the last dimension of the value tensor
    `[lower_bound_0, lower_bound_1, ...], [upper_bound_0, upper_bound_1, ...]`.
    """

    is_discrete = True

    def __init__(self, lower_bounds: Sequence[int], upper_bounds: Sequence[int]):
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError(
                f"lower and upper bounds should have the same size but found {len(lower_bounds.shape)=} and {len(upper_bounds.shape)=}"
            )
        self.lower_bounds = torch.tensor(lower_bounds)
        self.upper_bounds = torch.tensor(upper_bounds)
        super().__init__()

    def check(self, value):
        if value.shape[-1] != self.lower_bounds.shape[-1]:
            return False
        return torch.all(
            (value % 1 == 0)
            & (self.lower_bounds <= value)
            & (value <= self.upper_bounds)
        )

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bounds={self.lower_bounds}, upper_bounds={self.upper_bounds})"
        )
        return fmt_string


multi_integer_interval = _MultiIntegerInterval


class MultiCategorical(d.Distribution):
    def __init__(
        self, n_categories: Sequence[int], logits=None, probs=None, validate_args=None
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")

            start = 0
            segments = []
            for n_cat in n_categories:
                seg = probs[..., start:start + n_cat]
                seg = seg / seg.sum(dim=-1, keepdim=True)
                segments.append(seg)
                start += n_cat
            probs = torch.cat(segments, dim=-1)
            self.probs = probs
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            start = 0
            segments = []
            for n_cat in n_categories:
                seg = logits[..., start : start + n_cat]
                seg = seg - seg.logsumexp(dim=-1, keepdim=True)
                segments.append(seg)
                start += n_cat

            logits = torch.cat(segments, dim=-1)
            self.logits = logits
            


        self._param = self.probs if probs is not None else self.logits
        if self._param.shape[-1] != sum(n_categories):
            raise ValueError(
                f"The number of parameters (probs or logits) should match the total number of categories; but we received param of shape {self._param.shape} and {sum(n_categories)=}"
            )
        self._num_params = sum(n_categories)
        self._n_categories = torch.tensor(n_categories)
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        event_shape = (len(self._n_categories),)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @lazy_property
    def logits(self) -> torch.Tensor:
        # We can reuse probs_to_logits because we are simply clamping and computing a logarithm
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self) -> torch.Tensor:
        return multi_logits_to_probs(self.logits, self._n_categories)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultiCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_params,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_params = self._num_params
        super(MultiCategorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=True)
    def support(self):
        return multi_integer_interval(
            [0] * len(self._n_categories), self._n_categories - 1
        )

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_params)
        batch_size = probs_2d.shape[0]
        samples_2d = torch.empty(
            sample_shape.numel(), batch_size, len(self._n_categories),
            dtype=torch.long, device=probs_2d.device
        )
        start = 0
        for i, n_cat in enumerate(self._n_categories):
            cat_samples = torch.multinomial(
                probs_2d[..., start:start + n_cat], sample_shape.numel(), True
            )   # shape: [batch_size, sample_size]

            samples_2d[:, :, i] = cat_samples.permute(1, 0)
            start += n_cat
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, values):
        if self._validate_args:
            self._validate_sample(values)
        values = values.long()  # batch, _event_shape,
        # logits of shape _num_params
        # value, log_pmf = torch.broadcast_tensors(value, self.logits)
        # value = value[..., :1]
        # return log_pmf.gather(-1, value).squeeze(-1)
        # TODO: for loop and gather
        start = 0
        all_logits = self.logits
        log_pmf = torch.zeros(values.shape[:-1])
        for i, n_cat in enumerate(self._n_categories):
            value = values[..., i]
            logits = all_logits[..., start : start + n_cat]
            log_pmf += logits.gather(-1, value.unsqueeze(-1)).squeeze(-1)
            start += n_cat
        return log_pmf

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution i.e the most probable vector
        """
        _mode = torch.empty(self.logits.shape[:-1] + (len(self._n_categories),), dtype=torch.long, device=self.logits.device)
        start = 0
        for i, n_cat in enumerate(self._n_categories):
            _mode[..., i] = self.probs[..., start : start + n_cat].argmax(dim=-1)
            start += n_cat
        return _mode

    @property
    def deterministic_sample(self):
        return self.mode

    # def enumerate_support(self, expand=True):
    #     num_events = self._num_events
    #     values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
    #     values = values.view((-1,) + (1,) * len(self._batch_shape))
    #     if expand:
    #         values = values.expand((-1,) + self._batch_shape)
    #     return values


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

        self.back_bone = nn.Sequential(*layers)

    def forward(self, continuous, discrete) -> Sequence[torch.Tensor]:
        obs = torch.cat([continuous, discrete], dim=-1)
        logits = self.back_bone(obs)
        # logits_list = [
        #     logits[..., offset : offset + act_dim]
        #     for offset, act_dim in zip(self.offsets, self.action_dims)
        # ]
        # return tuple(logits_list)
        return logits


class PPODiscreteProbaActor(ProbabilisticActor):
    def __init__(
        self,
        env_specs,
        hidden_net_dims: Sequence[int],
    ):
        if env_specs is not None:
            input_dim = get_observation_vector_dim(env_specs)
            action_dims = get_action_vector_dims(env_specs)
        else:
            input_dim = 154
            action_dims = [5, 2, 2, 2, 2, 2, 7]

        policy_net = PPODiscretePolicyNet(input_dim, hidden_net_dims, action_dims)

        policy_module = TensorDictModule(
            policy_net,
            in_keys=["continuous", "discrete"],
            out_keys=["logits"],
        )

        super().__init__(
            module=policy_module,
            # spec=env_specs["input_spec"]["full_action_spec"]["action"],
            in_keys=["logits"],
            distribution_class=MultiCategorical,
            distribution_kwargs={"n_categories": action_dims},
            return_log_prob=True,
            # we'll need the log-prob for the numerator of the importance weights
            log_prob_key="sample_log_prob",
        )


class PPOValueNet(nn.Module):
    # Same idea as policy net but for use with a torchrl ValueOperator
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], Act=nn.Tanh):
        """
        nn.Module accepting a tensor of observations as input
        returns the parameters to a torch multi categorical distribution
        intended to be used wrapped in a TensorDictModule with a ProbabilisticActor
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

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, continuous, discrete) -> Sequence[torch.Tensor]:
        obs = torch.cat([continuous, discrete], dim=-1)
        return self.net(obs)


class PPOValueOperator(ValueOperator):
    def __init__(self, env_specs, hidden_net_dims: Sequence[int]):
        input_dim = get_observation_vector_dim(env_specs)

        value_net = PPOValueNet(input_dim, hidden_net_dims)

        super().__init__(value_net, in_keys=["continuous", "discrete"])


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
