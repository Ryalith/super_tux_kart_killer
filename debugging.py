"""
MultiCategorical Distribution for TorchRL - WORKING VERSION

Based on the PyTorch issue #43250 and torchrl documentation, this implementation:
1. Accepts a SINGLE concatenated tensor of logits (not a list)
2. Takes logits as a keyword argument (not probs/softmax)
3. Splits the logits internally based on action_dims

Your network should output ONE concatenated tensor, not a list!
"""

import torch
from torch.distributions import Categorical, Distribution
from typing import Sequence


class MultiCategorical(Distribution):
    """
    Multi-categorical distribution for gym MultiDiscrete action spaces.
    Compatible with torchrl's ProbabilisticActor.
    
    This distribution represents multiple independent categorical distributions.
    Instead of accepting a list of tensors, it accepts a single concatenated
    tensor of logits and splits it internally.
    
    Args:
        logits (torch.Tensor): Concatenated logits tensor of shape 
                              (batch_size, sum(action_dims))
        action_dims (Sequence[int]): Number of categories for each action dimension.
                                     For example, [2, 4, 3] means 3 action dimensions
                                     with 2, 4, and 3 categories respectively.
    
    Example with TorchRL:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import ProbabilisticActor
        >>> 
        >>> # Network outputs CONCATENATED logits (not a list!)
        >>> class ActorNet(nn.Module):
        ...     def __init__(self, obs_dim, action_dims):
        ...         super().__init__()
        ...         total_actions = sum(action_dims)
        ...         self.net = nn.Sequential(
        ...             nn.Linear(obs_dim, 64),
        ...             nn.ReLU(),
        ...             nn.Linear(64, total_actions)  # Single output!
        ...         )
        ...     
        ...     def forward(self, observation):
        ...         return self.net(observation)  # Returns single tensor
        >>> 
        >>> # Wrap in TensorDictModule
        >>> actor_net = ActorNet(obs_dim=10, action_dims=[2, 4, 3])
        >>> actor_module = TensorDictModule(
        ...     actor_net,
        ...     in_keys=["observation"],
        ...     out_keys=["logits"]  # Single key!
        ... )
        >>> 
        >>> # Create ProbabilisticActor with MultiCategorical
        >>> from functools import partial
        >>> actor = ProbabilisticActor(
        ...     actor_module,
        ...     in_keys=["logits"],
        ...     distribution_class=partial(MultiCategorical, action_dims=[2, 4, 3]),
        ...     return_log_prob=True
        ... )
    """
    
    def __init__(self, logits: torch.Tensor, action_dims: Sequence[int], validate_args=None):
        self.action_dims = list(action_dims)
        self.num_actions = len(action_dims)
        
        # Verify logits has correct shape
        total_categories = sum(action_dims)
        if logits.shape[-1] != total_categories:
            raise ValueError(
                f"Expected logits last dimension to be {total_categories} "
                f"(sum of action_dims {action_dims}), got {logits.shape[-1]}"
            )
        
        # Split logits and create categorical distributions
        logits_list = torch.split(logits, self.action_dims, dim=-1)
        self.categoricals = [Categorical(logits=logits_i) for logits_i in logits_list]
        
        # Get batch shape from first categorical
        batch_shape = self.categoricals[0].batch_shape
        
        # Event shape is (num_actions,) - each sample is a vector of actions
        event_shape = torch.Size([self.num_actions])
        
        super().__init__(batch_shape, event_shape, validate_args)
        
        # Store logits for later use
        self._logits = logits
    
    def sample(self, sample_shape=torch.Size()):
        """Sample from each action dimension independently."""
        samples = [cat.sample(sample_shape) for cat in self.categoricals]
        return torch.stack(samples, dim=-1)
    
    def log_prob(self, value: torch.Tensor):
        """
        Compute log probability of actions.
        
        Args:
            value: Tensor of shape (*batch_shape, num_actions)
        
        Returns:
            Tensor of shape (*batch_shape,)
        """
        if value.shape[-1] != self.num_actions:
            raise ValueError(
                f"Expected value to have {self.num_actions} actions, "
                f"got {value.shape[-1]}"
            )
        
        # Compute log prob for each dimension and sum (independent actions)
        log_probs = []
        for i, cat in enumerate(self.categoricals):
            action_i = value[..., i].long()
            log_probs.append(cat.log_prob(action_i))
        
        return torch.stack(log_probs, dim=-1).sum(dim=-1)
    
    def entropy(self):
        """Compute entropy (sum of individual entropies)."""
        entropies = [cat.entropy() for cat in self.categoricals]
        return torch.stack(entropies, dim=-1).sum(dim=-1)
    
    def mode(self):
        """Return the mode (most likely action) for each dimension."""
        modes = [cat.probs.argmax(dim=-1) for cat in self.categoricals]
        return torch.stack(modes, dim=-1)
    
    @property
    def mean(self):
        """Expected value (same as mode for categorical)."""
        return self.mode().float()
    
    @property
    def logits(self):
        """Return the concatenated logits tensor."""
        return self._logits
    
    def expand(self, batch_shape, _instance=None):
        """Expand distribution to new batch shape."""
        new = self._get_checked_instance(MultiCategorical, _instance)
        new.action_dims = self.action_dims
        new.num_actions = self.num_actions
        new._logits = self._logits.expand(batch_shape + (-1,))
        new.categoricals = [
            cat.expand(batch_shape) for cat in self.categoricals
        ]
        super(MultiCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


# ============================================================================
# CORRECTED NETWORK AND ACTOR FOR YOUR USE CASE
# ============================================================================

class PPODiscretePolicyNet(torch.nn.Module):
    """
    CORRECTED VERSION - outputs SINGLE concatenated tensor, not a list!
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        action_dims: Sequence[int],
        Act=torch.nn.ReLU,
    ):
        super().__init__()
        self.action_dims = action_dims
        
        # Build network layers
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(Act())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(Act())
        
        # Output layer: concatenated logits for all action dimensions
        embed_dim = hidden_dims[-1]
        total_categories = sum(action_dims)
        layers.append(torch.nn.Linear(embed_dim, total_categories))
        # DO NOT apply softmax here! Distributions expect raw logits
        
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, continuous, discrete):
        """Returns SINGLE tensor of concatenated logits."""
        obs = torch.cat([continuous, discrete], dim=-1)
        logits = self.net(obs)  # Shape: (batch, sum(action_dims))
        return logits  # Return single tensor, not a list!


def create_ppo_discrete_actor(env_specs, hidden_net_dims: Sequence[int]):
    """
    Creates a properly configured ProbabilisticActor for multi-discrete actions.
    
    Args:
        env_specs: Environment specifications from torchrl env
        hidden_net_dims: Hidden layer dimensions for the network
    
    Returns:
        ProbabilisticActor configured for multi-discrete actions
    """
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor
    from functools import partial
    
    # Get dimensions from env specs
    # You'll need to implement these helper functions based on your env
    # input_dim = get_observation_vector_dim(env_specs)
    # action_dims = get_action_vector_dims(env_specs)
    
    # For demonstration, using placeholder values
    input_dim = 10  # Replace with your actual function
    action_dims = [2, 4, 3]  # Replace with your actual function
    
    # Create network that outputs SINGLE concatenated logits tensor
    policy_net = PPODiscretePolicyNet(input_dim, hidden_net_dims, action_dims)
    
    # Wrap in TensorDictModule with SINGLE output key
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["continuous", "discrete"],
        out_keys=["logits"]  # Single key for concatenated logits!
    )
    
    # Create distribution class with action_dims baked in
    distribution_class = partial(MultiCategorical, action_dims=action_dims)
    
    # Create ProbabilisticActor
    # CRITICAL: You MUST specify out_keys=["action"] to tell the actor 
    # where to write the sampled action!
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],  # Single key!
        out_keys=["action"],  # THIS IS REQUIRED - where to write the action
        distribution_class=distribution_class,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        # spec=env_specs["input_spec"]["full_action_spec"]["action"],
    )
    
    return actor


# ============================================================================
# DEBUGGING SCRIPT FOR YOUR ISSUE
# ============================================================================

def debug_actor_collector_issue():
    """
    Diagnostic script to debug the KeyError: 'action' not found
    
    Based on your error trace, the issue is that ProbabilisticActor is not
    writing the 'action' key to the TensorDict. Let's diagnose why.
    """
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule
    from torchrl.modules import ProbabilisticActor
    from functools import partial
    import torch
    
    print("="*70)
    print("DEBUGGING: Why is 'action' key missing?")
    print("="*70)
    
    # Simulate your network and actor setup
    class TestPolicyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(10, 9)  # 10 input, 9 output (2+4+3)
        
        def forward(self, continuous, discrete):
            obs = torch.cat([continuous, discrete], dim=-1)
            return self.net(obs)
    
    # Create the actor
    policy_net = TestPolicyNet()
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["continuous", "discrete"],
        out_keys=["logits"]
    )
    
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],  # This should write to "action" key
        distribution_class=partial(MultiCategorical, action_dims=[2, 4, 3]),
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Create a test TensorDict like your environment would provide
    test_td = TensorDict({
        "continuous": torch.randn(4, 5),
        "discrete": torch.randn(4, 5),
    }, batch_size=[4])
    
    print("Input TensorDict keys:", test_td.keys())
    print("Input TensorDict:\n", test_td)
    
    # Call the actor
    try:
        output_td = actor(test_td)
        print("\n✓ Actor executed successfully!")
        print("Output TensorDict keys:", output_td.keys())
        print("\nAction shape:", output_td["action"].shape)
        print("Action:", output_td["action"])
        if "sample_log_prob" in output_td.keys():
            print("Log prob shape:", output_td["sample_log_prob"].shape)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("POSSIBLE ISSUES TO CHECK:")
    print("="*70)
    print("""
    1. Does your environment actually provide 'continuous' and 'discrete' keys?
       - Check env.observation_spec to see what keys it provides
       - The collector expects these keys to exist
    
    2. Is your action_spec compatible with the output shape?
       - MultiCategorical returns shape (batch, num_actions)
       - Make sure your env's action_spec expects this shape
    
    3. Are you passing the correct env_specs to your actor?
       - The spec parameter in ProbabilisticActor must match env's action_spec
    
    4. Try testing your actor standalone BEFORE using it in a collector:
       ```python
       # Test the actor
       test_data = TensorDict({
           "continuous": torch.randn(1, cont_dim),
           "discrete": torch.randn(1, disc_dim),
       }, batch_size=[1])
       
       output = self.ppo_actor(test_data)
       print("Output keys:", output.keys())
       print("Action:", output.get("action", "KEY MISSING!"))
       ```
    """)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the debugging script first
    debug_actor_collector_issue()
    
    print("\n\n")
    print("="*70)
    print("EXAMPLE 1: Standalone MultiCategorical")
    print("="*70)
    
    batch_size = 4
    action_dims = [2, 4, 3]  # 3 action dimensions
    total_cats = sum(action_dims)  # 9 total categories
    
    # Create concatenated logits (NOT a list!)
    logits = torch.randn(batch_size, total_cats)
    
    # Create distribution
    dist = MultiCategorical(logits=logits, action_dims=action_dims)
    
    print(f"Action dimensions: {action_dims}")
    print(f"Total categories: {total_cats}")
    print(f"Logits shape: {logits.shape}")
    
    # Sample actions
    actions = dist.sample()
    print(f"\nSampled actions shape: {actions.shape}")
    print(f"Actions:\n{actions}")
    
    # Compute log probability
    log_prob = dist.log_prob(actions)
    print(f"\nLog probabilities: {log_prob}")
    
    # Entropy
    entropy = dist.entropy()
    print(f"Entropy: {entropy}")
    
    print("\n" + "="*70)
    print("EXAMPLE 2: With Neural Network (Corrected)")
    print("="*70)
    
    obs_dim = 10
    hidden_dims = [64, 64]
    action_dims = [2, 4, 3]
    
    # Create network
    net = PPODiscretePolicyNet(obs_dim, hidden_dims, action_dims)
    
    # Test forward pass
    continuous = torch.randn(batch_size, 5)
    discrete = torch.randn(batch_size, 5)
    
    logits_output = net(continuous, discrete)
    print(f"Network output shape: {logits_output.shape}")  # Should be (4, 9)
    print(f"Expected shape: ({batch_size}, {sum(action_dims)})")
    
    # Create distribution from network output
    dist = MultiCategorical(logits=logits_output, action_dims=action_dims)
    actions = dist.sample()
    print(f"Sampled actions: {actions}")
    
    print("\n" + "="*70)
    print("KEY CHANGES TO YOUR CODE:")
    print("="*70)
    print("""
THE MOST LIKELY ISSUE:
    KeyError: 'key "action" not found in TensorDict'
    
Based on your error, the issue is likely ONE of these:

A. WRONG ACTION SPEC - Your environment's action_spec doesn't match MultiCategorical output
   - MultiCategorical outputs shape (batch, num_actions) with integer values
   - Check your env.action_spec - it might expect a different shape or type
   
   FIX: Remove or fix the 'spec' parameter in ProbabilisticActor:
   ```python
   super().__init__(
       module=policy_module,
       in_keys=["logits"],
       out_keys=["action"],
       distribution_class=partial(MultiCategorical, action_dims=action_dims),
       return_log_prob=True,
       log_prob_key="sample_log_prob",
       # Try REMOVING this line or fixing it:
       # spec=env_specs["input_spec"]["full_action_spec"]["action"],
   )
   ```

B. ENV DOESN'T HAVE THE RIGHT OBSERVATION KEYS
   - Your policy expects ["continuous", "discrete"]
   - Make sure your env provides these exact keys
   
   DEBUG: Print your env's observation_spec:
   ```python
   print("Env observation keys:", env.observation_spec.keys())
   ```

REQUIRED CODE CHANGES:
1. REMOVE softmax from network - use raw logits
2. Network outputs SINGLE concatenated tensor, not list
3. TensorDictModule out_keys=["logits"] (singular)
4. ProbabilisticActor in_keys=["logits"], out_keys=["action"]
5. USE partial(MultiCategorical, action_dims=[...])
6. REMOVE or FIX the 'spec' parameter (most likely culprit!)

Your corrected PPODiscreteProbaActor:

class PPODiscreteProbaActor(ProbabilisticActor):
    def __init__(self, env_specs, hidden_net_dims: Sequence[int]):
        input_dim = get_observation_vector_dim(env_specs)
        action_dims = get_action_vector_dims(env_specs)
        policy_net = PPODiscretePolicyNet(input_dim, hidden_net_dims, action_dims)
        
        policy_module = TensorDictModule(
            policy_net, 
            in_keys=["continuous", "discrete"], 
            out_keys=["logits"]
        )
        
        super().__init__(
            module=policy_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=partial(MultiCategorical, action_dims=action_dims),
            return_log_prob=True,
            log_prob_key="sample_log_prob",
            # TRY REMOVING THIS LINE - it's likely the issue:
            # spec=env_specs["input_spec"]["full_action_spec"]["action"],
        )

RUN THE DEBUG SCRIPT ABOVE to test your actor before using it in the collector!
""")
