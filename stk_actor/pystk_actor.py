from typing import List, Callable
import gymnasium as gym
import torch
import torch.nn as nn
from pathlib import Path
from bbrl.agents import Agent
import zipfile
import os

#: The base environment name (you can change that)
env_name = "supertuxkart/flattened_continuous_actions-v0"

#: Player name (you must change that)
player_name = "Tuxyz"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return []


class RandomActorAgent(Agent):
    """
    Simple random agent that samples actions from the action space.
    """
    
    def __init__(self, action_space: gym.spaces.Space, name: str = "random_actor"):
        super().__init__(name=name)
        self.action_space = action_space
    
    def forward(self, t: int, **kwargs):
        """Sample a random action from the action space.
        
        :param t: Time step (BBRL passes this as positional argument)
        """
        # Sample action from the action space
        action = self.action_space.sample()
        # Convert to tensor
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        
        # Ensure action has at least 1 dimension (not 0-dimensional/scalar)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        
        # Write to workspace using self.set() (BBRL Agent method)
        self.set(("action", t), action_tensor)


class NativeSACPolicy(nn.Module):
    """
    Native PyTorch implementation of SB3's MultiInputPolicy for SAC.
    This matches the architecture used by SB3's MultiInputPolicy.
    """
    
    def __init__(self, continuous_dim: int, discrete_dim: int, action_dim: int, net_arch: list = [256, 256]):
        super().__init__()
        
        # Feature extractor: process continuous and discrete observations
        # SB3 uses separate feature extractors then concatenates
        self.continuous_feature_dim = continuous_dim
        self.discrete_feature_dim = discrete_dim
        
        # Build shared feature extractor layers
        # Input: continuous + discrete (one-hot encoded)
        input_dim = continuous_dim + discrete_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head: outputs mean and log_std for continuous actions
        self.latent_pi = prev_dim
        self.mean_actions = nn.Linear(self.latent_pi, action_dim)
        self.log_std = nn.Linear(self.latent_pi, action_dim)
        
    def forward(self, continuous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns deterministic action (mean).
        
        :param continuous: Continuous observation tensor
        :param discrete: Discrete observation tensor (one-hot encoded)
        :return: Action tensor
        """
        # Concatenate continuous and discrete features
        features = torch.cat([continuous, discrete], dim=-1)
        
        # Extract features through MLP
        latent = self.feature_extractor(features)
        
        # Get mean action
        mean = self.mean_actions(latent)
        
        # For deterministic actions, return mean (tanh squashed)
        action = torch.tanh(mean)
        
        return action
    
    def get_action_distribution(self, continuous: torch.Tensor, discrete: torch.Tensor):
        """
        Get action distribution (for stochastic sampling if needed).
        """
        features = torch.cat([continuous, discrete], dim=-1)
        latent = self.feature_extractor(features)
        
        mean = self.mean_actions(latent)
        log_std = self.log_std(latent)
        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std


class NativeSACActorAgent(Agent):
    """
    BBRL Agent using native PyTorch policy (no SB3 dependency).
    Writes actions to the 'action' key in workspace.
    """
    
    def __init__(self, policy_net: nn.Module, deterministic: bool = True, name: str = "native_sac_actor"):
        """
        :param policy_net: Native PyTorch policy network
        :param deterministic: Whether to use deterministic (mean) or stochastic actions
        :param name: Name of the agent
        """
        super().__init__(name=name)
        self.policy_net = policy_net
        self.deterministic = deterministic
        self.policy_net.eval()  # Set to evaluation mode
    
    def forward(self, t: int, **kwargs):
        """
        Forward pass that writes action into workspace.
        
        :param t: Time step (BBRL passes this as positional argument)
        """
        # Retrieve observations from workspace using self.get()
        obs_continuous = self.get(("env/env_obs/continuous", t))
        
        # Try to get discrete observation if it exists
        obs_discrete = None
        try:
            obs_discrete = self.get(("env/env_obs/discrete", t))
        except (KeyError, AttributeError):
            # Discrete observation not available - use zeros
            # Get shape from continuous observation
            if obs_continuous.ndim > 1:
                batch_size = obs_continuous.shape[0]
            else:
                batch_size = 1
            # SB3 uses one-hot encoding for discrete observations
            # MultiDiscrete([10, 7, 7, 7, 7, 7, 2, 4, 11]) = 9 dimensions
            # One-hot encoding: sum([10, 7, 7, 7, 7, 7, 2, 4, 11]) = 62 dimensions
            discrete_dim = 62  # Sum of MultiDiscrete nvec
            obs_discrete = torch.zeros((batch_size, discrete_dim), dtype=torch.float32, device=obs_continuous.device)
        
        # Handle batch dimensions
        if obs_continuous.ndim == 1:
            obs_continuous = obs_continuous.unsqueeze(0)
        if obs_discrete.ndim == 1:
            obs_discrete = obs_discrete.unsqueeze(0)
        
        # Convert discrete to one-hot if needed (if it's integer indices)
        if obs_discrete.dtype in (torch.int64, torch.int32, torch.long):
            # Convert MultiDiscrete indices to one-hot encoding
            # MultiDiscrete nvec: [10, 7, 7, 7, 7, 7, 2, 4, 11] = 9 dimensions
            # One-hot: sum([10, 7, 7, 7, 7, 7, 2, 4, 11]) = 62 dimensions
            nvec = [10, 7, 7, 7, 7, 7, 2, 4, 11]
            batch_size = obs_discrete.shape[0]
            one_hot_list = []
            for i, n in enumerate(nvec):
                # Get the i-th discrete value for all batches
                if obs_discrete.shape[1] == len(nvec):
                    discrete_val = obs_discrete[:, i]
                else:
                    # Assume it's already flattened one-hot
                    break
                # Create one-hot encoding
                one_hot = torch.zeros((batch_size, n), dtype=torch.float32, device=obs_discrete.device)
                one_hot.scatter_(1, discrete_val.long().unsqueeze(1), 1.0)
                one_hot_list.append(one_hot)
            if one_hot_list:
                # Concatenate all one-hot encodings
                obs_discrete = torch.cat(one_hot_list, dim=1)
            else:
                # Already in one-hot format or wrong shape - use as is
                obs_discrete = obs_discrete.float()
        else:
            # Already float (likely one-hot), ensure it's the right dtype
            obs_discrete = obs_discrete.float()
        
        # Get action from policy network
        with torch.no_grad():
            if self.deterministic:
                action = self.policy_net(obs_continuous, obs_discrete)
            else:
                mean, log_std = self.policy_net.get_action_distribution(obs_continuous, obs_discrete)
                std = torch.exp(log_std)
                # Sample from normal distribution
                action = torch.tanh(mean + std * torch.randn_like(mean))
        
        # Ensure action has batch dimension [1, action_dim]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        
        # Store action in workspace using self.set()
        self.set(("action", t), action)


def load_sb3_checkpoint(checkpoint_path: str):
    """
    Load SB3 checkpoint and extract policy weights and architecture info.
    Returns state_dict and metadata without requiring SB3 at runtime.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        # Try with .zip extension
        checkpoint_path = checkpoint_path.with_suffix('.zip')
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load the zip file
    with zipfile.ZipFile(checkpoint_path, 'r') as zip_file:
        # SB3 saves policy as 'policy.pth' which is a standard PyTorch state dict
        if 'policy.pth' not in zip_file.namelist():
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'policy.pth'. "
                           "This checkpoint format is not supported.")
        
        # Load policy state dict directly
        with zip_file.open('policy.pth', 'r') as f:
            policy_state_dict = torch.load(f, map_location='cpu')
        
        # Metadata is not strictly necessary - we can infer it from observation/action spaces
        metadata = {}
    
    return policy_state_dict, metadata


def convert_sb3_to_native_state_dict(policy_state_dict: dict) -> dict:
    """
    Convert SB3 policy state dict keys to NativeSACPolicy format.
    
    SB3 SAC uses: actor.latent_pi.* -> feature_extractor.*
                  actor.mu.* -> mean_actions.*
                  actor.log_std.* -> log_std.*
    """
    our_state_dict = {}
    for key, value in policy_state_dict.items():
        # Skip critic network weights - we only need the actor/policy
        if key.startswith('critic.'):
            continue
        
        new_key = key
        # Map SB3 keys to our keys
        if key.startswith('actor.latent_pi.'):
            # actor.latent_pi.0.weight -> feature_extractor.0.weight
            new_key = key.replace('actor.latent_pi.', 'feature_extractor.')
        elif key.startswith('actor.mu.'):
            # actor.mu.weight -> mean_actions.weight
            new_key = key.replace('actor.mu.', 'mean_actions.')
        elif key.startswith('actor.log_std.'):
            # actor.log_std.weight -> log_std.weight
            new_key = key.replace('actor.log_std.', 'log_std.')
        elif 'mlp_extractor.policy_net' in key:
            # Fallback for other SB3 versions
            new_key = key.replace('mlp_extractor.policy_net', 'feature_extractor')
        elif 'mlp_extractor.shared_net' in key:
            # Fallback for other SB3 versions
            new_key = key.replace('mlp_extractor.shared_net', 'feature_extractor')
        elif 'action_net' in key:
            # Fallback for other SB3 versions
            new_key = key.replace('action_net', 'mean_actions')
        
        our_state_dict[new_key] = value
    
    return our_state_dict


def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """Creates a new actor (BBRL agent) that writes into `action`

    :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
                  Can be:
                  1. A dict with converted policy weights (preferred - no path resolution needed)
                  2. A dict with 'checkpoint_path' pointing to SB3 .zip file
                  3. A path string to the checkpoint file
                  
                  Example (converted weights):
                  torch.save(converted_state_dict, "pystk_actor.pth")
                  
                  Example (checkpoint path):
                  torch.save({"checkpoint_path": "sac_stk-800000-fullobs"}, "pystk_actor.pth")
    :param observation_space: The environment observation space (with wrappers)
    :param action_space: The environment action space (with wrappers)
    :return: a BBRL agent
    """
    # If no state provided, return a random agent
    if state is None:
        return RandomActorAgent(action_space=action_space, name="random_actor")
    
    # Check if state is already converted weights (has keys like 'feature_extractor', 'mean_actions', etc.)
    if isinstance(state, dict):
        # Check if it looks like converted weights (has our key format)
        has_converted_keys = any(
            key.startswith(('feature_extractor.', 'mean_actions.', 'log_std.'))
            for key in state.keys()
        )
        
        if has_converted_keys:
            # This is already converted weights - use directly
            our_state_dict = state
            # Try to get metadata if available
            metadata = state.get('_metadata', {})
        elif "checkpoint_path" in state or "path" in state:
            # This is a checkpoint path - need to load and convert
            checkpoint_path = state.get("checkpoint_path") or state.get("path")
            our_state_dict = None  # Will be loaded below
            metadata = {}
        else:
            raise ValueError(
                "state dict must contain either:\n"
                "  1. Converted policy weights (keys like 'feature_extractor.*', 'mean_actions.*')\n"
                "  2. 'checkpoint_path' or 'path' key pointing to SB3 checkpoint\n"
                "Example: torch.save(converted_state_dict, 'pystk_actor.pth')"
            )
    elif isinstance(state, (str, Path)):
        # This is a checkpoint path string
        checkpoint_path = str(state)
        our_state_dict = None  # Will be loaded below
        metadata = {}
    else:
        raise ValueError(f"state must be str, Path, or dict, got {type(state)}")
    
    # If we need to load from checkpoint, do it now
    if our_state_dict is None:
        # Handle relative paths - look for checkpoint in the same directory as pystk_actor.pth
        if not Path(checkpoint_path).is_absolute():
            # The checkpoint should be in the same directory as this file (where pystk_actor.pth is)
            file_dir = Path(__file__).parent
            possible_paths = [
                file_dir / checkpoint_path,  # Same directory as pystk_actor.py
                file_dir / f"{checkpoint_path}.zip",
            ]
            
            # Also try parent directory (project root) as fallback
            parent_dir = file_dir.parent
            possible_paths.extend([
                parent_dir / checkpoint_path,
                parent_dir / f"{checkpoint_path}.zip",
            ])
            
            # Try each path until we find one that exists
            found = False
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    found = True
                    break
            
            if not found:
                # If still not found, raise a more helpful error
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Tried the following locations:\n" + 
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        
        try:
            # Load checkpoint and extract policy weights
            policy_state_dict, checkpoint_metadata = load_sb3_checkpoint(checkpoint_path)
            metadata.update(checkpoint_metadata)
            
            # Convert SB3 format to our format
            our_state_dict = convert_sb3_to_native_state_dict(policy_state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e
    
    try:
        # Get observation and action dimensions
        # Note: We don't store observation_space in metadata (to avoid weights_only issues),
        # so we always use the provided observation_space parameter
        obs_space = observation_space
        
        if isinstance(obs_space, gym.spaces.Dict):
            continuous_dim = obs_space['continuous'].shape[0]
            # Discrete: MultiDiscrete([10, 7, 7, 7, 7, 7, 2, 4, 11])
            # One-hot encoding: sum of all nvec
            discrete_nvec = obs_space['discrete'].nvec
            discrete_dim = int(discrete_nvec.sum())
        else:
            # Fallback dimensions
            continuous_dim = 92
            discrete_dim = 62
        
        action_dim = action_space.shape[0]
        
        # Extract net_arch from metadata or infer from state dict
        net_arch = metadata.get('net_arch', [256, 256])
        if isinstance(net_arch, dict):
            # SB3 uses dict format like {'pi': [256, 256], 'vf': [256, 256]}
            net_arch = net_arch.get('pi', [256, 256])
        
        # Create native policy network
        policy_net = NativeSACPolicy(
            continuous_dim=continuous_dim,
            discrete_dim=discrete_dim,
            action_dim=action_dim,
            net_arch=net_arch
        )
        
        # Load the converted state dict (strict=False to handle any mismatches)
        policy_net.load_state_dict(our_state_dict, strict=False)
        
        # Create BBRL agent with native policy
        agent = NativeSACActorAgent(
            policy_net=policy_net,
            deterministic=True,  # Use deterministic actions for evaluation
            name="native_sac_actor"
        )
        
        return agent
        
    except Exception as e:
        raise RuntimeError(f"Failed to create actor: {e}") from e
