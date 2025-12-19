from typing import List, Callable
import gymnasium as gym
import torch
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from bbrl.agents import Agent
from tensordict import TensorDict
import numpy as np

#: The base environment name (you can change that)
env_name = "supertuxkart/flattened_continuous_actions-v0"

#: Player name (you must change that)
player_name = "Tuxy"


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


class SB3ActorAgent(Agent):
    """
    BBRL Agent that wraps an SB3 policy.
    Writes actions to the 'action' key in workspace.
    This is NOT a temporal agent - it processes each step independently.
    """
    
    def __init__(self, sb3_policy: BasePolicy, deterministic: bool = True, name: str = "sb3_actor"):
        """
        :param sb3_policy: The SB3 policy to wrap
        :param deterministic: Whether to use deterministic (mode) or stochastic actions
        :param name: Name of the agent
        """
        super().__init__(name=name)
        self.sb3_policy = sb3_policy
        self.deterministic = deterministic
        self.sb3_policy.eval()  # Set to evaluation mode
    
    def forward(self, t: int, **kwargs):
        """
        Forward pass that writes action into workspace.
        
        :param t: Time step (BBRL passes this as positional argument)
        """
        # Retrieve observations from workspace using self.get()
        # BBRL/master-mind uses "env/env_obs/continuous" and "env/env_obs/discrete" keys
        obs_continuous = self.get(("env/env_obs/continuous", t))
        
        # Try to get discrete observation if it exists
        obs_discrete = None
        try:
            obs_discrete = self.get(("env/env_obs/discrete", t))
        except (KeyError, AttributeError):
            # Discrete observation not available
            pass
        
        # Convert observations to numpy and prepare for SB3
        # BBRL stores tensors with batch/time dimensions, we need to extract the right slice
        if obs_continuous.ndim > 1:
            # Remove batch/time dimensions - get the actual observation
            obs_continuous_np = obs_continuous.squeeze().cpu().numpy()
        else:
            obs_continuous_np = obs_continuous.cpu().numpy()
        
        # Ensure it's 1D
        if obs_continuous_np.ndim == 0:
            obs_continuous_np = obs_continuous_np[np.newaxis]
        elif obs_continuous_np.ndim > 1:
            obs_continuous_np = obs_continuous_np.flatten()
        
        # Construct observation dict for SB3
        obs_for_sb3 = {"continuous": obs_continuous_np}
        
        if obs_discrete is not None:
            if obs_discrete.ndim > 1:
                obs_discrete_np = obs_discrete.squeeze().cpu().numpy()
            else:
                obs_discrete_np = obs_discrete.cpu().numpy()
            
            # Ensure it's 1D
            if obs_discrete_np.ndim == 0:
                obs_discrete_np = obs_discrete_np[np.newaxis]
            elif obs_discrete_np.ndim > 1:
                obs_discrete_np = obs_discrete_np.flatten()
            
            obs_for_sb3["discrete"] = obs_discrete_np
        
        # Get action from SB3 policy
        with torch.no_grad():
            action, _ = self.sb3_policy.predict(obs_for_sb3, deterministic=self.deterministic)
        
        # Convert action to tensor
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        
        # Ensure action is 1D with shape (action_dim,)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim > 1:
            # Remove extra dimensions
            action_tensor = action_tensor.squeeze()
            if action_tensor.ndim == 0:
                action_tensor = action_tensor.unsqueeze(0)
        
        # BBRL stores tensors with batch dimensions to match observations
        # The observations have shape [1, obs_dim], so we should store action as [1, action_dim]
        # This ensures the wrapper can properly slice it
        if action_tensor.ndim == 1:
            # Add batch dimension to match observation format
            action_tensor = action_tensor.unsqueeze(0)  # Shape: [1, action_dim]
        
        # Store action in workspace using self.set()
        self.set(("action", t), action_tensor)


def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """Creates a new actor (BBRL agent) that writes into `action`

    :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
                  This should be a dict with 'checkpoint_path' pointing to SB3 .zip file
                  OR a path string to the checkpoint file.
                  Example: torch.save({"checkpoint_path": "sac_stk-800000-fullobs"}, "pystk_actor.pth")
    :param observation_space: The environment observation space (with wrappers)
    :param action_space: The environment action space (with wrappers)
    :return: a BBRL agent
    """
    # If no state provided, return a random agent
    if state is None:
        return RandomActorAgent(action_space=action_space, name="random_actor")
    
    # Determine checkpoint path from state
    if isinstance(state, (str, Path)):
        checkpoint_path = str(state)
    elif isinstance(state, dict):
        if "checkpoint_path" in state:
            checkpoint_path = state["checkpoint_path"]
        elif "path" in state:
            checkpoint_path = state["path"]
        else:
            raise ValueError("state dict must contain 'checkpoint_path' or 'path' key. "
                           "Example: torch.save({'checkpoint_path': 'sac_stk-800000-fullobs'}, 'pystk_actor.pth')")
    else:
        raise ValueError(f"state must be str, Path, or dict, got {type(state)}")
    
    # Handle relative paths - assume checkpoint is in the project root
    if not Path(checkpoint_path).is_absolute():
        # Try to find the checkpoint in common locations
        project_root = Path(__file__).parent.parent
        possible_paths = [
            project_root / checkpoint_path,
            project_root / f"{checkpoint_path}.zip",
        ]
        for path in possible_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
    
    # Create a dummy env to load the SB3 model (needed for proper initialization)
    from pystk2_gymnasium import AgentSpec
    dummy_env = gym.make(env_name, render_mode=None, agent=AgentSpec(use_ai=False))
    
    try:
        # Load SB3 model
        sb3_model = SAC.load(checkpoint_path, env=dummy_env)
        
        # Extract the policy
        sb3_policy = sb3_model.policy
        
        # Create BBRL agent wrapping the SB3 policy
        agent = SB3ActorAgent(
            sb3_policy=sb3_policy,
            deterministic=True,  # Use deterministic actions for evaluation
            name="sb3_actor"
        )
        
        return agent
        
    finally:
        dummy_env.close()
