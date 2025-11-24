from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect

# Note the use of relative imports
from .actors import Actor
from .pystk_actor import env_name, get_wrappers, player_name

if __name__ == "__main__":
    # Setup the environment
    SAC_config = ...
    SAC_algo = ...

    # Learn

    # (3) Save the actor sate
    torch.save(SAC_algo.actor.state_dict(), ... / "pystk_actor.pth")
