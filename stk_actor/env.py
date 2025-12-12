# TODO: create a wrapper that updates the specs of the pystk2 env for use with torchrl
# see https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html
# in the 'Define an environment' section
from torchrl.envs import GymEnv
import pystk2_gymnasium


def get_observation_vector_dim(env_specs):
    """
    Calculate the dimension of the flattened observation vector after processing.
    - Continuous observations: kept as-is
    - Discrete observations (MultiOneHot): already one-hot encoded, kept as-is

    Args:
        env_spec: Environment spec (from env.specs)

    Returns:
        int: Total dimension of the processed observation vector
    """
    obs_spec = env_specs["output_spec"]["full_observation_spec"]
    total_dim = 0

    # Handle composite observation spaces
    for key in obs_spec.keys():
        spec = obs_spec[key]
        # Each component contributes its flattened shape
        total_dim += spec.shape.numel()

    return total_dim


# TODO: check if this is redundant
# It seems multi one hot takes care of these automatically
# might not need to split outputdims
# instead concat all outputs ?
def get_action_vector_dims(env_specs):
    """
    Calculate the dimension of the multiple actions of the env.
    env actions are assumed to be MultiDiscrete, meaning action is multionehot

    Args:
        env_spec: Environment spec (from env.specs)

    Returns:
        Sequence[int]: dimensions of the actions vectors
    """
    # Handle if passed full env spec or just observation spec
    act_spec = env_specs["input_spec"]["full_action_spec"]["action"]

    return [box.n for box in act_spec.space]


def make_discrete_env(render_mode = None, use_ai = False):
    env = GymEnv(
        "supertuxkart/flattened_multidiscrete-v0",
        render_mode=render_mode,
        agent=pystk2_gymnasium.AgentSpec(use_ai=use_ai, name="STKKillerAI"),
    )
    return env
