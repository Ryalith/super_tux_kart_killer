from functools import partial

# Note the use of relative imports
from .algorithms import SACAlgoSB
from .env import make_cont_env

if __name__ == "__main__":
    # Setup the environment
    train_env_fn = make_cont_env

    # Learn
    model = SACAlgoSB(train_env_fn)
