import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import SAC
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

class RenderCallback(BaseCallback):
    """
    Periodically run one evaluation episode with rendering enabled
    to visualize the current policy without slowing down all of training.
    """

    def __init__(self, render_every_n_episodes: int = 5, start_after_n_episodes: int = 70, verbose: int = 0):
        super().__init__(verbose)
        self.render_every_n_episodes = render_every_n_episodes
        self._episodes = 0
        self.start_after_n_episodes = start_after_n_episodes

    def _on_step(self) -> bool:
        # Count completed episodes from the training env
        dones = self.locals.get("dones")
        if dones is not None and dones.any():
            self._episodes += int(dones.sum().item())
            print(f"Episode: {self._episodes}")

            if self._episodes < self.start_after_n_episodes:
                return True

            if self._episodes % self.render_every_n_episodes == 0:
                # Separate env with rendering for visualization.
                # Match the training environment exactly (no wrappers currently):
                eval_env = gym.make(
                    "supertuxkart/flattened_continuous_actions-v0",
                    render_mode="human",
                    agent=AgentSpec(use_ai=False),
                )
                # Apply the same wrappers as training (currently none are active)
                # If you uncomment wrappers in training, uncomment them here too:
                # keep_indices = (...)
                # eval_env = ContinuousSubsetWrapper(eval_env, keep_indices=keep_indices)
                # eval_env = ContinuousOnlyWrapper(eval_env)
                
                total_reward = 0
                obs, info = eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    print(f"reward: {reward}, total: {total_reward}, info: {info}")

                eval_env.close()

        return True


class RewardClipWrapper(gym.RewardWrapper):
    """
    Clip rewards to prevent exploit bugs from dominating the learning signal.
    Also logs episodes with extreme rewards for debugging.
    """
    
    def __init__(
        self,
        env: gym.Env,
        finish_reward_bonus: float = 100.0,
        log_extreme: bool = True,
    ):
        super().__init__(env)
        self.finish_reward_bonus = finish_reward_bonus
        self.log_extreme = log_extreme
    
    def reset(self, **kwargs):           
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _is_race_finished(self, terminated=False):
        """
        Check if the race has finished by checking if episode was terminated.
        Returns (is_finished: bool, reason: str)
        """
        if terminated:
            return True, "Episode terminated"
        return False, "No finish detected"

    def step(self, action):
        """
        Boost reward magnitude when the race finishes.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if race finished (episode terminated) to boost reward
        race_finished, finish_reason = self._is_race_finished(terminated)

        # Add bonus reward when race finishes (episode terminated)
        if race_finished:
            original_reward = reward
            reward = reward + self.finish_reward_bonus
            if self.log_extreme:
                print(
                    f"Race finished! Reason: {finish_reason}. "
                    f"Reward boosted from {original_reward:.2f} to {reward:.2f} "
                    f"(added bonus: {self.finish_reward_bonus})"
                )

        return obs, reward, terminated, truncated, info


class ContinuousSubsetWrapper(gym.ObservationWrapper):
    """
    Keep only a subset of entries from obs['continuous'] while leaving
    the rest of the Dict observation unchanged.
    """

    def __init__(self, env: gym.Env, keep_indices: list[int]):
        super().__init__(env)
        self.keep_indices = list(keep_indices)

        old_box = self.observation_space["continuous"]
        low = old_box.low[self.keep_indices]
        high = old_box.high[self.keep_indices]
        new_box = gym.spaces.Box(low=low, high=high, dtype=old_box.dtype)

        spaces = dict(self.observation_space.spaces)
        spaces["continuous"] = new_box
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        obs = dict(obs)
        obs["continuous"] = obs["continuous"][self.keep_indices]
        return obs


# Use a larger MLP for the shared + policy/value networks
policy_kwargs = dict(
    net_arch=[256, 256],  # change capacity compared to the default
)

class ContinuousOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.observation_space["continuous"]

    def observation(self, obs):
        # Return only the continuous vector
        return obs["continuous"]

if __name__ == "__main__":
    # Simplest setup: plain gymnasium env, no DummyVecEnv / VecNormalize wrappers.
    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0",
        render_mode=None,  # no rendering during training
        agent=AgentSpec(use_ai=False),
    )

    # Keep only a subset of the 92-dim continuous observation:
    # center_path, center_path_distance, distance_down_track, front,
    # paths_distance, paths_end, paths_start, paths_width, velocity.
    keep_indices = (
        [
            2, 3, 4,  # center_path[0:3]
            5,        # center_path_distance[0]
            6,        # distance_down_track[0]
            8, 9, 10  # front[0:3]
        ]
        + list(range(42, 52))   # paths_distance[0:5, 0:2]
        + list(range(52, 67))   # paths_end[0:5, 0:3]
        + list(range(67, 82))   # paths_start[0:5, 0:3]
        + list(range(82, 87))   # paths_width[0:5, 0:1]
        + [89, 90, 91]          # velocity[0:3]
    )
    # env = ContinuousSubsetWrapper(env, keep_indices=keep_indices)

    # Drop the discrete part: make the observation a pure Box by using only 'continuous'
    

    # env = ContinuousOnlyWrapper(env)

    # Add 100 bonus reward when race finishes to strongly encourage completing races
    env = RewardClipWrapper(env, finish_reward_bonus=300.0, log_extreme=True)

    # Increase exploration by targeting higher policy entropy
    action_dim = env.action_space.shape[0]
    target_entropy = -2 * action_dim  # default is around -action_dim; this is more exploratory

    # model = SAC(
    #     # "MlpPolicy",
    #     "MultiInputPolicy",
    #     env,
    #     learning_rate=3e-4,
    #     buffer_size=1_000_000,
    #     learning_starts=100_000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=1,
    #     gradient_steps=2,
    #     ent_coef="auto",
    #     target_entropy=target_entropy,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     tensorboard_log=f"/home/gael/Documents/MS2A/4_RL/super_tux_kart_killer/runs/{current_time}-SAC",
    # )

    model = SAC.load("sac_stk-600000-fullobs", env=env)

    # Train longer to allow the agent to discover good driving behavior
    render_callback = RenderCallback(render_every_n_episodes=10, start_after_n_episodes=70)
    model.learn(total_timesteps=200_000, log_interval=1,
     callback=render_callback
     )

    print("Saving")
    model.save("sac_stk-700000-fullobs")

    del model  # free resources

    env.close()
# print("Loading")    
# model = SAC.load("sac_pendulum")

# print("Testing")
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()


