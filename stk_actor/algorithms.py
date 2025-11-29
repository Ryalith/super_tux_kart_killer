# TODO: create SAC class
# Take an env and a config
# Create the actors and critics
# implement training loop:
# reset the env, store actions sampled from the actor into the memory
# train by sampling random batches from the replay memory
# backprop the SAC loss on the batch

# TODO: Look into Hybrid models than can handle a mix of discrete continuous actions

# TODO: add tensorboard loggers to check the status of training

import copy
from pathlib import Path

import gymnasium as gym
import torch
import torchrl
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .utils import (
    stk_action_tensor_to_dict,
    stk_state_dict_to_tensor,
)


class PPOAlgo:
    default_config = {
        "buffer_capactiy": 1024,
        "batch_size": 128,
        "discount_factor": 0.99,
        "critic_coef": 1.0,
        "gamma": ...,
        "lmbda": ...,
        "clip_epsilon": ...,
        "entropy_eps": ...,
        "loss_critic_type": "smooth_l1",
        "lr": ...,
        "max_grad_norm": ...,
    }

    def __init__(self, env, config: dict | None = None):
        """
        Instantiate a PPOAlgorithm to be used with discreet actions
        """
        self.env = env
        self.config = config if config is not None else self.default_config
        self.v_critic = ...
        self.old_v_critic = copy.deepcopy(self.v_critic)
        self.kl_agent = ...
        self.ppo_actor = ...
        self.old_ppo_actor = copy.deepcopy(self.ppo_actor)
        self.replay_buffer = ReplayBuffer(self.config.buffer_capacity)

    def _train_one_step(
        self,
        tensordict_data,
        advantage_module,
        loss_module,
        optim,
        replay_buffer,
        frames_per_batch,
        sub_batch_size,
        device,
    ):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            optim.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), self.config.max_grad_norm
            )
            optim.step()

    def train(self, total_frames, frames_per_batch, sub_batch_size, device):
        # TODO: check better collectors for parallel execution
        collector = SyncDataCollector(
            self.env,
            self.ppo_actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.lmbda,
            value_network=self.v_critic,
            average_gae=True,
            device=device,
        )

        loss_module = ClipPPOLoss(
            actor_network=self.ppo_actor,
            critic_network=self.v_critic,
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=bool(self.config.entropy_eps),
            entropy_coef=self.config.entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=self.config.critic_coef,
            loss_critic_type=self.config.loss_critic_type,  # "smooth_l1",
        )

        optim = torch.optim.Adam(loss_module.parameters(), self.config.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )

        for i, tensordict_data in enumerate(collector):
            loss_val = self._train_one_step(
                tensordict_data,
                advantage_module,
                loss_module,
                optim,
                replay_buffer,
                frames_per_batch,
                sub_batch_size,
                device,
            )
            scheduler.step()

    def load_checkpoint(self, checkpoint_path: Path):
        pass


class SACAlgo:
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.actor = ...
        self.q_critic_1 = ...
        self.q_critic_2 = ...
        self.q_tgt_critic_1 = ...
        self.q_tgt_critic_2 = ...
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

    def _fill_replay_buffer(self, max_iter: int):
        ix = 0
        done = False
        state, *_ = self.env.reset()

        while not done:
            ix += 1
            state_tensor = stk_state_dict_to_tensor(state)
            action_tensor = self.actor(state_tensor)
            action = stk_action_tensor_to_dict(action_tensor)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state_tensor = stk_state_dict_to_tensor(next_state)
            done = truncated or terminated
            self.replay_buffer.push(
                Transition(
                    state_tensor,
                    action_tensor,
                    next_state_tensor,
                    torch.tensor(reward),
                    torch.tensor(terminated),
                    torch.tensor(truncated),
                )
            )
            state = next_state
            if ix >= max_iter:
                break

    def _train_on_replay_buffer(self):
        ...
        # TODO: define SAC loss compute it and back propagate the loss
        # There might be a need to change the way the replay buffer is implemented to avoid resampling
        # Maybe define the deplay buffer as a dataset and create a data loader ?

    def train(self, epochs):
        for epoch in range(epochs):
            self._fill_replay_buffer(...)
            self._train_on_replay_buffer(...)
