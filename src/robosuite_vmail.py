import logging
import os
from pathlib import Path

import gym
import numpy as np
import robosuite as suite
import torch
import torch.nn as nn
from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("logs.log")
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
console_format = logging.Formatter(format_str)
file_format = logging.Formatter(format_str)
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class GymWrapper(gym.Env):
    def __init__(self, _env, action_space):
        super().__init__()
        self.env = _env
        self.action_space = action_space
        self.observation_space = Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

    def step(self, _action):
        _obs, _reward, _done, _info = self.env.step(_action)
        return (
            np.transpose(_obs["agentview_image"], (2, 0, 1)),
            _reward,
            _done,
            _info,
        )

    def reset(self):
        return np.transpose(self.env.reset()["agentview_image"], (2, 0, 1))

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


class EpsilonGreedyWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        action_space,
        start_eps=1.0,
        end_eps=0.01,
        decay_rate=0.001,
        decay_steps=10000,
    ):
        super().__init__(env)
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step_count = 0
        self._action_space = action_space
        self._unwrapped = env

    @property
    def action_space(self):
        return self._action_space

    @property
    def unwrapped(self):
        return self._unwrapped

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if np.random.rand() < self._current_epsilon():
            action_space = self.action_space
            if isinstance(self.action_space, Box):
                action = self.action_space.sample()
            elif isinstance(self.action_space, gym.spaces.Discrete):
                action = self.action_space.sample()
        return self.env.step(action)

    def _current_epsilon(self):
        return self.end_eps + (self.start_eps - self.end_eps) * np.exp(
            -1.0 * self.step_count / self.decay_steps
        )


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=128):
        super(CustomCNN, self).__init__(observation_space, feature_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels, 32, kernel_size=8, stride=4, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim), nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


logger.info("Creating environment...")
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=True,
    use_touch_obs=False,
    camera_heights=84,
    camera_widths=84,
    reward_shaping=True,
    control_freq=20,
)
action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
env = EpsilonGreedyWrapper(
    env,
    action_space,
    start_eps=1.0,
    end_eps=0.01,
    decay_rate=0.001,
    decay_steps=10000,
)
env = DummyVecEnv([lambda: GymWrapper(env, action_space)])

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(feature_dim=128),
)

logger.info("Creating model...")
model = SAC("CnnPolicy", env, buffer_size=int(1e6), policy_kwargs=policy_kwargs, verbose=1)

logger.info("Starting model learning...")
total_timesteps = int(6e6)
current_dir = Path(__file__).resolve().parent
log_path = current_dir / "../logs"
tmp_path = log_path / "sac_vmail"
new_logger = configure(str(tmp_path), ["stdout", "tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=total_timesteps, log_interval=4)

num_iterations = 1000
num_rollouts = 100
num_timesteps = 100

data_dir = "saved_data"
os.makedirs(data_dir, exist_ok=True)

for iteration in range(num_iterations):
    for rollout_idx in range(num_rollouts):
        obs = env.reset()
        episode_reward = 0
        rollout_rewards = []
        rollout_actions = []
        rollout_discounts = []
        rollout_images = []
        rollout_states = []
        for t in range(num_timesteps):
            action, _states = model.predict(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            logger.info(
                f"Iteration: {iteration}, Rollout: {rollout_idx}, Step: {t}, Reward: {reward[0]}"
            )
            rollout_rewards.append(reward)
            rollout_actions.append(action)
            rollout_discounts.append(not done)
            rollout_images.append(obs[0])
            rollout_states.append(obs)

            obs = next_obs

            if done:
                break

        episode_reward = sum(rollout_rewards)
        filename = os.path.join(
            data_dir, f"iteration_{iteration}_rollout_{rollout_idx}.npz"
        )
        np.savez(
            filename,
            reward=rollout_rewards,
            action=rollout_actions,
            discount=rollout_discounts,
            image=rollout_images,
            states=rollout_states,
        )
