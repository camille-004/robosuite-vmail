import logging
import os

import gym
import numpy as np
import robosuite as suite
import torch
import torch.nn as nn
from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

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


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super(ProgressCallback, self).__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)
    
    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.update(0)
    
    def _on_training_end(self):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class GymWrapper(gym.Env):
    def __init__(self, _env, _action_space):
        super().__init__()
        self.env = _env
        self.action_space = _action_space
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


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=512):
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
    camera_heights=84,
    camera_widths=84,
    reward_shaping=True,
    control_freq=20,
)
action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
env = DummyVecEnv([lambda: GymWrapper(env, action_space)])

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(feature_dim=512),
)

logger.info("Creating model...")
model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

logger.info("Starting model learning...")
total_timesteps = int(1e6)
callback = ProgressCallback(total_timesteps)
model.learn(total_timesteps=total_timesteps, callback=callback)

num_iterations = 1000
num_rollouts = 100
num_timesteps = 100

data_dir = "saved_data"
os.makedirs(data_dir, exist_ok=True)

logger.info("Starting iterations...")
for iteration in range(num_iterations):
    logger.info(f"Starting iteration {iteration}...")
    for rollout_idx in range(num_rollouts):
        logger.info(f"Starting rollout {rollout_idx}...")
        obs = env.reset()
        rollout_rewards = []
        rollout_actions = []
        rollout_discounts = []
        rollout_images = []
        rollout_states = []
        for t in range(num_timesteps):
            logger.info(f"Observation shape: {obs.shape}")
            action, _states = model.predict(obs)
            logger.info(f"Action: {action}")
            next_obs, reward, done, info = env.step(action)
            logger.info(f"Reward: {reward}, Done: {done}")
            rollout_rewards.append(reward)
            rollout_actions.append(action)
            rollout_discounts.append(not done)
            rollout_images.append(obs["agentview_image"])
            rollout_states.append(obs)

            obs = next_obs

            if done:
                logging.info(
                    f"Iteration: {iteration}, Rollout: {rollout_idx}, Step: {t}, Done: {done}"
                )
                break

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
