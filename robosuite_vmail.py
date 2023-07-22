import os
import logging

import gym
import numpy as np
import torch
import torch.nn as nn
from gym import Wrapper
from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

import robosuite as suite
from robosuite.wrappers import GymWrapper

logging.basicConfig(filename="logs.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s")

class GymWrapper(gym.Env):
	def __init__(self, env, action_space):
		super().__init__()
		self.env = env
		self.action_space = action_space
		self.observation_space = Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
	
	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return np.transpose(obs["agentview_image"], (2, 0, 1)), reward, done, info

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
			nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
			nn.Flatten(),
		)

		with torch.no_grad():
			n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

		self.linear = nn.Sequential(nn.Linear(n_flatten, feature_dim), nn.ReLU())

	def forward(self, observations):
		return self.linear(self.cnn(observations))


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
        features_extractor_kwargs=dict(feature_dim=512)
)
model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=1e6)

num_iterations = 1000
num_rollouts = 100
num_timesteps = 100

data_dir = "saved_data"
os.makedirs(data_dir, exist_ok=True)

for iteration in range(num_iterations):
	for rollout_idx in range(num_rollouts):
		obs = env.reset()
		rollout_rewards = []
		rollout_actions = []
		rollout_discounts = []
		rollout_images = []
		rollout_states = []
		for t in range(num_timesteps):
			action, _states = model.predict(obs)
			next_obs, reward, done, info = env.step(action)
			rollout_rewards.append(reward)
			rollout_actions.append(action)
			rollout_discounts.append(not done)
			rollout_images.append(obs["image"])
			rollout_states.append(obs)
			
			obs = next_obs
			
			if done:
				logging.info(f"Iteration: {iteration}, Rollout: {rollout_idx}, Step: {t}, Done: {done}")
				break

		filename = os.path.join(data_dir, f"iteration_{iteration}_rollout_{rollout_idx}.npz")
		np.savez(filename, reward=rollout_reward, action=rollout_actions, discount=rollout_discounts, image=rollout_images, states=rollout_states)
