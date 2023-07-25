import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 256
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        num_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                num_input_channels, 32, kernel_size=8, stride=4, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            num_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(num_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
