import datetime
import io
import uuid
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from policy import CustomCNN

import robosuite as suite  # isort:skip


class Wrapper(gym.Env):
    def __init__(self, env, callbacks=None, log_callback=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._log_callback = log_callback
        self._precision = precision
        self._episode = None
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        _obs, _reward, _done, _info = self._env.step(action)
        image = np.transpose(_obs["agentview_image"], (2, 0, 1))
        transition = {
            "image": image,
            "reward": _reward,
            "discount": _info.get("discount", 1 - float(_done)),
            "action": action,
        }
        self._episode.append(transition)
        episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
        episode = {k: self._convert(v) for k, v in episode.items()}
        if self._log_callback:
            self._log_callback(episode)
        if _done:
            _info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        return image, _reward, _done, _info

    def reset(self):
        _obs = self._env.reset()
        image = np.transpose(_obs["agentview_image"], (2, 0, 1))
        transition = {
            "image": image,
            "reward": 0.0,
            "discount": 1.0,
            "action": np.zeros(self.action_space.shape),
        }
        self._episode = [transition]
        return image

    def _convert(self, v):
        return np.array(v, dtype=np.float32)


def save_episode_to_dir(directory):
    def save_episode(episode):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%T%H%M%S")
        identifier = str(uuid.uuid4().hex)
        length = len(episode["reward"])
        filename = self.directory / f"{timestamp}-{identifier}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())

    return save_episode


def log_progress():
    def log_episode(episode):
        total_timesteps = len(episode["action"])
        if total_timesteps % 10 == 0:
            total_reward = sum(episode["reward"])
            done = episode["discount"][-1] == 0
            print(
                f"Total timesteps: {total_timesteps:<5} | Total reward: {total_reward:<10.2f} | Done: {done}"
            )

    return log_episode


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    directory = Path(directory).expanduser()
    random = np.random.RandomState(seed)
    cache = {}
    while True:
        for filename in directory.glob("*.npz"):
            if filename not in cache:
                try:
                    with filename.open("rb") as f:
                        episode = np.load(f)
                        episode = {
                            k: episode[k]
                            for k in [
                                "image",
                                "action",
                                "reward",
                                "discount",
                            ]
                        }
                except Exception as e:
                    print(f"Could not load episode: {e}")
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in random.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 1:
                    print(f"Skipped short episode of length {available}.")
                    continue
                if balance:
                    index = min(random.randint(0, total), available)
                else:
                    index = int(random.randint(0, available))
                episode = {
                    k: v[index : index + length] for k, v in episode.items()
                }
            yield episode


def count_episodes(directory):
    directory = Path(directory).exanduser()
    filenames = directory.glob("*.npz")
    lengths = [int(n.stem.rsplit("-", 1)[-1]) - 1 for n in filenames]
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps


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
demo_dir = Path("../demos").expanduser()
demo_dir.mkdir(parents=True, exist_ok=True)
callbacks = [save_episode_to_dir(demo_dir)]
env = Wrapper(env, callbacks, log_callback=log_progress())
env = DummyVecEnv([lambda: env])
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps, log_interval=4)
