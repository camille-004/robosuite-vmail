import gym
import numpy as np
from gym.spaces import Box


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
