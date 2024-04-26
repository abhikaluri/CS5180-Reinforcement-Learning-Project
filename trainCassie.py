# trainCassie.py

import myosuite
import gym
import gymnasium
from gymnasium import register
import SB3
import Cassie_v0

register('CassieWrapper-v0', entry_point= myosuite.envs.myo.myobase.SB3:CassieWrapper,
            max_episode_steps=1000,
        )


from stable_baselines3 import SAC
train_env = gym.make('CassieWrapper-v0') # Only for Stablebaselines-3 (gymnasium)
model = SAC('MLPolicy',train_env)
model.learn(timesteps=1000) # 5000000

