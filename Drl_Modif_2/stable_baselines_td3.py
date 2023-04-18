import gym
import numpy as np
from time import sleep

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines_env import UnityEnv


max_timesteps = 1024

env = UnityEnv('bluerov2_scenario.launch', 256)

sleep(20)

model = TD3(MlpPolicy, env)
model.learn(total_timesteps=max_timesteps, log_interval=10)
model.save("model_bluerov2_td3")


