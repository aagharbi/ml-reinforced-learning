import os
import gym

import torch.cuda
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Make an environment for the agent to train in
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

#episodes = 5

#Read previous logs and trainings to base the next actions off the rewards
log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

#Prepare a model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

#Train the model x number of times
model.learn(total_timesteps=200000)

# Save Model
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path)

# Reloading model
del model
model = PPO.load(PPO_Path, env=env)

# Evaluation
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))