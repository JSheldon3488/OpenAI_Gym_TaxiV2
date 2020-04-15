from agents import *
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
sarsa_agent = Sarsa_Agent()
expected_sarsa_agent = Expected_Sarsa_Agent()
qlearning_agent = QLearning_Agent()
avg_rewards, best_avg_reward = interact(env, sarsa_agent)
avg_rewards, best_avg_reward = interact(env, expected_sarsa_agent)
avg_rewards, best_avg_reward = interact(env, qlearning_agent)