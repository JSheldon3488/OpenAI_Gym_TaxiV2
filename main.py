from agents import *
from monitor import interact
from simulation import *
import gym

'''
sarsa_agent = Sarsa_Agent()
expected_sarsa_agent = Expected_Sarsa_Agent()
qlearning_agent = QLearning_Agent()
avg_rewards, best_avg_reward = interact(env, sarsa_agent)
avg_rewards, best_avg_reward = interact(env, expected_sarsa_agent)
avg_rewards, best_avg_reward = interact(env, qlearning_agent)
'''


if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    # Search for good learning rates (already done, look at results in images directory)
    '''
    learningRate_gridSearch(env, Sarsa_Agent, 1, np.arange(0.01, 0.25, 0.01))
    learningRate_gridSearch(env, Expected_Sarsa_Agent, 20, np.arange(0.01, 0.25, 0.01))
    learningRate_gridSearch(env, QLearning_Agent, 1, np.arange(0.01, 0.25, 0.01))
    '''

    # Search for good exploration strategy by using different epsilon rates (already done, look at results in images directory)
    ''' Short term experiement to difference in how fast models learn based on exploration rate
    epsilon_experiement(env, Sarsa_Agent, 2000, display=False)
    epsilon_experiement(env, Expected_Sarsa_Agent, 2000,display=False)
    epsilon_experiement(env, QLearning_Agent, 2000,display=False)
    '''

    ''' Long term experiement to see if exploration rate mattered in the long run'''
    print("Sarsa Agent Results")
    epsilon_experiement(env, Sarsa_Agent, 50000, display=False, save=False)
    print("Expected Sarsa Agent Results")
    epsilon_experiement(env, Expected_Sarsa_Agent, 50000,display=False, save=False)
    print("Q_Learning Agent Results")
    epsilon_experiement(env, QLearning_Agent, 50000,display=False, save=False)
