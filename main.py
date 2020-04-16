from agents import *
from monitor import interact
from simulation import *
import gym


def learning_rate_grid_search(env, sarsa_sim: bool, expected_sarsa_sim: bool, q_learning_sim: bool):
    """ Run grid search simulations to find a good learning rate for different agents. Before running
    check images directory to see if simulations already ran and just use approx. best alpha from graphs.

    Params
    ======
    - env: the environment used to run the simulations in (from OpenAI Gym)
    - sarsa_sim: bool stating if you want to run this agents simulations or not
    - expected_sarsa_sim: bool stating if you want to run this agents simulations or not
    - q_learning_sim: bool stating if you want to run this agents simulations or not
    Returns
    =======
    - saves resulting graphs as .png files in the images directory for each model type
    """
    # Sarsa simulation to find best learning rate
    if (sarsa_sim):
        sarsa_alpha_grid, sarsa_average_rewards_per_alpha = sarsa_simulation(env)
        display_sim("Sarsa", sarsa_alpha_grid, sarsa_average_rewards_per_alpha)

    # Expected Sarsa simulation to find best learning rate
    if (expected_sarsa_sim):
        esarsa_alpha_grid, esarsa_average_rewards_per_alpha = expected_sarsa_simulation(env)
        display_sim("Expected_Sarsa", esarsa_alpha_grid, esarsa_average_rewards_per_alpha)

    # Q-Learning simulation to find best learning rate
    if (q_learning_sim):
        q_alpha_grid, q_average_rewards_per_alpha = qlearning_simulation(env)
        display_sim("Q_Learning", q_alpha_grid, q_average_rewards_per_alpha)


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
    learning_rate_grid_search(env, sarsa_sim=True, expected_sarsa_sim=True, q_learning_sim=True)