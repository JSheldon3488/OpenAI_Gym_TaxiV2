from agents import *
from monitor import interact
import matplotlib.pyplot as plt
import numpy as np


def learningRate_gridSearch(env, Agent, num_sims, alpha_grid):
    """ Run grid search simulations to find a good learning rate for different agents. Before running
    check images directory to see if simulations already ran and just use approx. best alpha from graphs.

    Params
    ======
    - env: the environment used to run the simulations in (from OpenAI Gym)
    - Agent: agent used in the simulation (sarsa, expected_sarsa, Q_learning)
    - num_sims: number of simulations per alpha (used to get an average)
    - alpha_grid: array of all the alpha values you want to run the simulation on

    Returns
    =======
    - saves resulting graphs as .png files in the images directory for each model type
    """
    avg_rewards_per_alpha = []
    for alpha in alpha_grid:
        sim_results = np.empty(num_sims)
        for sim in range(num_sims):
            agent = Agent(alpha)
            # Only need best_avg_reward from each individual simulation
            _, best_avg_reward = interact(env, agent)
            sim_results[sim] = best_avg_reward
        avg_rewards_per_alpha.append(np.mean(sim_results))

    # Create and save graph with the results from the simulation
    display_sim(Agent, alpha_grid, avg_rewards_per_alpha)


def display_sim(modelName, grid, results, display = True, save = True):
    """ Displays and/or saves the results of the simulations

    Params
    ======
    - modelName: The Agent Class used in the simulation
    - grid: the numpy array of values (x axis)
    - results: the numpy array of simulation results (y axis)
    - display: bool for whether or not to show the graph (default True)
    - save: bool for whether or not to save the graphs (default True)

    Returns
    =======
    - potentially shows and ssaves resulting graphs as .png files in the images directory for each model type
    """
    # Turn modelName from Class type to string for graphing
    if modelName == Sarsa_Agent:
        modelName = "Sarsa"
    elif modelName == Expected_Sarsa_Agent:
        modelName = "Expected_Sarsa"
    elif modelName == QLearning_Agent:
        modelName = "Q_Learning"

    #Set up plots
    fig, ax = plt.subplots()
    ax.plot(grid, results)
    ax.set(xlabel = "alpha", ylabel = "Average Reward", title = f"Finding best alpha for {modelName}")
    ax.grid()

    #Graph max point
    ymax = np.max(results)
    xmax = grid[np.argmax(results)]
    text = "alpha={:.2f}, reward={:.3f}".format(xmax, ymax)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
    #Set y lim so the annotation looks good
    ax.set_ylim(np.min(results), np.max(results)+0.1)

    if save:
        filepath = f"C:\Dev\Python\RL\Taxi_Problem\images\\{modelName}.png"
        fig.savefig(filepath)

    if display:
        plt.show()


def epsilon_experiement(env, Agent, num_episodes, display=True, save=True,):
    """ Try out different epsilon (exploration rate) strategies and graph the

    Params
    ======
    - env: the environment used to run the simulations in (from OpenAI Gym)
    - Agent: agent used in the simulation (sarsa, expected_sarsa, Q_learning)
    - num_episodes: the number of episodes each experiment will run
    Returns
    =======
    - saves resulting graphs as .png files in the images directory for each model type
    """
    # Set up Graph for plotting results
    fig, ax = plt.subplots()
    if Agent == Sarsa_Agent:
        modelName = "Sarsa"
    elif Agent == Expected_Sarsa_Agent:
        modelName = "Expected_Sarsa"
    elif Agent == QLearning_Agent:
        modelName = "Q_Learning"
    ax.set(xlabel = "Episode Number", ylabel = "Reward", title = f"Exploration Strategy Comparison for {modelName} Agent")
    ax.grid()

    #Run the experiements
    # Epsilon decay at rate = 1/num_episode (after 100 episodes only 1% exploration)
    agent = Agent(0.1, decay_by_episode=True)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes)
    ax.plot(range(10,len(avg_rewards)), list(avg_rewards)[10:], label="low_exploration")

    # Epsilon decay rate = max(0.01, old_epsilon*0.999) (capped minimum exploration)
    agent = Agent(0.1, decay_capped_min=True)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes)
    ax.plot(range(10,len(avg_rewards)), list(avg_rewards)[10:], label="medium_exploration")

    # Epsilon decay rate = old_epsilon*0.9999 (lot of exploring)
    agent = Agent(0.1)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes)
    ax.plot(range(10,len(avg_rewards)), list(avg_rewards)[10:], label="high_exploration")

    #Save and/or show the results
    ax.legend()
    if save:
        filepath = f"C:\Dev\Python\RL\Taxi_Problem\images\\{modelName}_epsilon_experiment.png"
        fig.savefig(filepath)

    if display:
        plt.show()