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


#Function to display results of simulations
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
