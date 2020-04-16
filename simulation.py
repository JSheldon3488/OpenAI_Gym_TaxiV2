from agents import *
from monitor import interact
import matplotlib.pyplot as plt
import numpy as np
import os


#Create simulation to find good alpha for Sarsa Agent
def sarsa_simulation(env):
    num_sims = 20;
    alpha_grid = np.arange(0.01,0.25,0.01)
    average_rewards_per_alpha = []
    
    for alpha in alpha_grid:
        print(alpha)
        sim_results = np.empty(num_sims)
        for sim in range(num_sims):
            #Create agent
            sarsa_agent = Sarsa_Agent(alpha)
            #run simulation and save result (only need best average reward)
            _, best_avg_reward = interact(env, sarsa_agent)
            sim_results[sim] = best_avg_reward
        
        #End of sim for this alpha, average results and save
        average_rewards_per_alpha.append(np.mean(sim_results))
        
    
    return alpha_grid, np.asarray(average_rewards_per_alpha)
            
            


#Create simulation to find good alpha for Expected Sarsa Agent

#Create simulation to find good alpha for QLearning Agent


#Function to display results of simulations
def display_sim(modelName, grid, results):
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

    #Save graph
    filepath = f"C:\Dev\Python\RL\Taxi_Problem\images\\{modelName}.png"
    fig.savefig(filepath)
    #Look at result
    plt.show()
