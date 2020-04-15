from agents import *
from monitor import interact
import matplotlib.pyplot as plt
import numpy as np


#Create simulation to find good alpha for Sarsa Agent
def sarsa_simulation(env):
    num_sims = 10;
    alpha_grid = np.arange(0.1,5.1,0.2)
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
    fig.savefig(f"images/{modelName}.png")