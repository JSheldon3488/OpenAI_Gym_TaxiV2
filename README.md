# OpenAI_Gym_TaxiV3
OpenAI Gym Taxi problem from Deep Reinforcement Learning Udacity Nanodegree.

## Problem Statement

There are four locations (R,G,Y,B) and the job of the agent is to pick up a passenger at one location and drop him off in the correct location (just like a taxi would). The passenger can be in one of those 4 locations or in the taxi and the destination is always at one of those four locations. The agent receives +20 for a successful dropoff and loses 1 point for every action it takes that is not a terminal action. There is also a 10 point penalty for illegal pick-up and drop-off actions. There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.

**State Space:** (taxi_row, taxi_col, passenger_location, destination)

**Action Space:** 
    0: move south,
    1: move north,
    2: move east,
    3: move west,
    4: pickup passenger,
    5: dropoff passenger

## Agents

## Hyperparameters
### Learning Rate (alpha)
In an attempt to find the best learning rate for the different agents I ran a grid search simulation for each agent. I ran 20 simulations for learning rates in the range of 0.01-0.25 incremented by 0.01 each time and took the average of the results. The results are below. It appears that learning rates between 0.05 and 0.25 have similar results (while holding exploration rate constant).

<p align="center">
    <img src="/images/Sarsa.png" width="400" height="360">
    <img src="/images/Expected_Sarsa.png" width="400" height="360">
</p>
<p align="center">
    <img src="/images/Q_Learning.png" width="480" height="360">
</p>

## Dependencies
  * [OpenAI Gym](https://gym.openai.com/) 
  * [Matplotlib](https://matplotlib.org/)
  * [Numpy](https://numpy.org/)
  
