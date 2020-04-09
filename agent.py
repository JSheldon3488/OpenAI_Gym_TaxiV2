import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1

        
    def select_action(self, state, episode_num):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #Update Epsilon based on the episode number
        self.epsilon /= episode_num
        
        #Set up epsilon greedy policy
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        policy[np.argmax(self.Q[state])] += 1-self.epsilon
        
        #Select action based on epsilon greedy policy
        return np.random.choice(np.arange(self.nA), p=policy)
    
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #Q-Learning off policy solution with constant learning rate 
        alpha = 0.1
        gamma = 1
        self.Q[state][action] = (1-alpha)*self.Q[state][action] + alpha*(reward + gamma*np.max(self.Q[next_state]))
        