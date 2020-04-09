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
        self.epsilon_decay_rate = 0.9999
        self.q_learning_alpha = 0.01
        self.expected_sarsa_alpha = 1
        self.gamma = 1

        
    def select_action(self, state, episode_num, num_episodes):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #Update Epsilon (two different options)
        #self.epsilon *= self.epsilon_decay_rate
        self.epsilon /= num_episodes
        
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
        #Two options below Q-Learning and Expected Sarsa
        
        #Q-Learning off policy solution with constant learning rate
        #self.Q[state][action] = (1-self.q_learning_alpha)*self.Q[state][action] + self.q_learning_alpha*(reward + self.gamma*np.max(self.Q[next_state]))
        
        #Expected Sars on policy solution with constant learning rate
        next_state_policy = np.ones(self.nA)*(self.epsilon/self.nA)
        next_state_policy[np.argmax(self.Q[next_state])] += 1-self.epsilon
        self.Q[state][action] = (1-self.expected_sarsa_alpha)*self.Q[state][action] + self.expected_sarsa_alpha*(reward + self.gamma*np.dot(self.Q[next_state], next_state_policy))
        
        