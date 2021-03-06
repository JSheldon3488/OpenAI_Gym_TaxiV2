import numpy as np
from collections import defaultdict

class Sarsa_Agent:
    def __init__(self, alpha, nA=6, decay_by_episode = False, decay_capped_min = False):
        """ Initialize agent.

        Params
        ======
        - alpha: used to set the update learning rate
        - nA: number of actions available to the agent
        - decay_by_episode: bool used to specify exploration strategy
        - decay_capped_min: bool used to specify exploration strategy
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.epsilon_decay_rate = 0.9999
        self.decay_by_episode = decay_by_episode
        self.decay_capped_min = decay_capped_min
        self.alpha = alpha
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
        #Update Epsilon (based on different epsilon strategies)
        if (self.decay_by_episode):
            self.epsilon = 1/episode_num
        elif (self.decay_capped_min):
            self.epsilon = max(self.epsilon*0.999, 0.01)
        else:
            self.epsilon *= self.epsilon_decay_rate
        
        #Set up epsilon greedy policy
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        policy[np.argmax(self.Q[state])] += 1-self.epsilon
        
        #Select action based on epsilon greedy policy
        return np.random.choice(np.arange(self.nA), p=policy)
    
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using Sarsa and the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #Sarsa solution with constant learning rate; For Sarsa we need to choose a next action for a given policy
        #Get a next action
        next_state_policy = np.ones(self.nA)*(self.epsilon/self.nA)
        next_state_policy[np.argmax(self.Q[next_state])] += 1-self.epsilon
        next_action = np.random.choice(np.arange(self.nA), p=next_state_policy)
        #Update Q Table
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*self.Q[next_state][next_action])
        
        
        
class Expected_Sarsa_Agent:
    def __init__(self, alpha, nA=6, decay_by_episode = False, decay_capped_min = False):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.epsilon_decay_rate = 0.9999
        self.decay_by_episode = decay_by_episode
        self.decay_capped_min = decay_capped_min
        self.alpha = alpha
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
        #Update Epsilon (based on different epsilon strategies)
        if (self.decay_by_episode):
            self.epsilon = 1/episode_num
        elif (self.decay_capped_min):
            self.epsilon = max(self.epsilon*0.999, 0.01)
        else:
            self.epsilon *= self.epsilon_decay_rate
        
        #Set up epsilon greedy policy
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        policy[np.argmax(self.Q[state])] += 1-self.epsilon
        
        #Select action based on epsilon greedy policy
        return np.random.choice(np.arange(self.nA), p=policy)
    
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using Sarsa and the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #Expected Sarsa solution with constant learning rate
        #Epsilon greedy policy for this update
        next_state_policy = np.ones(self.nA)*(self.epsilon/self.nA)
        next_state_policy[np.argmax(self.Q[next_state])] += 1-self.epsilon
        #Update Q Table using expected value of future state given epsilon greedy policy
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*np.dot(self.Q[next_state], next_state_policy))
        
        
        
        
class QLearning_Agent:
    def __init__(self, alpha, nA=6, decay_by_episode = False, decay_capped_min = False):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.epsilon_decay_rate = 0.9999
        self.decay_by_episode = decay_by_episode
        self.decay_capped_min = decay_capped_min
        self.alpha = alpha
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
        #Update Epsilon (based on different epsilon strategies)
        if (self.decay_by_episode):
            self.epsilon = 1/episode_num
        elif (self.decay_capped_min):
            self.epsilon = max(self.epsilon*0.999, 0.01)
        else:
            self.epsilon *= self.epsilon_decay_rate
        
        #Set up epsilon greedy policy
        policy = np.ones(self.nA)*(self.epsilon/self.nA)
        policy[np.argmax(self.Q[state])] += 1-self.epsilon
        
        #Select action based on epsilon greedy policy
        return np.random.choice(np.arange(self.nA), p=policy)
    
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using Sarsa and the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #Q-Learning off policy solution with constant learning rate
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + self.alpha*(reward + self.gamma*np.max(self.Q[next_state]))
        
        