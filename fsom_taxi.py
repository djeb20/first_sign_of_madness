import gym
from gym import spaces
import numpy as np
from q_agent import Agent
from q_agent_crazy import Agent_crazy

class FactoredState(gym.ObservationWrapper):

    def __init__(self, env):

        super().__init__(env)

        self._observation_space = spaces.MultiDiscrete([5, 5, 5, 4])

    def observation(self, obs):

        # Decode function copied from Taxi-v3: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

        out = np.empty(4)
        out[3] = obs % 4; obs //= 4
        out[2] = obs % 5; obs //= 5
        out[1] = obs % 5; obs //= 5
        out[0] = obs
        return out 

env = FactoredState(gym.make('Taxi-v3'))

# FULLY OBSERVABLE WITHOUT MESSAGES

# S = np.arange(1, env.observation_space.shape[0])
# agent = Agent(env, S, epsilon=0.05, gamma=0.99, alpha=0.2)
# num_episodes = 100000
# q_table = agent.train(num_episodes)

# POMDP WITH MESSAGES

S = np.arange(1, env.observation_space.shape[0])
agent = Agent_crazy(env, S, epsilon=0.05, gamma=0.99, alpha=0.2)
num_episodes = 100000
q_table = agent.train(num_episodes)