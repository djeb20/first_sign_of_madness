import gym
from gym import spaces
import numpy as np
from q_agent import Agent

class FactoredState(gym.ObservationWrapper):

    def __init__(self, env):

        super().__init__(env)

        self._observation_space = spaces.Dict({"taxi_row": spaces.Discrete(5), "taxi_col": spaces.Discrete(5), "passenger_location": spaces.Discrete(5), "destination": spaces.Discrete(4)})

    def observation(self, obs):

        # Decode function copied from Taxi-v3: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

        out = np.empty(4)
        out[0] = obs % 4; obs //= 4
        out[1] = obs % 5; obs //= 5
        out[2] = obs % 5; obs //= 5
        out[3] = obs
        assert 0 <= obs < 5
        return out 

env = FactoredState(gym.make('Taxi-v3'))

S = np.arange(env.action_space.n)

agent = Agent(env, S, epsilon=0.1, gamma=0.99, alpha=0.2)

num_episodes = 100000

q_table = agent.train(num_episodes)

