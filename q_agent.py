import numpy as np
from tqdm import tqdm
from collections import defaultdict
    
class Agent:
    
    def __init__(self, env, S, epsilon=0.05, gamma=0.99, alpha=0.2):
        
        # Agent hyper-parameters
        self.action_dim = env.action_space.n
        self.actions = np.arange(self.action_dim)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # Environment and coalition
        self.env = env
        self.S = S
        self.not_S = [i for i in range(self.action_dim) if i not in S]

        # # Parameters for epsilon and alpha decay
        # n = 7
        # self.decay_ep = float('0.' + '9' * n)
        
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))
        
        # Chooses action with epsilon greedy policy
        self.choose_action = lambda state : np.random.randint(self.action_dim) if np.random.rand() < self.epsilon else self.best_action(state)

    def best_action(self, state):
        """
        Finds the greedy best action.
        Can only choose from valid "actions"
        """

        q_values = self.Q_table[state.tobytes()]

        return np.random.choice(self.actions[q_values == q_values.max()])
        
    def update(self, state, action, new_state, reward, done):
        
        # Usual update, for only valid "actions"
        td_error = reward + (1 - done) * self.gamma * self.Q_table[new_state.tobytes()].max() - self.Q_table[state.tobytes()][action]
        self.Q_table[tuple(state)][action] += self.alpha * td_error

        # # Decay epsilon, when it is small enough, decay alpha.
        # self.epsilon *= self.decay_ep
        
        # Return a update error to stop training.
        return td_error

    def mask_state(self, state):
        """
        Takes a state and masks out state features according to a coalition.
        """

        out = state.copy()
        out[self.not_S] = -1

        return out    

    def train(self, num_episodes, q_tables_dict={}, tol=1e-10, scale=100):
        """
        Trains one agent using Q-Learning.
        """

        count = 0
        mean_error = 0
        returns = []

        for i in tqdm(range(num_episodes)):

            s = self.env.reset()
            state = self.mask_state(s)
            ret = 0

            while True:

                # Usual RL, choose action, execute, update
                # Only choose and update from valid actions
                action = self.choose_action(state)

                new_s, reward, done, _ = self.env.step(action)
                if reward == -10:
                    reward = -1
                new_state = self.mask_state(new_s)
                update = self.update(state, action, new_state, reward, done)
                state = new_state

                ret += reward

                # Track errors
                count += 1
                mean_error *= count - 1
                mean_error += abs(update)
                mean_error /= count

                if done: break

            returns.append(ret)

            if i % scale == 0:
                print()
                print('Average td error: {:0.04f}, \
                    Average ret: {:0.02f}, \
                        Step: {}'.format(mean_error, np.mean(returns[-100:]), count))

            # Learning converges according to tolerance
            if mean_error < tol: break

        # Top line is for normal, bottom for multiprocessing
        return dict(self.Q_table)
        # q_tables_dict[tuple(self.S)] = dict(self.Q_table)