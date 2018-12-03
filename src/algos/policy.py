import numpy as np


class PolicyBase:
    def __init__(self, q_values):
        self.q_values = q_values

    def __call__(self, s):
        raise NotImplementedError


class Greedy(PolicyBase):
    def __init__(self, q_values):
        super(Greedy, self).__init__(q_values)

    def __call__(self, s):
        return np.argmax(self.q_values[s])

    def get_greedy_pi(self):
        return np.argmax(self.q_values, axis=1)


class EpsilonGreedy(PolicyBase):
    def __init__(self, q_values, epsilon=0.1):
        super(EpsilonGreedy, self).__init__(q_values)
        self.epsilon = epsilon

    def __call__(self, s):
        return np.random.choice([np.argmax(self.q_values[s]), np.random.randint(self.q_values.shape[-1])],
                                p=[1 - self.epsilon, self.epsilon])


class Boltzman(PolicyBase):
    def __init__(self, q_values, alpha=0.1):
        super(Boltzman, self).__init__(q_values)
        self.temperature = 1/alpha

    def __call__(self, s):
        exp_q = np.exp(self.q_values[s]/self.temperature)
        z = np.sum(exp_q)
        return np.random.choice([*range(self.q_values.shape[-1])], p=exp_q/z)
