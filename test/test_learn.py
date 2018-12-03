"""
Run test with,
python -m unittest test/test_learn.py
"""
from src.algos import learn
import numpy as np
import unittest


class TestLearn(unittest.TestCase):
    """
    test class of algos.learn.py
    """

    def test_compute_q_s_with_v(self):
        class Env:
            def __init__(self):
                self.rewards = np.array([1, 2])
                self.n_states = 2
                self.n_actions = 2
                self.trans_probs = np.empty((self.n_states, self.n_actions, self.n_states))
                for s in range(self.n_states):
                    for a in range(self.n_actions):
                        trans_prob_sas0 = np.random.random()
                        self.trans_probs[s, a] = [trans_prob_sas0, 1-trans_prob_sas0]
        np.random.seed(5)
        gamma = np.random.random()
        state_values = np.array([0, 1])
        env = Env()
        for s in range(env.n_states):
            q = learn.__compute_q_s_with_v(env, s=s, state_values=state_values, gamma=gamma)
            q_true = np.array([env.rewards[s] + np.sum(gamma * env.trans_probs[s, a] * state_values)
                               for a in range(env.n_states)])
            self.assertTrue(np.all(q == q_true), msg='q{}: {} != {}'.format(s, q, q_true))


if __name__ == '__main__':
    unittest.main()