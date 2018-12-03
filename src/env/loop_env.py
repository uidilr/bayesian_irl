import numpy as np


class LoopEnv:
    def __init__(self, rewards=None, loop_states=None, trans_probs=None):
        self.n_states = 4
        self.n_actions = 2
        self.states = np.array([*range(self.n_states)], dtype=int)
        self.loop_states = loop_states
        self.trans_probs = trans_probs if trans_probs else self._get_trans_probs()
        self.state = None
        self._rewards = rewards

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards):
        if isinstance(rewards, list):
            assert len(rewards) == self.n_states, 'Invalid rewards specified'
            rewards = np.array(rewards)
        assert rewards.shape == (self.n_states,), 'Invalid rewards specified'
        self._rewards = rewards

    def step(self, a):
        # breakpoint()
        assert 0 <= a < self.n_actions, '{} is invalid action index. ' \
                                        'Action must be in the range of [0, {}]'.format(a, self.n_actions)
        self.state = np.random.choice(self.states, p=self.trans_probs[self.state, a])
        reward = self._get_reward()
        return self.state, reward

    def reset(self):
        self.state = np.random.randint(self.n_states)
        return self.state

    def _get_reward(self, state=None):
        assert self.rewards is not None, 'rewards is not specified'
        state = self.state if state is None else state
        return self.rewards[state]

    def _get_trans_probs(self):
        a0 = 0
        a1 = 1
        trans_probs = np.empty(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        a1_next_state = int(*[s for s in self.states if s not in self.loop_states])
        for state in self.states:
            trans_probs[a1_next_state, a0, state] = 0 if state == a1_next_state else 1/3
        for state, a0_next_state in zip(self.loop_states, self.loop_states[1:] + [self.loop_states[0]]):
            trans_probs[state, 0] = np.eye(self.n_states)[a0_next_state]
        trans_probs[:, a1] = np.eye(self.n_states, dtype=np.float32)[a1_next_state]
        return trans_probs


if __name__ == '__main__':
    # The expert in this environments loops s1, s2, s3
    trans_probs = np.empty(shape=(4, 2, 4), dtype=np.float32)
    loop_states = [1, 3, 2]
    a1_next_state = [s for s in range(trans_probs.shape[0]) if s not in loop_states][0]
    trans_probs[:, 1] = np.eye(4)[a1_next_state]
    for state in range(trans_probs.shape[0]):
        trans_probs[a1_next_state, 0, state] = 0 if state == a1_next_state else 1/3
    for state, a0_next_state in zip(loop_states, loop_states[1:] + [loop_states[0]]):
        trans_probs[state, 0] = np.eye(4)[a0_next_state]

    env = LoopEnv(rewards=[0, 0, 0, 1], loop_states=loop_states)
    obs = env.reset()
    for _ in range(100):
        a = np.random.randint(env.n_actions)
        obs, reward = env.step(a)
        print('obs: {}, action: {}, reward: {}'.format(obs, a, reward))