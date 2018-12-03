import numpy as np
from .policy import PolicyBase
import copy


def q_learn(env, policy, n_episode=int(1e4), epi_length=10, gamma=0.95, alpha=0.1):
    assert isinstance(policy, PolicyBase), 'policy must inherit Policy class'
    q_values = np.ones((env.n_states, env.n_actions))
    obs = env.reset()
    for _ in range(n_episode):
        for i in range(epi_length):
            action = policy(q_values[obs])
            next_obs, reward = env.step(action)
            q_values[obs, action] = q_values[obs, action] \
                                    + alpha * (reward + gamma * np.max(q_values[next_obs]) - q_values[obs, action])
            obs = next_obs
        obs = env.reset()
    return q_values


def compute_q_via_dp(env, delta=1e-4, gamma=0.95):
    """
    :param env:
    :param delta:
    :param gamma:
    :return: state_values, q_values
    """
    state_values = np.zeros(env.n_states, dtype=np.float32)
    while True:
        max_delta = 0
        for s in range(env.n_states):
            old_vs = state_values[s]
            state_values[s] = np.max(__compute_q_s_with_v(env, s, state_values, gamma))
            max_delta = max(abs(state_values[s] - old_vs), max_delta)
        # print('max delta: {:>.5f}'.format(max_delta))
        if max_delta < delta:
            break
    return __compute_q_with_v(env, state_values, gamma)


def policy_iteration(env, gamma, pi=None):
    if pi is None:
        pi = np.random.randint(env.n_actions, size=env.n_states)
    n_iter = 0
    state_values = copy.deepcopy(env.rewards)
    while True:
        old_pi = copy.deepcopy(pi)
        state_values = compute_v_for_pi(env, pi, gamma, state_values)
        pi = np.argmax(__compute_q_with_v(env, state_values, gamma), axis=1)
        if np.all(old_pi == pi):
            return pi
        else:
            n_iter += 1
            if n_iter > 1000:
                print('n_iter: ', n_iter)
                print('rewards: ', env.rewards)


def compute_v_for_pi(env, pi, gamma, state_values_pi=None, delta=1e-4):
    """
    policy evaluation
    """
    trans_probs_pi = np.concatenate([np.expand_dims(env.trans_probs[s, pi[s]], axis=0) for s in range(env.n_states)])
    assert trans_probs_pi.shape == 2 * (env.n_states,), 'Invalid shape computed'
    state_values_pi = copy.deepcopy(env.rewards) if state_values_pi is None else state_values_pi
    while True:
        max_delta = 0
        for s in range(env.n_states):
            old_vs = state_values_pi[s]
            state_values_pi[s] = env.rewards[s] + trans_probs_pi[s].dot(gamma * state_values_pi)
            max_delta = max(abs(state_values_pi[s] - old_vs), max_delta)
        if max_delta < delta:
            return state_values_pi


def compute_q_for_pi(env, pi, gamma):
    state_values = compute_v_for_pi(**locals())
    return __compute_q_with_v(env, state_values, gamma)


def __compute_q_with_v(env, state_values, gamma):
    q_values = np.concatenate([np.expand_dims(__compute_q_s_with_v(env, s, state_values, gamma), axis=0)
                               for s in range(env.n_states)])
    assert q_values.shape == (env.n_states, env.n_actions), 'Invalid shape {}'.format(q_values.shape)
    return q_values


def __compute_q_s_with_v(env, s, state_values, gamma):
    assert state_values.shape == (env.n_states,), 'Invalid state_values are given {}'.format(state_values)
    q_values_s = env.rewards[s] + gamma * np.sum(env.trans_probs[s] * state_values, axis=-1)
    assert q_values_s.shape == (env.n_actions,), 'Invalid shape computed'
    return q_values_s
