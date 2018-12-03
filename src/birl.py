import numpy as np
from algos import learn, policy
from env import LoopEnv
from utils import sample_demos, prob_dists
import argparse
import copy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Inverse Reinforcement Learning')
    parser.add_argument('--policy', '-p', choices=('eps', 'bol'))
    parser.add_argument('--alpha', '-a', default=10, type=float, help='1/temperature of boltzmann distribution, '
                                                                      'larger value makes policy close to the greedy')
    parser.add_argument('--env_id', default=0, type=int)
    parser.add_argument('--r_max', default=1, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--n_iter', default=5000, type=int)
    parser.add_argument('--burn_in', default=1000, type=int)
    parser.add_argument('--dist', default='uniform', type=str, choices=['uniform', 'gaussian', 'beta', 'gamma'])
    return parser.parse_args()


def bayesian_irl(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, burn_in, sample_freq):
    assert burn_in <= n_iter
    sampled_rewards = np.array(list(policy_walk(**locals()))[burn_in::sample_freq])
    return sampled_rewards


def policy_walk(env, demos, step_size, n_iter, r_max, prior, alpha, gamma, **kwargs):
    assert r_max > 0, 'r_max must be positive'
    # step 1
    env.rewards = sample_random_rewards(env.n_states, step_size, r_max)
    # step 2
    pi = learn.policy_iteration(env, gamma)
    # step 3
    for _ in tqdm(range(n_iter)):
        env_tilda = copy.deepcopy(env)
        env_tilda.rewards = mcmc_reward_step(env.rewards, step_size, r_max)
        q_pi_r_tilda = learn.compute_q_for_pi(env, pi, gamma)
        if is_not_optimal(q_pi_r_tilda, pi):
            pi_tilda = learn.policy_iteration(env_tilda, gamma, pi)
            if np.random.random() < compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
                env, pi = env_tilda, pi_tilda
        else:
            if np.random.random() < compute_ratio(demos, env_tilda, pi, env, pi, prior, alpha, gamma):
                env = env_tilda
        yield env.rewards


def is_not_optimal(q_values, pi):
    n_states, n_actions = q_values.shape
    for s in range(n_states):
        for a in range(n_actions):
            if q_values[s, pi[s]] < q_values[s, a]:
                return True
    return False


def compute_ratio(demos, env_tilda, pi_tilda, env, pi, prior, alpha, gamma):
    ln_p_tilda = compute_posterior(demos, env_tilda, pi_tilda, prior, alpha, gamma)
    ln_p = compute_posterior(demos, env, pi, prior, alpha, gamma)
    ratio = np.exp(ln_p_tilda - ln_p)
    return ratio


def compute_posterior(demos, env, pi, prior, alpha, gamma):
    q = learn.compute_q_for_pi(env, pi, gamma)
    ln_p = np.sum([alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s]))) for s, a in demos]) + np.log(prior(env.rewards))
    return ln_p


def mcmc_reward_step(rewards, step_size, r_max):
    new_rewards = copy.deepcopy(rewards)
    index = np.random.randint(len(rewards))
    step = np.random.choice([-step_size, step_size])
    new_rewards[index] += step
    new_rewards = np.clip(a=new_rewards, a_min=-r_max, a_max=r_max)
    if np.all(new_rewards == rewards):
        new_rewards[index] -= step
    assert np.any(rewards != new_rewards), 'rewards do not change: {}, {}'.format(new_rewards, rewards)
    return new_rewards


def sample_random_rewards(n_states, step_size, r_max):
    """
    sample random rewards form gridpoint(R^{n_states}/step_size).
    :param n_states:
    :param step_size:
    :param r_max:
    :return: sampled rewards
    """
    rewards = np.random.uniform(low=-r_max, high=r_max, size=n_states)
    # move these random rewards toward a gridpoint
    # add r_max to make mod to be always positive
    # add step_size for easier clipping
    rewards = rewards + r_max + step_size
    for i, reward in enumerate(rewards):
        mod = reward % step_size
        rewards[i] = reward - mod
    # subtracts added values from rewards
    rewards = rewards - (r_max + step_size)
    return rewards

def prepare_prior(dist, r_max):
    prior = getattr(prob_dists, dist[0].upper() + dist[1:] + 'Dist')
    if dist == 'uniform':
        return prior(xmax=r_max)
    elif dist == 'gaussian':
        return prior()
    elif dist in {'beta', 'gamma'}:
        return prior(loc=-r_max, scale=1/(2 * r_max))
    else:
        raise NotImplementedError('{} is not implemented.'.format(dist))

def main(args):
    np.random.seed(5)

    # prepare environments
    if args.env_id == 0:
        env_args = dict(loop_states=[1, 3, 2])
    else:
        assert args.env_id == 1, 'Invalid env id is given'
        env_args = dict(loop_states=[0, 3, 2])
    env_args['rewards'] = [0, 0, 0.7, 0.7]
    env = LoopEnv(**env_args)

    # sample expert demonstrations
    expert_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    if args.policy == 'bol':
        expert_policy = policy.Boltzman(expert_q_values, args.alpha)
        print('pi \n', np.array([np.exp(args.alpha * expert_q_values[s])
                                 / np.sum(np.exp(args.alpha * expert_q_values[s]), axis=-1) for s in env.states]))
    else:
        expert_policy = policy.EpsilonGreedy(expert_q_values, epsilon=0.1)
    demos = np.array(list(sample_demos(env, expert_policy)))
    print('sub optimal actions {}/{}'.format(demos[:, 1].sum(), len(demos)))
    assert np.all(expert_q_values[:, 0] > expert_q_values[:, 1]), 'a0 must be optimal action for all the states'

    # run birl
    prior = prepare_prior(args.dist, args.r_max)
    sampled_rewards = bayesian_irl(env, demos, step_size=0.05, n_iter=args.n_iter, r_max=args.r_max, prior=prior,
                                   alpha=args.alpha, gamma=args.gamma, burn_in=args.burn_in, sample_freq=1)

    # plot rewards
    fig, ax = plt.subplots(1, env.n_states, sharey='all')
    for i, axes in enumerate(ax.flatten()):
        axes.hist(sampled_rewards[:, i], range=(-args.r_max, args.r_max))
    fig.suptitle('Loop Environment {}'.format(args.env_id), )
    path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-2], 'results',
                              'samples_env{}.png'.format(args.env_id))
    plt.savefig(path)

    est_rewards = np.mean(sampled_rewards, axis=0)
    print('True rewards: ', env_args['rewards'])
    print('Estimated rewards: ', est_rewards)

    # compute optimal q values for estimated rewards
    env.rewards = est_rewards
    learner_q_values = learn.compute_q_via_dp(env, gamma=args.gamma)
    for print_value in ('expert_q_values', 'learner_q_values'):
        print(print_value + '\n', locals()[print_value])
    print('Is a0 optimal action for all states: ', np.all(learner_q_values[:, 0] > learner_q_values[:, 1]))


if __name__ == '__main__':
    args = get_args()
    main(args)
