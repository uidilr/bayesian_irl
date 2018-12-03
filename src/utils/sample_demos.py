def sample_demos(env, policy, n_episode=20, epi_length=10):
    obs = env.reset()
    for _ in range(n_episode):
        for i in range(epi_length):
            action = policy(obs)
            yield [obs, action]
            obs, _ = env.step(action)
        obs = env.reset()
