"""REINFOCE trains RL agents with a policy-gradient method.
* Update parameters by stochastic gradient ascent
* Uses policy gradient theorm
* Uses return $v_t$ as an unbiased sample of $Q^{\pi_\theta}(s_t, a_t)$
"""


def reinforce(policy, env, n_episodes):
    """Optimizes `policy` using REINFORCE algorithm
    Assumes policy is already initialized.
    """
    for i_episode in range(n_episodes):
        episode = run_episode(policy, env)
        for t in range(len(episode)):
            v_t = sum(episode[1:])
            