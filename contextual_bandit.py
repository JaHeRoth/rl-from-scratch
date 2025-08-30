# %%
import numpy as np
import matplotlib.pyplot as plt


def multiarmed_bandit(
    n_arms: int,
    seed: int,
    n_episodes: int,
    init_func: callable,
    action_func: callable,
    update_func: callable,
    name: str,
) -> None:
    true_values = np.random.default_rng(seed).standard_normal(size=n_arms)

    estimated_values = init_func(n_arms)
    expected_rewards = []
    for i in range(n_episodes):
        action = action_func(estimated_values, seed + i)
        reward = np.random.default_rng(seed + i).normal(loc=true_values[action])
        estimated_values = update_func(estimated_values, action, reward)
        expected_rewards.append(true_values[action])

    print("=" * 40 + name + "=" * 40)
    print(f"True values: {[round(value, 3) for value in true_values]}")
    print(f"Estimated values: {[round(value, 3) for value in estimated_values]}")

    plt.plot(expected_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Expected reward")
    plt.title(name)
    plt.grid()
    plt.show()
    plt.clf()


def greedy_action(values: np.ndarray, random_state):
    greedy_actions = np.nonzero(values == values.max())[0]
    return np.random.default_rng(random_state).choice(greedy_actions)


# %%
n_arms = 10
seed = 42
n_episodes = 10 ** 5
lr = 0.01

# Naive greedy
multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=n_episodes,
    init_func=lambda n_arms: np.zeros((n_arms,)),
    action_func=greedy_action,
    update_func=lambda estimates, action, reward: estimates[action] + lr * (reward - estimates[action]),
    name="Naive greedy",
)
