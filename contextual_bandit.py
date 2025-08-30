import numpy as np
import matplotlib.pyplot as plt


def greedy_action(values: np.ndarray, random_state):
    greedy_actions = np.nonzero(values == values.max())[0]
    return np.random.default_rng(random_state).choice(greedy_actions)


n_arms = 10
seed = 42
n_episodes = 10 ** 5
lr = 0.01

true_values = np.random.default_rng(seed).standard_normal(size=n_arms)


estimated_values = np.zeros((n_arms,))
expected_rewards = []
for i in range(n_episodes):
    action = greedy_action(estimated_values, seed + i)
    reward = np.random.default_rng(seed + i).normal(loc=true_values[action])
    estimated_values[action] += lr * (reward - estimated_values[action])
    expected_rewards.append(true_values[action])


print(f"True values: {[round(value, 3) for value in true_values]}")
print(f"Estimated values: {[round(value, 3) for value in estimated_values]}")

plt.plot(expected_rewards)
plt.xlabel("Episode")
plt.ylabel("Expected reward")
plt.grid()
plt.show()
plt.clf()
