# %%
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Agent(ABC):
    estimated_values: np.ndarray

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def act(self, random_state) -> int:
        ...

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        ...


def multiarmed_bandit(
    n_arms: int,
    seed: int,
    n_episodes: int,
    agent: Agent,
) -> None:
    name = str(type(agent))
    true_values = np.random.default_rng(seed).standard_normal(size=n_arms)

    expected_rewards = []
    for i in range(n_episodes):
        action = agent.act(seed + i)
        reward: float = np.random.default_rng(seed + i).normal(loc=true_values[action])  # noqa
        agent.update(action, reward)
        expected_rewards.append(true_values[action])

    print("=" * 40 + name + "=" * 40)
    print(f"True values: {[round(value, 3) for value in true_values]}")
    print(f"Estimated values: {[round(value, 3) for value in agent.estimated_values]}")

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


n_arms = 10
seed = 42
n_episodes = 10 ** 5
lr = 0.01

# %%
# Naive greedy
class NaiveGreedyAgent(Agent):
    def __init__(self, n_arms: int, lr: float):
        self.estimated_values = np.zeros((n_arms,))
        self.lr = lr

    def act(self, random_state):
        return greedy_action(values=self.estimated_values, random_state=random_state)

    def update(self, action: int, reward: float):
        self.estimated_values[action] += self.lr * (reward - self.estimated_values[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=n_episodes,
    agent=NaiveGreedyAgent(n_arms=n_arms, lr=lr),
)
