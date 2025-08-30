# %%
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.stats import norm


class Agent(ABC):
    estimates: np.ndarray = np.array([])

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
    name = type(agent).__name__
    true_values = np.random.default_rng(seed).standard_normal(size=n_arms)

    expected_rewards = []
    cumulative_rewards = []
    for i in range(n_episodes):
        action = agent.act(seed + i)
        reward: float = np.random.default_rng(seed + i).normal(loc=true_values[action])  # noqa
        agent.update(action, reward)
        expected_rewards.append(true_values[action])
        cumulative_rewards.append(
            cumulative_rewards[-1] + reward
            if len(cumulative_rewards) > 0
            else reward
        )

    print("=" * 40 + name + "=" * 40)
    print(f"True values: {[round(value, 3) for value in true_values]}")
    print(f"Estimates: {[round(value, 3) for value in agent.estimates]}")

    plt.plot(expected_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Expected reward")
    plt.title(name)
    plt.grid()
    plt.show()
    plt.clf()

    plt.plot(cumulative_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title(name)
    plt.grid()
    plt.show()
    plt.clf()


def greedy_action(values: np.ndarray, random_state):
    greedy_actions = np.nonzero(values == values.max())[0]
    return np.random.default_rng(random_state).choice(greedy_actions)


n_arms = 10
seed = 42
lr = 0.01

# %%
class NaiveGreedyAgent(Agent):
    def __init__(self, n_arms: int, lr: float):
        self.estimates = np.zeros((n_arms,))
        self.lr = lr

    def act(self, random_state):
        return greedy_action(values=self.estimates, random_state=random_state)

    def update(self, action: int, reward: float):
        self.estimates[action] += self.lr * (reward - self.estimates[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=10 ** 4,
    agent=NaiveGreedyAgent(n_arms=n_arms, lr=lr),
)

# %%
class OptimisticGreedyAgent(Agent):
    def __init__(self, n_arms: int, lr: float):
        self.estimates = np.zeros((n_arms,)) + norm.ppf(0.99)
        self.lr = lr

    def act(self, random_state):
        return greedy_action(values=self.estimates, random_state=random_state)

    def update(self, action: int, reward: float):
        self.estimates[action] += self.lr * (reward - self.estimates[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=10 ** 4,
    agent=OptimisticGreedyAgent(n_arms=n_arms, lr=lr),
)

# %%
class EpsGreedyAgent(Agent):
    def __init__(self, n_arms: int, lr: float, eps: float):
        self.estimates = np.zeros((n_arms,))
        self.lr = lr
        self.eps = eps

    def act(self, random_state):
        rng = np.random.default_rng(random_state)
        return (
            greedy_action(values=self.estimates, random_state=random_state)
            if rng.uniform() > self.eps
            else rng.integers(len(self.estimates))
        )

    def update(self, action: int, reward: float):
        self.estimates[action] += self.lr * (reward - self.estimates[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=10 ** 4,
    agent=EpsGreedyAgent(n_arms=n_arms, lr=lr, eps=0.01),
)

# %%
class UCBAgent(Agent):
    def __init__(self, n_arms: int, lr: float, c: float):
        self.estimates = np.zeros((n_arms,))
        self.num_updates = np.zeros((n_arms,))
        self.total_updates = 0
        self.lr = lr
        self.c = c

    def act(self, random_state):
        if self.num_updates.min() == 0:
            return np.argmin(self.num_updates)

        ucb_values = self.estimates + self.c * np.sqrt(np.log(self.num_updates.sum()) / self.num_updates)
        return greedy_action(values=ucb_values, random_state=random_state)

    def update(self, action: int, reward: float):
        self.num_updates[action] += 1
        self.estimates[action] += self.lr * (reward - self.estimates[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=2 * 10 ** 5,
    agent=UCBAgent(n_arms=n_arms, lr=lr, c=100),
)

# %%
class PolicyGradientAgent(Agent):
    def __init__(self, n_arms: int, lr: float):
        self.estimates = np.zeros((n_arms,))
        self.num_updates = 0
        self.lr = lr

    @staticmethod
    def _softmax(x: np.ndarray):
        exponentiated = np.exp(x - x.max())  # Subtraction for numerical stability
        return exponentiated / exponentiated.sum()

    def act(self, random_state):
        return np.random.default_rng(random_state).choice(
            np.arange(len(self.estimates)),
            p=self._softmax(self.estimates * np.log(self.num_updates + 1)),
        )

    def update(self, action: int, reward: float):
        self.num_updates += 1
        self.estimates[action] += self.lr * (reward - self.estimates[action])


multiarmed_bandit(
    n_arms=n_arms,
    seed=seed,
    n_episodes=10 ** 3,
    agent=PolicyGradientAgent(n_arms=n_arms, lr=lr),
)
