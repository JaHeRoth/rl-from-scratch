# %%
import gymnasium as gym
import numpy as np
import polars as pl
from numpy.random import Generator
from tqdm import tqdm
from joblib import Parallel, delayed

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

env_params = dict(
    id="FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True
)
gamma = 0.95

env = gym.make(**env_params)
model = pl.DataFrame(
    [
        [s, a, p, s_next, r, terminal]
        for (s, state_action_dict) in env.unwrapped.P.items()
        for (a, outcomes) in state_action_dict.items()
        for (p, s_next, r, terminal) in outcomes
    ],
    orient="row",
    schema={
        "state": pl.Int64,
        "action": pl.Int64,
        "probability": pl.Float64,
        "next_state": pl.Int64,
        "reward": pl.Float64,
        "done": pl.Boolean,
    },
)

def eps_greedy_policy(q: pl.DataFrame, eps: float) -> pl.DataFrame:
    return (
        q.with_columns(q_max=pl.max("q").over("state"))
        .with_columns(greedy_choice=pl.col("q") == pl.col("q_max"),)
        .select(
            "state",
            "action",
            policy=(
                (1 - eps) * pl.col("greedy_choice").cast(pl.Float64)
                / pl.sum("greedy_choice").over("state")
                + eps / pl.len().over("state")
            ),
        )
    )

def sample_from_policy(
    policy: pl.DataFrame, state: int, seed: int | None = None
) -> int:
    relevant_policy = (
        policy
        .filter(pl.col("state") == state)
        .sort("action")
    )
    action = np.random.default_rng(seed).choice(
        relevant_policy["action"], p=relevant_policy["policy"]
    )
    return int(action)

def bellman_optimality_equation(
    model: pl.DataFrame, q: pl.DataFrame, gamma: float
) -> pl.DataFrame:
    v = q.group_by("state").agg(v=pl.max("q"))
    return (
        model
        .join(v, left_on="next_state", right_on="state")
        .group_by(["state", "action"])
        .agg(
            q=(
                pl.col("probability") * (pl.col("reward") + gamma * pl.col("v"))
            ).sum(),
        )
    )

# (Synchronous) value iteration
required_delta = 10 ** -1.25  # We want an almost-optimal policy
q = model.group_by(["state", "action"]).agg(q=pl.lit(0.0))
max_delta = np.inf
while max_delta >= required_delta:
    q_new = bellman_optimality_equation(model, q, gamma)
    max_delta = (
        q
        .join(q_new, on=["state", "action"], suffix="_new")
        .select((pl.col("q_new") - pl.col("q")).abs().max())
        .item()
    )
    q = q_new

policy = eps_greedy_policy(q, eps=0.0)
print(policy.sort(["state", "action"]).pivot(on="action", index="state"))

# %%
# Run learned (greedy) policy
num_episodes = 10
seed = 42
env = gym.make(**env_params)

total_reward = 0.0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset(seed=seed + episode)

    episode_over = False
    while not episode_over:
        action = sample_from_policy(policy, state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
print(f"{total_reward=}")
env.close()

# %%
# Monte Carlo rollout (or rather, n-step TD rollout)
def simulate_policy(
    model: pl.DataFrame,
    policy: pl.DataFrame,
    state: int,
    action: int,
    max_depth: int,
    q: pl.DataFrame,
    seed: int,
) -> float:
    reward = 0.0
    factor = 1.0
    for i in range(max_depth):
        realization = model.filter(
            (pl.col("state") == state) & (pl.col("action") == action)
        ).sample(seed=seed + i)
        reward += factor * realization["reward"].item()
        state = realization["next_state"].item()
        if realization["done"].item():
            return reward

        action = sample_from_policy(policy, state, seed=seed + max_depth + i)
        factor *= gamma

    leaf_q = q.filter((pl.col("state") == state) & (pl.col("action") == action))["q"].item()
    return reward + factor * leaf_q

def n_depth_rollout(
    model: pl.DataFrame,
    policy: pl.DataFrame,
    state: int,
    n_rollouts: int,
    max_depth: int,
    q: pl.DataFrame,
    seed: int,
) -> float:
    return max(
        model["action"].unique(),
        key=lambda action: np.mean(
            Parallel(n_jobs=-1)(
                delayed(simulate_policy)(
                    model, policy, state, action, max_depth, q, seed=seed + action * 10 ** 8 + sim_i * 10 ** 3
                )
                for sim_i in range(n_rollouts)
            )
        ),
    )

num_episodes = 10
n_rollouts = 1000
max_rollout_depth = 10
seed = 42
env = gym.make(**env_params)

total_reward = 0.0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset(seed=seed + episode)

    episode_over = False
    while not episode_over:
        seed += 10 ** 10
        action = n_depth_rollout(
            model, policy, state, n_rollouts, max_depth=max_rollout_depth, q=q, seed=seed
        )
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
print(f"{total_reward=}")
env.close()

# %%
# Monte Carlo Tree Search (MCTS)
class MonteCarloSearchTree:
    def __init__(self, model: pl.DataFrame, policy: pl.DataFrame, q: pl.DataFrame, gamma: float, seed: int):
        self.model = model
        self.policy = policy
        self.gamma = gamma
        self.seed = seed

        self.q = q.pivot(on="action", index="state").drop("state").to_numpy()
        self.n_visits = np.zeros((q["state"].n_unique(), q["action"].n_unique()))

    def next_action(
        self, state: int, n_rollouts: int, max_depth: int, lr: float, ucb_c: float
    ) -> int:
        for _ in range(n_rollouts):
            trajectory = []
            for step in range(max_depth):
                state_ucb_values = (
                    self.q[state, :]
                    + ucb_c
                    * np.sqrt(np.log(self.n_visits[state, :].sum() + 1) / (self.n_visits[state, :] + 1))
                )
                action = state_ucb_values.argmax()
                realization = model.filter(
                    (pl.col("state") == state) & (pl.col("action") == action)
                ).sample(seed=self.seed)
                trajectory.append((state, action, realization["reward"].item()))
                state = realization["next_state"].item()
                if realization["done"].item():
                    break
                self.seed += 1

            target = 0.0 if realization["done"].item() else self.q[state, :].max()
            for state, action, reward in reversed(trajectory):
                target = self.gamma * target + reward
                self.q[state, action] += lr * (target - self.q[state, action])
                self.n_visits[state, action] += 1

        return self.q[state, :].argmax()


num_episodes = 10
n_rollouts = 1000
max_rollout_depth = 10
mcts_lr = 0.01
mcts_ucb_c = q["q"].max()
seed = 42
mcts_tree = MonteCarloSearchTree(model, policy, q, gamma, seed)
env = gym.make(**env_params)

total_reward = 0.0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset(seed=seed + 10 ** 15 + episode)

    episode_over = False
    while not episode_over:
        action = mcts_tree.next_action(
            state, n_rollouts, max_depth=max_rollout_depth, lr=mcts_lr, ucb_c=mcts_ucb_c
        )
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
print(f"{total_reward=}")
env.close()

# %%
