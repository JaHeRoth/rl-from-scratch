# %%
from typing import Iterable

import gymnasium as gym
import numpy as np
import polars as pl
from numpy.random import Generator
from tqdm import tqdm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

env_params = dict(
    id="FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True
)
gamma = 0.95

env = gym.make(**env_params)
# TODO: Migrate away from tabular state space
state_space = list(range(env.observation_space.n))
action_space = list(range(env.action_space.n))

def init_q(state_space: Iterable, action_space: Iterable) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "state": state,
                "action": action,
                "q": 0.0,
            }
            for state in state_space
            for action in action_space
        ]
    )

def init_policy(state_space: Iterable, action_space: Iterable) -> pl.DataFrame:
    return (
        pl.DataFrame(
            [
                {
                    "state": state,
                    "action": action,
                }
                for state in state_space
                for action in action_space
            ]
        )
        .with_columns(policy=1.0 / pl.n_unique("action"))
    )

def sample_from_policy(
    policy: pl.DataFrame, state: int, seed: int | None | Generator = None
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

def learning_rate_for_update(
    base_learning_rate: float, update_number: int, period: float, numerator_power: float
) -> float:
    numerator = max(1, update_number / period) ** numerator_power
    return base_learning_rate / numerator

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

def policy_iteration(
    q_in: pl.DataFrame,
    policy_in: pl.DataFrame,
    evaluation_func: callable,
    gamma: float,
    lr_schedule: Iterable[float],
    eps_schedule: Iterable[float],
    verbose: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    q = q_in
    policy = policy_in

    for eps in eps_schedule:
        while True:
            q = evaluation_func(
                q_in=q, policy=policy, gamma=gamma, lr_schedule=lr_schedule
            )
            next_policy = eps_greedy_policy(q, eps)

            policy_deltas = (
                policy
                .join(next_policy, on=["state", "action"], suffix="_next")
                .filter(pl.col("policy") != pl.col("policy_next"))
            )
            if len(policy_deltas) == 0:
                break

            policy = next_policy
            if verbose:
                print(q.sort(["state", "action"]).pivot(on="action", index="state"))
                print(policy.sort(["state", "action"]).pivot(on="action", index="state"))

    return q, policy

# %%
# Sarsa
def sarsa(
    q_in: pl.DataFrame, policy: pl.DataFrame, gamma: float, lr_schedule: Iterable[float], seed: int = 42
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    state, _ = env.reset(seed=seed)
    action = sample_from_policy(policy, state, seed=seed)
    for step, lr in tqdm(enumerate(lr_schedule), desc="Running steps"):
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        if episode_over:
            target = float(reward)
        else:
            next_action = sample_from_policy(policy, state=next_state, seed=seed + step + 1)
            target = float(reward) + gamma * q[next_state, next_action]

        q[state, action] += lr * (target - q[state, action])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
            action = sample_from_policy(policy, state, seed=seed + step + 1)
        else:
            state = next_state
            action = next_action

    return q_in.with_columns(q=q.flatten())


q, policy = policy_iteration(
    q_in=init_q(state_space, action_space),
    policy_in=init_policy(state_space, action_space),
    evaluation_func=sarsa,
    gamma=gamma,
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~300k needed to reach optimal policy on 4x4
        for step in range(10_000)
    ],
    eps_schedule=[0.2, 0.1, 0.05, 0.02, 0.01, 0.0],
    verbose=False
)
print(
    q.join(policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)
print(policy.pivot(on="action", index="state").sort("state"))

# %%
# Run learned policy
human_env = gym.make(**env_params, render_mode='human')

num_episodes = 3
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = human_env.reset()

    episode_over = False
    while not episode_over:
        action = sample_from_policy(policy, state)
        state, _, terminated, truncated, _ = human_env.step(action)
        episode_over = terminated or truncated
human_env.close()

# %%
