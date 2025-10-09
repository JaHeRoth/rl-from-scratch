# %%
from typing import Iterable

import gymnasium as gym
import numpy as np
import polars as pl
from tqdm import tqdm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

map_name = "4x4"
env = gym.make('FrozenLake-v1', desc=None, map_name=map_name, is_slippery=True)

gamma = 0.95
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

def learning_rate_for_update(
    base_learning_rate: float, update_number: int, period: float, numerator_power: float
) -> float:
    numerator = max(1, update_number / period) ** numerator_power
    return base_learning_rate / numerator

def eps_greedy_policy(q: pl.DataFrame, eps: float) -> pl.DataFrame:
    max_q_per_state = q.group_by("state").agg(q=pl.max("q"))
    greedy_choice = (
        q.join(max_q_per_state, on="state", suffix="_max")
        .select(
            "state",
            "action",
            greedy_choice=pl.col("q") == pl.col("q_max"),
        )
    )
    # TODO: Try replacing group_by + agg + join with `over`
    counts = (
        greedy_choice
        .group_by("state")
        .agg(
            num_greedy_choices=pl.sum("greedy_choice"),
            num_choices=pl.len(),
        )
    )
    return (
        greedy_choice
        .join(counts, on="state")
        .select(
            "state",
            "action",
            policy=(
                (1 - eps) * pl.col("greedy_choice").cast(pl.Float64) / pl.col("num_greedy_choices")
                + eps / pl.col("num_choices")
            ),
        )
    )

def policy_iteration(
    q_in: pl.DataFrame,
    policy_in: pl.DataFrame,
    evaluation_func: callable,
    lr_schedule: list[float],
    eps_schedule: list[float],
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
# Run learned policy
human_env = gym.make('FrozenLake-v1', desc=None, map_name=map_name, is_slippery=True, render_mode='human')

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
