# %%
from typing import Iterable

import gymnasium as gym
import numpy as np
import polars as pl
from tqdm import tqdm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

gamma = 0.9
state_space = list(range(env.observation_space.n))
action_space = list(range(env.action_space.n))

def init_v(state_space: Iterable) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "state": state_space,
            "v": 0.0,
        }
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

# %%
# On-policy Monte Carlo policy evaluation
def learning_rate_for_update(base_learning_rate: float, update_number: int) -> float:
    numerator = max(1, update_number // 2500) ** 0.51
    return base_learning_rate / numerator

num_episodes = 40_000
learning_rate = 1e-2
seed = 42

policy = init_policy(state_space, action_space)
print(policy.sort(["state", "action"]).pivot(on="action", index="state"))

v = init_v(state_space)
num_update = 0
num_step = 0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset()

    state_trajectory = [state]
    reward_trajectory = []
    episode_over = False
    while not episode_over:
        num_step += 1
        action = sample_from_policy(policy, state, seed=seed + num_step)
        state, reward, terminated, truncated, _ = env.step(action)
        reward_trajectory.append(reward)
        state_trajectory.append(state)
        episode_over = terminated or truncated
    state_trajectory.pop()

    trajectory = (
        pl.DataFrame(
            {
                "step_count": range(len(state_trajectory)),
                "state": state_trajectory,
                "reward": reward_trajectory,
            }
        )
        .with_columns(factor=gamma ** pl.col("step_count"))
        .with_columns(remaining_return=(pl.col("reward") * pl.col("factor")).cum_sum(reverse=True) / pl.col("factor"))
    )
    targets = trajectory.group_by("state").agg(target=pl.mean("remaining_return"))

    num_update += 1
    lr = learning_rate_for_update(base_learning_rate=learning_rate, update_number=num_update)
    v = (
        v.join(targets, on="state", how="left")
        .select(
            "state",
            v=(
                pl.when(pl.col("target").is_not_null())
                .then((1 - learning_rate) * pl.col("v") + learning_rate * pl.col("target"))
                .otherwise(pl.col("v"))
            ),
        )
    )

print(v.sort("state"))


# %%
