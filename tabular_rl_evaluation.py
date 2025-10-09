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

def learning_rate_for_update(
    base_learning_rate: float, update_number: int, period: float, numerator_power: float
) -> float:
    numerator = max(1, update_number / period) ** numerator_power
    return base_learning_rate / numerator

# %%
# On-policy Monte Carlo policy evaluation
num_episodes = 10_000
learning_rate = 1e-1
learning_rate_schedule_period = 50
learning_rate_schedule_power = 0.51
seed = 42

policy = init_policy(state_space, action_space)
print(policy.sort(["state", "action"]).pivot(on="action", index="state"))

v = init_v(state_space)
num_step = 0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset(seed=seed + episode)

    state_trajectory = [state]
    reward_trajectory = []
    episode_over = False
    while not episode_over:
        action = sample_from_policy(policy, state, seed=seed + num_step)
        state, reward, terminated, truncated, _ = env.step(action)
        reward_trajectory.append(reward)
        state_trajectory.append(state)
        episode_over = terminated or truncated
        num_step += 1
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

    lr = learning_rate_for_update(
        base_learning_rate=learning_rate,
        update_number=episode,
        period=learning_rate_schedule_period,
        numerator_power=learning_rate_schedule_power,
    )
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
# Off-policy Monte Carlo policy evaluation (through weighted importance sampling)
num_episodes = 10_000
learning_rate = 1e-1
learning_rate_schedule_period = 75
learning_rate_schedule_power = 0.51
seed = 42

target_policy = (
    pl.DataFrame(
        [
            {
                "state": state,
                "action": action,
                "policy": 0.1 + 0.6 * (action == 1),
            }
            for state in state_space
            for action in action_space
        ]
    )
)
behavior_policy = init_policy(state_space, action_space)

v = init_v(state_space)
num_step = 0
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = env.reset(seed=seed + episode)

    trajectory = []
    episode_over = False
    while not episode_over:
        action = sample_from_policy(behavior_policy, state, seed=seed + num_step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(
            {
                "step_count": len(trajectory),
                "state": state,
                "action": action,
                "reward": reward,
            }
        )
        state = next_state
        episode_over = terminated or truncated
        num_step += 1

    targets = (
        pl.LazyFrame(trajectory)
        .join(target_policy.lazy(), on=["state", "action"])
        .join(behavior_policy.lazy(), on=["state", "action"], suffix="_behavior")
        .sort("step_count")
        .with_columns(
            factor=gamma ** pl.col("step_count"),
            importance_sampling_ratio=(pl.col("policy") / pl.col("policy_behavior")).cum_prod(reverse=True),
        )
        .with_columns(remaining_return=(pl.col("reward") * pl.col("factor")).cum_sum(reverse=True) / pl.col("factor"))
        .group_by("state")
        .agg(
            target=(
                pl.when(pl.sum("importance_sampling_ratio") == 0)
                .then(0.0)
                .otherwise(
                    (pl.col("importance_sampling_ratio") * pl.col("remaining_return")).sum()
                    / pl.sum("importance_sampling_ratio")
                )
            )
        )
    )

    lr = learning_rate_for_update(
        base_learning_rate=learning_rate,
        update_number=episode,
        period=learning_rate_schedule_period,
        numerator_power=learning_rate_schedule_power,
    )
    v = (
        v.lazy()
        .join(targets, on="state", how="left")
        .select(
            "state",
            v=(
                pl.when(pl.col("target").is_not_null())
                .then((1 - learning_rate) * pl.col("v") + learning_rate * pl.col("target"))
                .otherwise(pl.col("v"))
            ),
        )
        .collect()  # Only collecting at the very end causes SIGBUS
    )

print(v.sort("state"))

# %%
# TD(0)
num_steps = 40_000
learning_rate = 1e-1
learning_rate_schedule_period = 40
learning_rate_schedule_power = 0.51
seed = 42
policy = init_policy(state_space, action_space)

v = np.zeros_like(state_space, dtype=np.float64)
state, _ = env.reset(seed=seed)
for step in tqdm(range(num_steps), desc="Running steps"):
    action = sample_from_policy(policy, state, seed=seed + step)
    next_state, reward, terminated, truncated, _ = env.step(action)
    episode_over = terminated or truncated

    td_target = float(reward) + gamma * v[next_state]
    lr = learning_rate_for_update(
        base_learning_rate=learning_rate,
        update_number=step,
        period=learning_rate_schedule_period,
        numerator_power=learning_rate_schedule_power,
    )
    v[state] += lr * (td_target - v[state])

    if episode_over:
        state, _ = env.reset(seed=seed + step)
    else:
        state = next_state

for state, ret in enumerate(v):
    print(f"{state}: {ret:.3f}")

# %%
# n-step TD-learning
n = 5
num_steps = 40_000
learning_rate = 1e-1
learning_rate_schedule_period = 50
learning_rate_schedule_power = 0.51
seed = 42
policy = init_policy(state_space, action_space)

v = np.zeros_like(state_space, dtype=np.float64)
states = []
rewards = []
state, _ = env.reset(seed=seed)
for step in tqdm(range(num_steps), desc="Running steps"):
    action = sample_from_policy(policy, state, seed=seed + step)
    next_state, reward, terminated, truncated, _ = env.step(action)
    episode_over = terminated or truncated

    states.append(state)
    rewards.append(reward)

    lr = learning_rate_for_update(
        base_learning_rate=learning_rate,
        update_number=step,
        period=learning_rate_schedule_period,
        numerator_power=learning_rate_schedule_power,
    )

    if episode_over:
        for k in reversed(range(1, min(n, len(rewards)) + 1)):
            td_target = np.array(rewards[-k:]) @ gamma ** np.arange(k)
            v[states[-k]] += lr * (td_target - v[states[-k]])

        state, _ = env.reset(seed=seed + step)
        states = []
        rewards = []
    else:
        if len(rewards) >= n:
            td_target = (
                np.array(rewards[-n:]) @ gamma ** np.arange(n)
                + gamma ** n * v[next_state]
            )
            v[states[-n]] += lr * (td_target - v[states[-n]])

        state = next_state

for state, ret in enumerate(v):
    print(f"{state}: {ret:.3f}")

# %%
