# %%
from typing import Iterable

import gymnasium as gym
import numpy as np
import polars as pl
from tqdm import tqdm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

gamma = 0.95
state_space = list(range(env.observation_space.n))
action_space = list(range(env.action_space.n))

def init_v(state_space: Iterable) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "state": state_space,
            "v": 0.0,
        }
    )

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

def init_target_policy(state_space: Iterable, action_space: Iterable) -> pl.DataFrame:
    return (
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

def init_behavior_policy(state_space: Iterable, action_space: Iterable) -> pl.DataFrame:
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
# Off-policy Monte Carlo policy evaluation (through weighted importance sampling)
# Note: Haven't achieved converge on 4x4 FrozenLake, unsure if due to bug, bias or extremely slow convergence
num_episodes = 1_000
learning_rate = 1e-1
learning_rate_schedule_period = 100
learning_rate_schedule_power = 0.51
seed = 42

target_policy = init_target_policy(state_space, action_space)
behavior_policy = init_behavior_policy(state_space, action_space)

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
                .then((1 - lr) * pl.col("v") + lr * pl.col("target"))
                .otherwise(pl.col("v"))
            ),
        )
        .collect()  # Only collecting at the very end causes SIGBUS
    )

print(v.sort("state"))

# %%
# Expected Sarsa
num_steps = 100_000  # ~1M needed for convergence on 4x4

def expected_sarsa(
    q_in: pl.DataFrame,
    target_policy: pl.DataFrame,
    behavior_policy: pl.DataFrame,
    gamma: float,
    lr_schedule: list[float],
    seed: int = 42,
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    target_policy_probs = (
        target_policy.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    state, _ = env.reset(seed=seed)
    for step, lr in tqdm(enumerate(lr_schedule), desc="Running steps"):
        action = sample_from_policy(behavior_policy, state, seed=seed + step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        if episode_over:
            target = float(reward)
        else:
            target = float(reward) + gamma * target_policy_probs[next_state, :] @ q[next_state, :]

        q[state, action] += lr * (target - q[state, action])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    return q_in.with_columns(q=q.flatten())


target_policy = init_target_policy(state_space, action_space)
q = expected_sarsa(
    q_in=init_q(state_space, action_space),
    target_policy=target_policy,
    behavior_policy=init_behavior_policy(state_space, action_space),
    gamma=gamma,
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=300,
            numerator_power=0.51,
        )
        for step in range(num_steps)
    ],
)
print(
    q.join(target_policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)

# %%
# n-step Tree Backup
# Quite a bit of duplicate computation going on here, so should be possible to speed up quite a bit
num_steps = 40_000  # ~1M needed for convergence on 4x4

def n_step_tree_backup_update(trajectory, gamma, target_policy_probs, q, lr, n):
    target = trajectory[-n]["reward"]
    factor = gamma
    for k in range(1, n):
        target += factor * trajectory[-n + k]["target_contribution"]
        factor *= gamma * trajectory[-n + k]["realized_action_prob"]
    target += (
        factor * target_policy_probs[trajectory[-1]["next_state"], :] @ q[trajectory[-1]["next_state"], :]
    )
    q[trajectory[-n]["state"], trajectory[-n]["action"]] += (
        lr * (target - q[trajectory[-n]["state"], trajectory[-n]["action"]])
    )

def n_step_tree_backup(
    q_in: pl.DataFrame,
    target_policy: pl.DataFrame,
    behavior_policy: pl.DataFrame,
    gamma: float,
    n: int,
    lr_schedule: list[float],
    seed: int = 42,
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    target_policy_probs = (
        target_policy.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )

    trajectory = []
    state, _ = env.reset(seed=seed)
    for step, lr in tqdm(enumerate(lr_schedule), desc="Running steps"):
        action = sample_from_policy(behavior_policy, state, seed=seed + step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        trajectory.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "realized_action_prob": target_policy_probs[state, action],
                "target_contribution": (
                    target_policy_probs[state, :] @ q[state, :]
                    + target_policy_probs[state, action] * (float(reward) - q[state, action])
                ),
            }
        )

        if episode_over:
            for k in reversed(range(1, min(n, len(trajectory)) + 1)):
                n_step_tree_backup_update(trajectory, gamma, target_policy_probs, q, lr, n=k)

            state, _ = env.reset(seed=seed + step)
            trajectory = []
        else:
            if len(trajectory) >= n:
                n_step_tree_backup_update(trajectory, gamma, target_policy_probs, q, lr, n=n)

            state = next_state

    return q_in.with_columns(q=q.flatten())


target_policy = init_target_policy(state_space, action_space)
q = n_step_tree_backup(
    q_in=init_q(state_space, action_space),
    target_policy=target_policy,
    behavior_policy=init_behavior_policy(state_space, action_space),
    gamma=gamma,
    n=3,
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=300,
            numerator_power=0.51,
        )
        for step in range(num_steps)
    ],
)
print(
    q.join(target_policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)

# %%
