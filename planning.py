# %%
from typing import Iterable

from tqdm import tqdm
import numpy as np
import polars as pl
import gymnasium as gym

pl.Config.set_tbl_rows(50)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
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

gamma = 0.9
state_space = model["state"].unique()
action_space = model["action"].unique()

v = model.group_by("state").agg(v=pl.lit(0.0))
policy = model.group_by(["state", "action"]).agg(policy=pl.lit(1 / len(action_space)))


def bellman_equation(
    model: pl.DataFrame, v: pl.DataFrame, policy: pl.DataFrame, state: int | None
) -> pl.DataFrame:
    if state is not None:
        model = model.filter(pl.col("state") == state)

    return (
        model
        .join(v, left_on="next_state", right_on="state")
        .join(policy, on=["state", "action"])
        .group_by("state")
        .agg(
            v=(
                pl.col("policy")
                * pl.col("probability")
                * (pl.col("reward") + gamma * pl.col("v"))
            ).sum()
        )
    )


print(f"v = {bellman_equation(model, v, policy, state=None).sort('state')}")
for state in state_space:
    print(f"v({state}) = {bellman_equation(model, v, policy, state)['v'].item()}")


# %%
# Synchronous policy evaluation using DP
required_delta = 10 ** -10

max_delta = np.inf
while max_delta >= required_delta:
    v_new = bellman_equation(model, v, policy, state=None)
    max_delta = (
        v
        .join(v_new, on="state", suffix="_new")
        .select((pl.col("v_new") - pl.col("v")).abs().max())
        .item()
    )
    v = v_new

print(v.sort('state'))

# %%
# Async policy evaluation using DP
# Note: Probably not very efficient, since polars dataframes are immutable,
#  thus whole v dataframe must be copied on every state update
required_delta = 10 ** -7

max_delta = np.inf
while max_delta >= required_delta:
    max_delta = 0.0
    for state in state_space:
        new_state_v = bellman_equation(model, v, policy, state)["v"].item()
        old_state_v = v.filter(pl.col("state") == state)["v"].item()
        max_delta = max(max_delta, np.abs(new_state_v - old_state_v))
        v = v.select(
            "state",
            v=(
                pl.when(pl.col("state") == state)
                .then(new_state_v)
                .otherwise(pl.col("v"))
            ),
        )


for state in state_space:
    print(f"v({state}) = {v.filter(pl.col('state') == state)["v"].item():.3f}")


# %%
# Policy evaluation by solving system of equations
def get_coefficients(state: int):
    try:
        rewards, next_states, _ = zip(*[model[(state, action)] for action in action_space])

        A_row = np.zeros_like(v)
        A_row[state] = 1.0
        A_row[np.array(next_states)] -= gamma * policy(state)

        b_cell = policy(state) @ np.array(rewards)
        return A_row, b_cell
    except KeyError:
        # Hit for terminal states, since we always reset upon reaching these, thus don't record r, s' for these in model
        return np.zeros_like(v), 0.0

A, b = zip(
    *[
        get_coefficients(state)
        for state in state_space
    ]
)
A = np.array(A)
b = np.array(b)

# Fails because A is singular, which happens due to identical states (all the terminal states behave identically)
# print(
#     np.linalg.solve(A, b)
# )

# %%
# Policy iteration
gamma = 0.9
state_space = list(range(16))
action_space = list(range(4))

v = np.zeros(len(state_space))
policy = lambda s: np.ones(len(action_space)) / len(action_space)


def bellman_equation(
    state: int, policy: callable, gamma: float, v: np.ndarray, action_space: Iterable, model: dict
) -> float:
    try:
        target = 0.0
        for action in action_space:
            target += policy(state)[action] * (model[(state, action)][0] + gamma * v[model[(state, action)][1]])
        return target
    except KeyError:
        return 0.0  # Nothing is recorded in model for terminal states, since we always reset upon reaching these


def policy_evaluation(
    v: np.ndarray, policy: callable, gamma: float, action_space: list, required_delta: float, model
) -> np.ndarray:
    out_v = v.copy()
    max_delta = np.inf
    while max_delta >= required_delta:
        max_delta = 0.0
        for state in state_space:
            new_state_v = bellman_equation(state, policy, gamma, out_v, action_space, model)
            max_delta = max(max_delta, np.abs(new_state_v - out_v[state]))
            out_v[state] = new_state_v
    return out_v


def policy_improvement(v: np.ndarray, gamma: float, action_space: list, model: dict):
    all_rewards = {}
    all_next_states = {}
    for state in state_space:
        try:
            all_rewards[state] = np.array([model[(state, action)][0] for action in action_space])
            all_next_states[state] = np.array([model[(state, action)][1] for action in action_space])
        except KeyError:
            all_rewards[state] = np.zeros_like(action_space)
            all_next_states[state] = np.ones_like(action_space) * state

    def greedy_policy(state: int):
        action_indices = np.arange(len(action_space))
        action_returns = all_rewards[state] + gamma * v[all_next_states[state]]
        return (action_indices == np.argmax(action_returns)).astype(float)

    return greedy_policy


while True:
    old_v = v.copy()
    v = policy_evaluation(v, policy, gamma, action_space, required_delta=10 ** -5, model=model)
    policy = policy_improvement(v, gamma, action_space, model)
    if (v == old_v).all():
        print(v)
        for state in state_space:
            print(policy(state))
        print([int(np.argmax(policy(state))) for state in state_space])
        break
