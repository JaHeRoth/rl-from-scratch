# %%
from typing import Iterable

from tqdm import tqdm
import numpy as np
import polars as pl
import gymnasium as gym

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

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
    model: pl.DataFrame, v: pl.DataFrame, policy: pl.DataFrame, gamma: float, state: int | None
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


print(f"v = {bellman_equation(model, v, policy, gamma, state=None).sort('state')}")
for state in state_space:
    print(f"v({state}) = {bellman_equation(model, v, policy, gamma, state)['v'].item()}")


# %%
# Synchronous policy evaluation using DP
required_delta = 10 ** -10

v = model.group_by("state").agg(v=pl.lit(0.0))
max_delta = np.inf
num_sweeps = 0
while max_delta >= required_delta:
    num_sweeps += 1
    v_new = bellman_equation(model, v, policy, gamma, state=None)
    max_delta = (
        v
        .join(v_new, on="state", suffix="_new")
        .select((pl.col("v_new") - pl.col("v")).abs().max())
        .item()
    )
    v = v_new

print(f"After {num_sweeps} sweeps:")
print(v.sort('state'))

# %%
# Async policy evaluation using DP
# Note: Probably not very efficient, since polars dataframes are immutable,
#  thus whole v dataframe must be copied on every state update
required_delta = 10 ** -10

v = model.group_by("state").agg(v=pl.lit(0.0))
max_delta = np.inf
num_sweeps = 0
while max_delta >= required_delta:
    num_sweeps += 1
    max_delta = 0.0
    for state in state_space:
        new_state_v = bellman_equation(model, v, policy, gamma, state)["v"].item()
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

print(f"After {num_sweeps} sweeps:")
for state in state_space:
    print(f"v({state}) = {v.filter(pl.col('state') == state)["v"].item():.3f}")


# %%
# Policy evaluation by solving system of equations
def get_coefficients():
    joined = model.join(policy, on=["state", "action"])
    v_multiplier = (
        joined
        .group_by(["state", "next_state"])
        .agg(coeff=-gamma * (pl.col("policy") * pl.col("probability")).sum())
        .sort("next_state")
        .pivot(on="next_state", index="state", values="coeff")
        .sort("state")
        .fill_null(0.0)
    )
    expected_reward = (
        joined
        .group_by("state")
        .agg(expected_reward=(pl.col("policy") * pl.col("probability") * pl.col("reward")).sum())
        .sort("state")
    )

    return (
        v_multiplier.drop("state").to_numpy() + np.eye(len(state_space)),
        expected_reward.drop("state").to_numpy(),
    )


A, b = get_coefficients()
true_v = np.linalg.solve(A, b)

for i, state in enumerate(state_space):
    print(f"v({state}) = {true_v[i, 0]:.3f}")

# %%
# Policy iteration
v = model.group_by("state").agg(v=pl.lit(0.0))
policy = model.group_by(["state", "action"]).agg(policy=pl.lit(1 / len(action_space)))


def policy_evaluation(
    model: pl.DataFrame, v: pl.DataFrame, policy: pl.DataFrame, gamma: float, required_delta: float
) -> pl.DataFrame:
    max_delta = np.inf
    while max_delta >= required_delta:
        v_new = bellman_equation(model, v, policy, gamma, state=None)
        max_delta = (
            v
            .join(v_new, on="state", suffix="_new")
            .select((pl.col("v_new") - pl.col("v")).abs().max())
            .item()
        )
        v = v_new
    return v


def policy_improvement(model: pl.DataFrame, v: pl.DataFrame, gamma: float) -> pl.DataFrame:
    q = (
        model
        .join(v, left_on="next_state", right_on="state")
        .group_by(["state", "action"])
        .agg(
            q=(
                pl.col("probability")
                * (pl.col("reward") + gamma * pl.col("v"))
            ).sum()
        )
    )
    max_q_per_state = (
        q
        .group_by("state")
        .agg(q=pl.max("q"))
    )
    greedy_choice = (
        q
        .join(max_q_per_state, on="state", suffix="_max")
        .select(
            "state",
            "action",
            greedy_choice=pl.col("q") == pl.col("q_max"),
        )
    )
    num_greedy_choices = (
        greedy_choice
        .group_by("state")
        .agg(num_greedy_choices=pl.sum("greedy_choice"))
    )
    return (
        greedy_choice
        .join(num_greedy_choices, on="state")
        .select(
            "state",
            "action",
            policy=pl.col("greedy_choice").cast(pl.Float64) / pl.col("num_greedy_choices"),
        )
    )


while True:
    old_policy = policy

    v = policy_evaluation(model, v, policy, gamma, required_delta=10 ** -5)
    policy = policy_improvement(model, v, gamma)

    policy_deltas = (
        policy
        .join(old_policy, on=["state", "action"], suffix="_old")
        .filter(pl.col("policy") != pl.col("policy_old"))
    )
    if len(policy_deltas) == 0:
        break

print(v.sort("state"))
print(policy.sort(["state", "action"]).pivot(on="action", index="state"))

# %%
# (Synchronous) value iteration
def bellman_optimality_equation(
    model: pl.DataFrame, v: pl.DataFrame, gamma: float, state: int | None
) -> pl.DataFrame:
    if state is not None:
        model = model.filter(pl.col("state") == state)

    return (
        model
        .join(v, left_on="next_state", right_on="state")
        .group_by(["state", "action"])
        .agg(
            q=(
                pl.col("probability")
                * (pl.col("reward") + gamma * pl.col("v"))
            ).sum()
        )
        .group_by("state")
        .agg(
            v=pl.max("q")
        )
    )


required_delta = 10 ** -10

v = model.group_by("state").agg(v=pl.lit(0.0))
max_delta = np.inf
num_sweeps = 0
while max_delta >= required_delta:
    num_sweeps += 1
    v_new = bellman_optimality_equation(model, v, gamma, state=None)
    max_delta = (
        v
        .join(v_new, on="state", suffix="_new")
        .select((pl.col("v_new") - pl.col("v")).abs().max())
        .item()
    )
    v = v_new

print(f"After {num_sweeps} sweeps:")
print(v.sort('state'))

# %%
