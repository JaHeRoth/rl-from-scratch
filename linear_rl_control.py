# %%
from typing import Iterable

import gymnasium as gym
from gymnasium import Env
import numpy as np
import polars as pl
from numpy.random import Generator
from tqdm import tqdm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(30)

env_params = dict(id="CartPole-v1")
gamma = 0.95

env = gym.make(**env_params)

def featurize(state: np.ndarray, action, action_space: Iterable) -> np.ndarray:
    """:return Polynomial featurization of state-action pair"""
    state_base_features = np.concat(
        [
            np.ones(1),
            state,
        ]
    )
    action_base_features = np.concat(
        [
            np.ones(1),
            (np.array(action_space)[:-1] == action).astype(float),  # `:-1` to avoid overparametrization
        ]
    )
    state_features = np.outer(state_base_features, state_base_features).flatten()
    state_action_features = np.outer(state_features, action_base_features).flatten()
    return state_action_features

def init_q_weights(env: Env) -> np.ndarray:
    state = env.observation_space.sample()
    action = env.action_space.sample()
    featurization = featurize(state, action, env.action_space.n)
    return 0.0 * featurization

def sample_from_eps_greedy_policy(
    state: np.ndarray, q_weights: np.ndarray, action_space: Iterable, eps: float,
) -> int:
    q = pl.DataFrame(
        {
            "action": action_space,
            "policy": [
                q_weights @ featurize(state, action, action_space)
                for action in action_space
            ],
        }
    )
    policy = (
        q.with_columns(greedy_choice=pl.col("q") == pl.max("q"))
        .select(
            "action",
            policy=(
                    (1 - eps) * pl.col("greedy_choice").cast(pl.Float64)
                    / pl.sum("greedy_choice").over("state")
                    + eps / pl.len().over("state")
            ),
        )
    )
    action = np.random.default_rng().choice(
        policy["action"], p=policy["policy"]
    )
    return int(action)

def learning_rate_for_update(
    base_learning_rate: float, update_number: int, period: float, numerator_power: float
) -> float:
    numerator = max(1, update_number / period) ** numerator_power
    return base_learning_rate / numerator

def policy_iteration(
    env: Env,
    evaluation_func: callable,
    gamma: float,
    lr_schedule: Iterable[float],
    eps_schedule: Iterable[float],
    required_delta: float,
    verbose: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    q_weights = init_q_weights(env)

    for eps in eps_schedule:
        q_weights_delta_norm = np.inf
        while q_weights_delta_norm >= required_delta:
            next_q_weights = evaluation_func(
                q_weights_in=q_weights, eps=eps, gamma=gamma, lr_schedule=lr_schedule, action_space=env.action_space
            )
            q_weights_delta_norm = np.linalg.norm(next_q_weights - q_weights)
            q_weights = next_q_weights
            if verbose:
                print(q_weights)

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
