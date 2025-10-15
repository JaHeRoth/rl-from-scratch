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
    featurization = featurize(state, action, action_space=range(env.action_space.n))
    return 0.0 * featurization

def sample_from_eps_greedy_policy(
    state: np.ndarray, q_weights: np.ndarray, action_space: Iterable, eps: float,
) -> int:
    q = pl.DataFrame(
        {
            "action": action_space,
            "q": [
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
                (
                    (1 - eps)
                    * pl.col("greedy_choice").cast(pl.Float64)
                    / pl.sum("greedy_choice")
                )
                + eps / pl.len()
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
    eps_schedule: Iterable[float],
    verbose: bool,
    **kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    q_weights = init_q_weights(env)

    for eps in tqdm(eps_schedule):
        next_q_weights = evaluation_func(
            q_weights_in=q_weights, eps=eps, env=env, **kwargs
        )
        if verbose:
            print(f"{eps=:.3f}, q_weights_delta_norm={np.linalg.norm(next_q_weights - q_weights):.3f}")
        q_weights = next_q_weights

    return q_weights

# %%
# Sarsa
# Note: We'd probably benefit from some regularization and feature normalization,
#  but omitted here for brevity and simplicity

def sarsa(
    q_weights_in: np.ndarray, eps: float, gamma: float, lr_schedule: Iterable[float], env: Env
) -> np.ndarray:
    q_weights = q_weights_in.copy()
    action_space = range(env.action_space.n)

    state, _ = env.reset()
    action = sample_from_eps_greedy_policy(
        state, q_weights_in, action_space, eps
    )
    for step, lr in enumerate(lr_schedule):
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        if episode_over:
            target = float(reward)
        else:
            next_action = sample_from_eps_greedy_policy(
                next_state, q_weights_in, action_space, eps
            )
            target = (
                float(reward)
                + gamma
                * q_weights @ featurize(
                    state=next_state, action=next_action, action_space=action_space
                )
            )

        state_action_features = featurize(state, action, action_space)
        q_weights += lr * (target - q_weights @ state_action_features) * state_action_features

        if episode_over:
            state, _ = env.reset()
            action = sample_from_eps_greedy_policy(
                state, q_weights_in, action_space, eps
            )
        else:
            state = next_state
            action = next_action

    return q_weights


q_weights = policy_iteration(
    env=env,
    evaluation_func=sarsa,
    gamma=gamma,
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-2,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~10k and ~100k needed to reach strong and very strong policies
        for step in range(10_000)
    ],
    eps_schedule=np.linspace(0.5, 0.0, 30),
    verbose=True,
)
print(q_weights)

# %%
# Run learned policy
num_episodes = 3

human_env = gym.make(**env_params, render_mode='human')
action_space = range(human_env.action_space.n)

for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = human_env.reset()

    episode_over = False
    while not episode_over:
        action = sample_from_eps_greedy_policy(
            state, q _weights, action_space, eps=0.0
        )
        state, _, terminated, truncated, _ = human_env.step(action)
        episode_over = terminated or truncated
human_env.close()

# %%
