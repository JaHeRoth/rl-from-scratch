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

def featurize_state(state: np.ndarray) -> np.ndarray:
    """:return Polynomial featurization of state"""
    state_base_features = np.concat(
        [
            np.ones(1),
            state,
        ]
    )
    state_features = np.outer(state_base_features, state_base_features).flatten()
    return state_features

def featurize_state_action(state: np.ndarray, action: int, n_actions: int) -> np.ndarray:
    """:return Polynomial featurization of state-action pair"""
    state_features = featurize_state(state)
    action_base_features = np.concat(
        [
            np.ones(1),
            (np.arange(n_actions)[:-1] == action).astype(float),  # `:-1` to avoid overparametrization
        ]
    )
    state_action_features = np.outer(state_features, action_base_features).flatten()
    return state_action_features

def init_v_weights(env: Env) -> np.ndarray:
    state = env.observation_space.sample()
    featurization = featurize_state(state)
    return np.zeros_like(featurization)

def init_policy_weights(env: Env) -> np.ndarray:
    state = env.observation_space.sample()
    action = env.action_space.sample()
    featurization = featurize_state_action(state, action, n_actions=env.action_space.n)
    return np.zeros((env.action_space.n, len(featurization)))

def sample_from_policy(
    state: np.ndarray, policy_weights: np.ndarray
) -> tuple[int, np.ndarray]:
    """:return: tuple(action, log_probit_grad)"""  # grad(ln(pi(a|s,theta)))
    n_actions = len(policy_weights)
    feature_matrix = np.array(
        [
            featurize_state_action(state, action, n_actions)
            for action in range(n_actions)
        ]
    )
    policy_logits = np.array(
        [
            policy_weights[action, :] @ feature_matrix[action, :]
            for action in range(n_actions)
        ]
    )
    policy_probits = (  # Numerically stable softmax
        np.exp(policy_logits - policy_logits.max())
        / np.exp(policy_logits - policy_logits.max()).sum()
    )
    action = np.random.default_rng().choice(
        n_actions, p=policy_probits
    )
    log_probit_grad = (
        feature_matrix[action, :]
        - sum(
            policy_probits[_action] * feature_matrix[_action, :] for _action in range(n_actions)
        )
    )
    return int(action), log_probit_grad

def learning_rate_for_update(
    base_learning_rate: float, update_number: int, period: float, numerator_power: float
) -> float:
    numerator = max(1, update_number / period) ** numerator_power
    return base_learning_rate / numerator

# %%
# Run learned policy
num_episodes = 3

human_env = gym.make(**env_params, render_mode='human')
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    state, _ = human_env.reset()

    episode_over = False
    while not episode_over:
        action, _ = sample_from_policy(state, policy_weights)
        state, _, terminated, truncated, _ = human_env.step(action)
        episode_over = terminated or truncated
human_env.close()

# %%
