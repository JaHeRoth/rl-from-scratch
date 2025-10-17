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
# REINFORCE
# Note: We'd probably benefit from some regularization, parallelization, batching and feature normalization,
#  but omitted here for brevity and simplicity
# Note: We're kind of reinventing the wheel here by using numpy instead of pytorch with its autograd,
#  but figured it's nice to go as low-level as possible, since we're anyway dealing with very simple networks

def reinforce(
    lr_schedule: Iterable[float], gamma: float, env: Env
) -> np.ndarray:
    policy_weights = init_policy_weights(env)
    old_policy_weights = policy_weights.copy()

    for episode, lr in tqdm(enumerate(lr_schedule), desc="Running episodes"):
        state, _ = env.reset()
        episode_over = False
        total_return = 0.0
        factor = 1.0
        trajectory: list[np.ndarray] = []
        while not episode_over:
            action, log_probit_grad = (
                sample_from_policy(state, policy_weights)
            )
            trajectory.append(log_probit_grad)
            state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            factor *= gamma
            episode_over = terminated or truncated
        for step, log_probit_grad in enumerate(trajectory):
            policy_weights += lr * gamma ** step * total_return * log_probit_grad
        if (episode + 1) % 1000 == 0:
            print(f"policy_weights_delta_norm={np.linalg.norm(policy_weights - old_policy_weights)}")
            old_policy_weights = policy_weights.copy()

    return policy_weights


policy_weights = reinforce(
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-2,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~10k needed to sometimes reach half-decent policy (and sometimes not)
        for step in range(1_000)
    ],
    gamma=gamma,
    env=env,
)
print(policy_weights)

# %%
# REINFORCE with baseline
# Note: Can also use different learning rates for v_weights and policy_weights

def reinforce_with_baseline(
    lr_schedule: Iterable[float], gamma: float, env: Env
) -> np.ndarray:
    v_weights = init_v_weights(env)
    policy_weights = init_policy_weights(env)
    old_v_weights = v_weights.copy()
    old_policy_weights = policy_weights.copy()

    for episode, lr in tqdm(enumerate(lr_schedule), desc="Running episodes"):
        state, _ = env.reset()
        episode_over = False
        total_return = 0.0
        factor = 1.0
        trajectory = []
        while not episode_over:
            action, log_probit_grad = (
                sample_from_policy(state, policy_weights)
            )
            trajectory.append((state, log_probit_grad))
            state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            factor *= gamma
            episode_over = terminated or truncated
        for step, (state, log_probit_grad) in enumerate(trajectory):
            featurized_state = featurize_state(state)
            state_error = total_return - v_weights @ featurized_state
            v_weights += lr * state_error * featurized_state
            policy_weights += lr * gamma ** step * state_error * log_probit_grad
        if (episode + 1) % 1000 == 0:
            print(f"v_weights_delta_norm={np.linalg.norm(v_weights - old_v_weights)}")
            old_v_weights = v_weights.copy()
            print(f"policy_weights_delta_norm={np.linalg.norm(policy_weights - old_policy_weights)}")
            old_policy_weights = policy_weights.copy()

    return policy_weights


policy_weights = reinforce_with_baseline(
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-2,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~1k needed to sometimes reach half-decent policy (and sometimes not)
        # ~10k needed to reach near-optimal policy
        for step in range(1_000)
    ],
    gamma=gamma,
    env=env,
)
print(policy_weights)

# %%
# One-step Actor-Critic
def one_step_actor_critic(
    lr_schedule: Iterable[float], gamma: float, env: Env
) -> np.ndarray:
    v_weights = init_v_weights(env)
    policy_weights = init_policy_weights(env)
    old_v_weights = v_weights.copy()
    old_policy_weights = policy_weights.copy()

    for episode, lr in tqdm(enumerate(lr_schedule), desc="Running episodes"):
        factor = 1.0
        state, _ = env.reset()
        episode_over = False
        while not episode_over:
            action, log_probit_grad = (
                sample_from_policy(state, policy_weights)
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_over = terminated or truncated

            if episode_over:
                target = reward
            else:
                target = reward + gamma * v_weights @ featurize_state(state=next_state)
            featurized_state = featurize_state(state)
            error = target - v_weights @ featurized_state

            v_weights += lr * error * featurized_state
            policy_weights += lr * factor * error * log_probit_grad

            factor *= gamma
            state = next_state
        if (episode + 1) % 1000 == 0:
            print(f"v_weights_delta_norm={np.linalg.norm(v_weights - old_v_weights)}")
            old_v_weights = v_weights.copy()
            print(f"policy_weights_delta_norm={np.linalg.norm(policy_weights - old_policy_weights)}")
            old_policy_weights = policy_weights.copy()

    return policy_weights


policy_weights = one_step_actor_critic(
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-2,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~1k needed to seemingly learn something
        # ~10k needed to sometimes reach half-decent and other times reach near-optimal policy
        for step in range(1_000)
    ],
    gamma=gamma,
    env=env,
)
print(policy_weights)

# %%
# Actor-Critic(lambda) (i.e. with eligibility traces)
def actor_critic_lambda(
    lr_schedule: Iterable[float], trace_factor: float, gamma: float, env: Env
) -> np.ndarray:
    v_weights = init_v_weights(env)
    policy_weights = init_policy_weights(env)
    old_v_weights = v_weights.copy()
    old_policy_weights = policy_weights.copy()

    for episode, lr in tqdm(enumerate(lr_schedule), desc="Running episodes"):
        v_trace = np.zeros_like(v_weights)
        policy_trace = np.zeros_like(policy_weights)
        factor = 1.0
        state, _ = env.reset()
        episode_over = False
        while not episode_over:
            action, log_probit_grad = (
                sample_from_policy(state, policy_weights)
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_over = terminated or truncated

            if episode_over:
                target = reward
            else:
                target = reward + gamma * v_weights @ featurize_state(state=next_state)
            featurized_state = featurize_state(state)
            error = target - v_weights @ featurized_state

            v_trace = gamma * trace_factor * v_trace + featurized_state
            policy_trace = gamma * trace_factor * policy_trace + log_probit_grad
            v_weights += lr * error * v_trace
            policy_weights += lr * factor * error * policy_trace

            factor *= gamma
            state = next_state
        if (episode + 1) % 1000 == 0:
            print(f"v_weights_delta_norm={np.linalg.norm(v_weights - old_v_weights)}")
            old_v_weights = v_weights.copy()
            print(f"policy_weights_delta_norm={np.linalg.norm(policy_weights - old_policy_weights)}")
            old_policy_weights = policy_weights.copy()

    return policy_weights


policy_weights = actor_critic_lambda(
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-2,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~1k needed to sometimes reach half-decent policy (and sometimes not)
        # ~10k needed to sometimes reach half-decent and other times reach near-optimal policy
        for step in range(1_000)
    ],
    trace_factor=0.75,
    gamma=gamma,
    env=env,
)
print(policy_weights)

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
