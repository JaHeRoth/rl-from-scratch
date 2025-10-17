# %%
from typing import Iterable

import gymnasium as gym
import numpy as np
import polars as pl
from numpy.random import Generator
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
    policy: pl.DataFrame, state: int, seed: int | None | Generator = None
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
    return (
        q.with_columns(q_max=pl.max("q").over("state"))
        .with_columns(greedy_choice=pl.col("q") == pl.col("q_max"),)
        .select(
            "state",
            "action",
            policy=(
                (1 - eps) * pl.col("greedy_choice").cast(pl.Float64)
                / pl.sum("greedy_choice").over("state")
                + eps / pl.len().over("state")
            ),
        )
    )

def policy_iteration(
    q_in: pl.DataFrame,
    policy_in: pl.DataFrame,
    evaluation_func: callable,
    gamma: float,
    lr_schedule: Iterable[float],
    eps_schedule: Iterable[float],
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
# Expected Sarsa
def expected_sarsa(
    q_in: pl.DataFrame, policy: pl.DataFrame, gamma: float, lr_schedule: Iterable[float], seed: int = 42
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    state, _ = env.reset(seed=seed)
    for step, lr in tqdm(enumerate(lr_schedule), desc="Running steps"):
        action = sample_from_policy(policy, state, seed=seed + step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        if episode_over:
            target = float(reward)
        else:
            next_action_probs = policy.filter(pl.col("state") == next_state).sort("action")["policy"].to_numpy()
            target = float(reward) + gamma * next_action_probs @ q[next_state, :]

        q[state, action] += lr * (target - q[state, action])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    return q_in.with_columns(q=q.flatten())


q, policy = policy_iteration(
    q_in=init_q(state_space, action_space),
    policy_in=init_policy(state_space, action_space),
    evaluation_func=expected_sarsa,
    gamma=gamma,
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        # ~200k needed to reach optimal policy on 4x4
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
# Q-Learning
num_steps = 40_000  # ~1M needed to reach optimal policy on 4x4

def q_learning(
    q_in: pl.DataFrame,
    gamma: float,
    eps_schedule: Iterable[float],
    lr_schedule: Iterable[float],
    seed: int = 42,
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    state, _ = env.reset(seed=seed)
    for step, (eps, lr) in tqdm(enumerate(zip(eps_schedule, lr_schedule)), desc="Running steps"):
        # We only need the behavior policy for the current state
        state_q = pl.DataFrame(
            {
                "state": state,
                "action": range(q.shape[1]),
                "q": q[state, :],
            }
        )
        behavior_policy = eps_greedy_policy(q=state_q, eps=eps)

        action = sample_from_policy(policy=behavior_policy, state=state, seed=seed + step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        target = float(reward) + gamma * q[next_state, :].max()
        q[state, action] += lr * (target - q[state, action])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    return q_in.with_columns(q=q.flatten())


q = q_learning(
    q_in=init_q(state_space, action_space),
    gamma=gamma,
    eps_schedule=np.linspace(1.0, 0.25, num_steps),
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        for step in range(num_steps)
    ],
)
policy = eps_greedy_policy(q, eps=0.0)
print(
    q.join(policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)
print(policy.pivot(on="action", index="state").sort("state"))

# %%
# Double Q-Learning
num_steps = 40_000  # ~1M needed to reach optimal policy on 4x4

def double_q_learning(
    q_in: pl.DataFrame,
    gamma: float,
    eps_schedule: Iterable[float],
    lr_schedule: Iterable[float],
    seed: int = 42,
) -> pl.DataFrame:
    q1 = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    q2 = q1.copy()
    rng = np.random.default_rng(10 ** 10 + seed)
    state, _ = env.reset(seed=seed)
    for step, (eps, lr) in tqdm(enumerate(zip(eps_schedule, lr_schedule)), desc="Running steps"):
        # We only need the behavior policy for the current state
        state_q = pl.DataFrame(
            {
                "state": state,
                "action": range(q1.shape[1]),
                "q": (q1[state, :] + q2[state, :]) / 2,
            }
        )
        behavior_policy = eps_greedy_policy(q=state_q, eps=eps)

        action = sample_from_policy(policy=behavior_policy, state=state, seed=seed + step)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        if rng.random() > 0.5:
            q_actor = q1
            q_critic = q2
        else:
            q_actor = q2
            q_critic = q1

        target = float(reward) + gamma * q_critic[next_state, q_actor[next_state, :].argmax()]
        q_actor[state, action] += lr * (target - q_actor[state, action])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    q = (q1 + q2) / 2
    return q_in.with_columns(q=q.flatten())


q = q_learning(
    q_in=init_q(state_space, action_space),
    gamma=gamma,
    eps_schedule=np.linspace(1.0, 0.25, num_steps),
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        for step in range(num_steps)
    ],
)
policy = eps_greedy_policy(q, eps=0.0)
print(
    q.join(policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)
print(policy.pivot(on="action", index="state").sort("state"))

# %%
# Dyna-Q
# ~10k * 10 needed to reach optimal policy on 4x4
num_steps = 3_000
n_samples_per_step = 10

def dyna_q(
    q_in: pl.DataFrame,
    gamma: float,
    eps_schedule: Iterable[float],
    lr_schedule: Iterable[float],
    n_samples_per_step: int,
    seed: int = 42,
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    # If we rather want every experienced state-action pair to be sampled with equal probability, then this
    #  `[(state, action, next_state, reward)]` should be replaced by `{(state, action): [(next_state, reward)]`
    #  and the below single `rng.choice` call should be replaced by two `rng.choice` calls (one to choose the
    #  state-action pair and the other to choose the experience for that state-action pair)
    experiences: list[tuple[int, int, int, float]] = []

    rng = np.random.default_rng(seed)
    state, _ = env.reset(seed=seed)
    for step, (eps, lr) in tqdm(enumerate(zip(eps_schedule, lr_schedule)), desc="Running steps"):
        # We only need the behavior policy for the current state
        state_q = pl.DataFrame(
            {
                "state": state,
                "action": range(q.shape[1]),
                "q": q[state, :],
            }
        )
        behavior_policy = eps_greedy_policy(q=state_q, eps=eps)

        action = sample_from_policy(policy=behavior_policy, state=state, seed=rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        experiences.append((state, action, next_state, reward))

        target = float(reward) + gamma * q[next_state, :].max()
        q[state, action] += lr * (target - q[state, action])

        for _ in range(n_samples_per_step):
            _state, _action, _next_state, _reward = rng.choice(experiences)
            _target = float(_reward) + gamma * q[int(_next_state), :].max()
            q[int(_state), int(_action)] += lr * (_target - q[int(_state), int(_action)])

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    return q_in.with_columns(q=q.flatten())


q = dyna_q(
    q_in=init_q(state_space, action_space),
    gamma=gamma,
    eps_schedule=np.linspace(1.0, 0.25, num_steps),
    lr_schedule=[
        learning_rate_for_update(
            base_learning_rate=1e-1,
            update_number=step,
            period=100,
            numerator_power=0.51,
        )
        for step in range(num_steps)
    ],
    n_samples_per_step=n_samples_per_step,
)
policy = eps_greedy_policy(q, eps=0.0)
print(
    q.join(policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)
print(policy.pivot(on="action", index="state").sort("state"))

# %%
# Dyna-Q with prioritized sweeping
# ~30k * 10 needed to reach optimal policy on 4x4
num_steps = 3_000
n_samples_per_step = 10

class UniquePriorityQueue:
    _priority_mapping = {}

    def put(self, item, priority):
        if item not in self._priority_mapping or self._priority_mapping[item] < priority:
            self._priority_mapping[item] = priority

    def pop(self):
        max_priority_item = max(self._priority_mapping.keys(), key=self._priority_mapping.get)
        self._priority_mapping.pop(max_priority_item)
        return max_priority_item

    def __len__(self):
        return len(self._priority_mapping)

def dyna_q_with_prioritized_sweeping(
    q_in: pl.DataFrame,
    gamma: float,
    eps_schedule: Iterable[float],
    n_samples_per_step: int,
    required_delta: float = 10 ** -10,
    seed: int = 42,
) -> pl.DataFrame:
    q = (
        q_in.sort(["state", "action"])
        .pivot(on="action", index="state")
        .drop("state")
        .to_numpy()
    )
    model = pl.DataFrame(
        schema={
            "state": pl.Int64,
            "action": pl.Int64,
            "next_state": pl.Int64,
            "reward": pl.Float64,
            "experience_count": pl.Int64,
        }
    )
    pq = UniquePriorityQueue()

    rng = np.random.default_rng(seed)
    state, _ = env.reset(seed=seed)
    for step, eps in tqdm(enumerate(eps_schedule), desc="Running steps"):
        # We only need the behavior policy for the current state
        state_q = pl.DataFrame(
            {
                "state": state,
                "action": range(q.shape[1]),
                "q": q[state, :],
            }
        )
        behavior_policy = eps_greedy_policy(q=state_q, eps=eps)

        action = sample_from_policy(policy=behavior_policy, state=state, seed=rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        transition_filter_expr = (pl.col("state") == state) & (pl.col("action") == action) & (pl.col("next_state") == next_state)
        if model.filter(transition_filter_expr).is_empty():
            model = pl.concat(
                [
                    model,
                    pl.DataFrame(
                        {
                            "state": state,
                            "action": action,
                            "next_state": next_state,
                            "reward": reward,
                            "experience_count": 1,
                        }
                    ),
                ]
            )
        else:
            model = (
                model
                .with_columns(
                    reward=(
                        pl.when(transition_filter_expr)
                        .then(
                            (pl.col("reward") * pl.col("experience_count") + reward)
                            / (pl.col("experience_count") + 1)
                        )
                        .otherwise("reward")
                    ),
                    experience_count=pl.col("experience_count") + transition_filter_expr.cast(pl.Int64)
                )
            )

        abs_error = abs(float(reward) + gamma * q[next_state, :].max() - q[state, action])
        if abs_error > required_delta:
            pq.put(item=(state, action), priority=abs_error)

        for _ in range(n_samples_per_step):
            if len(pq) == 0:
                break

            _state, _action = pq.pop()
            relevant_model = (
                model.filter(
                    (pl.col("state") == _state) & (pl.col("action") == _action)
                )
                .with_columns(
                    probability=pl.col("experience_count") / pl.sum("experience_count")
                )
            )
            _target = (
                relevant_model["probability"].to_numpy()
                @ (
                    relevant_model["reward"].to_numpy()
                    + gamma * q[relevant_model["next_state"], :].max(axis=1)
                )
            )
            # print(f"{step=}, {_step=}, {_state=}, {_action=}, q={q[_state, _action]:.3f}, {_target=}")
            q[_state, _action] = _target

            # Probably quite inefficient to jump between polars and numpy, but makes for simple code
            q_df = pl.DataFrame(
                [
                    {
                        "state": state,
                        "action": action,
                        "q": q[state, action],
                    }
                ]
            )
            for _prev_state, _prev_action, _sample_abs_error in (
                model.filter(pl.col("next_state") == _state)
                .join(q_df, on=["state", "action"])
                .select(
                    "state",
                    "action",
                    sample_abs_error=(pl.col("reward") + gamma * q[_state, _action] - pl.col("q")).abs(),
                )
                .filter(pl.col("sample_abs_error") > required_delta)
            ).iter_rows():
                # Without this if statement we just keep re-adding the same state to the priority queue, since gamma < 1
                if _prev_state != state:
                    pq.put(item=(_prev_state, _prev_action), priority=_sample_abs_error)

        if episode_over:
            state, _ = env.reset(seed=seed + step)
        else:
            state = next_state

    print(model.sort(["state", "action", "next_state"]))
    return q_in.with_columns(q=q.flatten())


q = dyna_q_with_prioritized_sweeping(
    q_in=init_q(state_space, action_space),
    gamma=gamma,
    eps_schedule=np.linspace(1.0, 0.25, num_steps),
    n_samples_per_step=n_samples_per_step,
)
policy = eps_greedy_policy(q, eps=0.0)
print(
    q.join(policy, on=["state", "action"])
    .group_by("state")
    .agg(v=(pl.col("policy") * pl.col("q")).sum())
    .sort("state")
)
print(policy.pivot(on="action", index="state").sort("state"))

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
