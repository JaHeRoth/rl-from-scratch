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

def init_v(state_space: Iterable):
    return pl.DataFrame(
        {
            "state": state_space,
            "v": 0.0,
        }
    )

def init_policy(state_space: Iterable, action_space: Iterable):
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
