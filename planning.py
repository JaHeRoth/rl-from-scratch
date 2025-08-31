# %%
import numpy as np
from tqdm import tqdm

model = {
    (0, 0): (0.0, 0, False), (0, 1): (0.0, 4, False), (0, 2): (0.0, 1, False), (0, 3): (0.0, 0, False),
    (1, 0): (0.0, 0, False), (1, 1): (0.0, 5, True), (1, 2): (0.0, 2, False), (1, 3): (0.0, 1, False),
    (2, 0): (0.0, 1, False), (2, 1): (0.0, 6, False), (2, 2): (0.0, 3, False), (2, 3): (0.0, 2, False),
    (3, 0): (0.0, 2, False), (3, 1): (0.0, 7, True), (3, 2): (0.0, 3, False), (3, 3): (0.0, 3, False),
    (4, 0): (0.0, 4, False), (4, 1): (0.0, 8, False), (4, 2): (0.0, 5, True), (4, 3): (0.0, 0, False),
    (6, 0): (0.0, 5, True), (6, 1): (0.0, 10, False), (6, 2): (0.0, 7, True), (6, 3): (0.0, 2, False),
    (8, 0): (0.0, 8, False), (8, 1): (0.0, 12, True), (8, 2): (0.0, 9, False), (8, 3): (0.0, 4, False),
    (9, 0): (0.0, 8, False), (9, 1): (0.0, 13, False), (9, 2): (0.0, 10, False), (9, 3): (0.0, 5, True),
    (10, 0): (0.0, 9, False), (10, 1): (0.0, 14, False), (10, 2): (0.0, 11, True), (10, 3): (0.0, 6, False),
    (13, 0): (0.0, 12, True), (13, 1): (0.0, 13, False), (13, 2): (0.0, 14, False), (13, 3): (0.0, 9, False),
    (14, 0): (0.0, 13, False), (14, 1): (0.0, 14, False), (14, 2): (1.0, 15, True), (14, 3): (0.0, 10, False),
}


gamma = 0.9
state_space = list(range(16))
action_space = list(range(4))

v = np.zeros(len(state_space))
policy = lambda s: np.ones(len(action_space)) / len(action_space)


def bellman_equation(state: int):
    try:
        target = 0.0
        for action in action_space:
            target += policy(state)[action] * (model[(state, action)][0] + gamma * v[model[(state, action)][1]])
        return target
    except KeyError:
        return 0.0  # Nothing is recorded in model for terminal states, since we always reset upon reaching these


for state in state_space:
    print(f"v({state}) = {bellman_equation(state)}")


# %%
# (Async) policy evaluation using DP
required_delta = 10 ** -5

max_delta = np.inf
while max_delta >= required_delta:
    max_delta = 0.0
    for state in state_space:
        new_state_v = bellman_equation(state)
        max_delta = max(max_delta, np.abs(new_state_v - v[state]))
        v[state] = new_state_v


for state in state_space:
    print(f"v({state}) = {v[state]:.3f}")


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
