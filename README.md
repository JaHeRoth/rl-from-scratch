# rl-from-scratch
This repository contains my implementations of most of the algorithms presented in the second edition of
Sutton & Barto's seminal book [_Reinforcement Learning: An Introduction_](http://incompleteideas.net/book/RLbook2020.pdf).

The motivation for this was a desire to deepen my understanding of these algorithms, and to display how simply many of
them can be implemented, in hopes of inspiring others.

This also means that my implementations prioritize simplicity and readability over maximizing computational efficiency.
Specifically, the choice of data representation is in many places suboptimal (e.g. using immutable tables), memoization
is in many place omitted, the hyperparameters haven't been carefully tuned, and classic policy approximation tricks
such as regularization and normalization have been omitted.

## Algorithms

### bandits.py
Algorithms for action selection seeking to optimizing cumulative reward in the multiarmed bandit problem:
- Greedy
- Greedy with optimistic initialization
- Epsilon greedy
- Upper confidence bound (UCB)
- Gradient bandit

### planning.py
Algorithms for policy evaluation and policy optimization when the environment model is known and tractable:
- Synchronous policy evaluation using dynamic programming
- Asynchronous policy evaluation using dynamic programming
- Policy evaluation by solving system of linear equations
- Policy iteration (with synchronous policy evaluation using dynamic programming)
- Synchronous value iteration
- Real-time value iteration
- Value iteration with prioritized sweeping

### tabular_rl_on_policy_evaluation.py
Algorithms for evaluating a policy when the environment model is unknown but the state space is known and tractable:
and explored by following the policy:
- Monte Carlo
- TD(0)
- n-step TD
- TD(lambda) with Dutch traces (known in book as _True Online TD(lambda)_)
- Sarsa

### tabular_rl_off_policy_evaluation.py
Algorithms for evaluating a policy when the environment model is unknown but the state space is known and tractable:
and explored by following another policy:
- Monte Carlo with weighted importance sampling
- Expected Sarsa
- n-step tree backup

### tabular_rl_control.py
Algorithms for finding an optimal policy when the environment model is unknown but the state space is known and
tractable:
- Policy iteration with Sarsa
- Policy iteration with Expected Sarsa
- Q-learning
- Double Q-learning
- Dyna-Q
- Dyna-Q with prioritized sweeping

### decision_time_planning.py
Algorithms for at decision time using a known environment model to improve upon the action selections of a suboptimal
policy:
- Monte Carlo rollout
- Monte Carlo tree search (MCTS)

### linear_rl_control.py
Algorithms for finding a near-optimal policy when the environment model is unknown and the state space is intractable:
- Policy iteration with Sarsa with linear action-value function approximation
- Policy iteration with Sarsa(lambda) with linear action-value function approximation

### policy_gradient.py
Algorithms for directly finding a near-optimal policy when the environment model is unknown and the state space is
intractable:
- REINFORCE
- REINFORCE with baseline
- One-step Actor-Critic
- Actor-Critic with additive eligibility traces