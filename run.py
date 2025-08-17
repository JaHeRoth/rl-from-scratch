from tqdm import tqdm
import gymnasium as gym
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

num_episodes = 10
for episode in tqdm(range(num_episodes), desc="Running episodes"):
    observation, info = env.reset()

    episode_over = False
    total_reward = 0
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
env.close()