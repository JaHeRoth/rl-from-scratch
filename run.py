import gymnasium as gym
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

observation, info = env.reset()
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()
    print(f"Taking action: {action}")

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"New observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()