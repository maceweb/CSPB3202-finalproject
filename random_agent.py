import gymnasium as gym

# Create the Acrobot environment
env = gym.make("Acrobot-v1")

# Number of episodes
num_episodes = 100

# Store results
episode_rewards = []
episode_steps = []
episode_successes = []

# Run the random agent
for episode in range(num_episodes):

    # Start a new episode
    observation, info = env.reset()

    total_reward = 0
    steps = 0

    while True:

        # Randomly choose one of the three actions
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)

        # Record reward and step count
        total_reward += reward
        steps += 1

        # Check whether the episode is over
        if terminated or truncated:
            break

    # An episode is successful if it terminated naturally
    success = terminated

    # Save this episode's results
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    episode_successes.append(success)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward}, "
        f"Steps = {steps}, "
        f"Success = {success}"
    )

# Calculate averages
average_reward = sum(episode_rewards) / num_episodes
average_steps = sum(episode_steps) / num_episodes
success_rate = sum(episode_successes) / num_episodes * 100

print("\n===== Random Agent Results =====")
print(f"Average reward: {average_reward:.2f}")
print(f"Average steps: {average_steps:.2f}")
print(f"Success rate: {success_rate:.2f}%")

env.close()