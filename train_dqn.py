import gymnasium as gym
import torch
import csv

from dqn_agent import DQN
from replay_buffer import ReplayBuffer


# Create the Acrobot environment
env = gym.make("Acrobot-v1")


# Acrobot dimensions
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# Training parameters
num_episodes = 500
batch_size = 64
buffer_capacity = 10000


# DQN parameters
gamma = 0.99
learning_rate = 0.0005


# Exploration parameters
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995


# Create the DQN
model = DQN(state_size, action_size,learning_rate=learning_rate)


# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)


# Store training results
episode_rewards = []
episode_steps = []
episode_losses = []


# =========================
# Training loop
# =========================

for episode in range(num_episodes):

    # Reset the environment
    observation, info = env.reset()

    total_reward = 0
    steps = 0
    losses = []

    while True:

        # Convert state to a tensor
        state = torch.tensor(
            observation,
            dtype=torch.float32
        )

        # Choose an action
        action = model.choose_action(
            state,
            epsilon
        )

        # Take the action
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Check if episode is finished
        done = terminated or truncated

        # Store experience in replay buffer
        replay_buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done
        )

        # Update statistics
        total_reward += reward
        steps += 1

        # Train if we have enough experiences
        if len(replay_buffer) >= batch_size:

            batch = replay_buffer.sample(batch_size)

            loss = model.learn(
                batch,
                gamma=gamma
            )

            losses.append(loss)

        # Move to the next state
        observation = next_observation

        # End episode
        if done:
            break

    # Reduce exploration over time
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Save results
    episode_rewards.append(total_reward)
    episode_steps.append(steps)

    if losses:
        average_loss = sum(losses) / len(losses)
    else:
        average_loss = 0

    episode_losses.append(average_loss)

    # Print results
    print(
        f"Episode {episode + 1}/{num_episodes} | "
        f"Reward: {total_reward:.0f} | "
        f"Steps: {steps} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {average_loss:.4f}"
    )


# Close environment
env.close()
# Save training results to a CSV file
with open("results/experiments/lr_0005/dqn_results.csv", "w", newline="") as file:

    writer = csv.writer(file)

    # Column names
    writer.writerow([
        "episode",
        "reward",
        "steps",
        "loss"
    ])

    # Write each episode's results
    for i in range(num_episodes):
        writer.writerow([
            i + 1,
            episode_rewards[i],
            episode_steps[i],
            episode_losses[i]
        ])

print("\nTraining results saved to results/dqn_results.csv")

# Save the trained DQN
torch.save(
    model.state_dict(),
    "results/experiments/lr_0005/dqn_model.pth"
)

print("Trained DQN saved to results/dqn_model.pth")