import gymnasium as gym
import torch

from dqn_agent import DQN


# Create Acrobot
env = gym.make("Acrobot-v1")

# Get environment dimensions
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# Create the DQN
model = DQN(state_size, action_size)

# Load the trained model
model.load_state_dict(
    torch.load(
        "results/experiments/epsilon_decay/dqn_model.pth",
        weights_only=True
    )
)
print("Model loaded successfully.")

print("\nTest Q-values:")

test_observation, info = env.reset()

test_state = torch.tensor(
    test_observation,
    dtype=torch.float32
)

with torch.no_grad():
    print(model(test_state))

# Put the model in evaluation mode
model.eval()


# Number of evaluation episodes
num_episodes = 100

rewards = []
steps_list = []
successes = 0


# Evaluate the agent
for episode in range(num_episodes):

    observation, info = env.reset()

    total_reward = 0
    steps = 0

    while True:

        # Convert state to tensor
        state = torch.tensor(
            observation,
            dtype=torch.float32
        )

        # Choose the best action
        # epsilon = 0 means no random exploration
        action = model.choose_action(
            state,
            epsilon=0.0
        )

        # Take action
        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        observation = next_observation

        # Check if episode ended
        if terminated or truncated:

            # Acrobot gives reward 0 when it reaches the goal
            if terminated:
                successes += 1

            break

    rewards.append(total_reward)
    steps_list.append(steps)


# Calculate results
average_reward = sum(rewards) / len(rewards)
average_steps = sum(steps_list) / len(steps_list)
success_rate = successes / num_episodes * 100


print("\n===== DQN Evaluation Results =====")
print(f"Average reward: {average_reward:.2f}")
print(f"Average steps: {average_steps:.2f}")
print(f"Success rate: {success_rate:.2f}%")

env.close()