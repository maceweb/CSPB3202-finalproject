import gymnasium as gym
import torch
import time

from dqn_agent import DQN


# Create Acrobot with human rendering
env = gym.make("Acrobot-v1", render_mode="human")

# Get environment dimensions
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create the DQN
model = DQN(state_size, action_size)

# Load trained model
model.load_state_dict(
    torch.load(
        "results/dqn_model.pth",
        weights_only=True
    )
)

# Evaluation mode
model.eval()


# Watch the agent play
num_episodes = 5

for episode in range(num_episodes):

    observation, info = env.reset()

    total_reward = 0
    steps = 0

    while True:

        # Display the game
        env.render()

        # Convert state to tensor
        state = torch.tensor(
            observation,
            dtype=torch.float32
        )

        # Choose the best action
        action = model.choose_action(
            state,
            epsilon=0.0
        )

        # Take action
        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        observation = next_observation

        # Slow it down so we can watch
        time.sleep(0.02)

        if terminated or truncated:
            break

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward}, "
        f"Steps = {steps}"
    )

env.close()