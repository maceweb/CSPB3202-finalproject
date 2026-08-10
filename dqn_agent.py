import torch
import torch.nn as nn
import torch.optim as optim

##create the neural network
class DQN(nn.Module):

    def __init__(self, state_size, action_size, learning_rate=0.001):
        super().__init__()

        self.network = nn.Sequential(
            ## 6 inpits, 128 neurons
            nn.Linear(state_size, 128),
            ## add nonlinear activation function
            nn.ReLU(),

            ##another hidden layer
            nn.Linear(128, 128),
            nn.ReLU(),

            ##produces the q values
            nn.Linear(128, action_size)
        )
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

    def forward(self, state):
        return self.network(state)

    ##this will choose the largest q value
    def choose_action(self, state, epsilon):
        """
        Choose an action using epsilon-greedy exploration.
        """

        if torch.rand(1).item() < epsilon:
            # Explore
            return torch.randint(0, 3, (1,)).item()

        else:
            # Exploit
            with torch.no_grad():
                q_values = self(state)
                return torch.argmax(q_values).item()

    def learn(self, batch, gamma=0.99):
        """
        Update the DQN using a batch of experiences.
        """

        # Separate the experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert everything into PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get Q-values for the actions that were actually taken
        current_q_values = self(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Calculate the best Q-value for the next states
        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]

            # If the episode ended, there is no future reward
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Calculate the difference between our prediction and target
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Clear old gradients
        self.optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update the network weights
        self.optimizer.step()

        return loss.item()
##test
'''if __name__ == "__main__":

    import gymnasium as gym

    # Create Acrobot
    env = gym.make("Acrobot-v1")

    # Get the initial state
    observation, info = env.reset()

    print("Acrobot observation:")
    print(observation)

    # Create the DQN
    state_size = 6
    action_size = 3

    model = DQN(state_size, action_size)

    # Convert the observation into a PyTorch tensor
    state = torch.tensor(observation, dtype=torch.float32)

    # Give the state to the DQN
    q_values = model(state)

    print("\nQ-values:")
    print(q_values)

    # Choose the action with the highest Q-value
    action = torch.argmax(q_values).item()

    print("\nChosen action:")
    print(action)

    env.close() '''

##test
if __name__ == "__main__":

    import gymnasium as gym

    # Create Acrobot
    env = gym.make("Acrobot-v1")

    # Get an initial state
    observation, info = env.reset()

    # Convert observation to a PyTorch tensor
    state = torch.tensor(observation, dtype=torch.float32)

    # Create the DQN
    state_size = 6
    action_size = 3
    model = DQN(state_size, action_size)

    # Test different epsilon values
    for epsilon in [1.0, 0.5, 0.0]:

        print(f"\nEpsilon = {epsilon}")

        actions = []

        # Choose 10 actions
        for i in range(10):
            action = model.choose_action(state, epsilon)
            actions.append(action)

        print("Actions:", actions)

    # -----------------------------
    # Test the learning function
    # -----------------------------

    print("\n===== Testing Learning =====")

    # Create a small batch of fake experiences
    batch = [
        (
            observation,
            0,
            -1.0,
            observation,
            False
        ),
        (
            observation,
            1,
            -1.0,
            observation,
            False
        ),
        (
            observation,
            2,
            -1.0,
            observation,
            False
        )
    ]

    # Get Q-values before learning
    state = torch.tensor(observation, dtype=torch.float32)

    print("\nQ-values before learning:")
    print(model(state))

    # Train on the fake experiences
    loss = model.learn(batch)

    print("\nLoss:")
    print(loss)

    # Get Q-values after learning
    print("\nQ-values after learning:")
    print(model(state))
    env.close()

## acrobat is giving the dqn 6 numbers
## it processes those numbers and gives us three numbers
## then torch.argmax(q_values) finds the largest q value
## the q values don't mean anything yet (basically random) because we haven't trained them
