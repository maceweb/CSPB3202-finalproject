import torch
import torch.nn as nn

##create the neural network
class DQN(nn.Module):

    def __init__(self, state_size, action_size):
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

    def forward(self, state):
        return self.network(state)

    ##this will choose the largest q value

##test
if __name__ == "__main__":

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

    env.close()

## acrobat is giving the dqn 6 numbers
## it processes those numbers and gives us three numbers
## then torch.argmax(q_values) finds the largest q value
## the q values don't mean anything yet (basically random) because we haven't trained them
