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
    state_size = 6
    action_size = 3

    model = DQN(state_size, action_size)

    print(model)