import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):
        #max number of experiences to store
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        #store one experience
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        #randomly select a batch of experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        #return the number of experiences currently stored
        return len(self.buffer)


##test

if __name__ == "__main__":

    #create a buffer that can hold 5 experiences
    buffer = ReplayBuffer(5)

    #add some test experiences
    for i in range(5):
        buffer.add(
            state=i,
            action=i % 3,
            reward=-1,
            next_state=i + 1,
            done=False
        )

    print("Buffer size:", len(buffer))

    #Sample 2 random experiences
    batch = buffer.sample(2)

    print("\nRandom batch:")
    for experience in batch:
        print(experience)
