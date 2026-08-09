import gymnasium as gym

#creating environment
env = gym.make("Acrobot-v1", render_mode="human")

#resetting environment
observation, info = env.reset()

print("Initial observation:")
print(observation)

print("\nObservation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

#random action test
action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

print("\nAfter one action:")
print("Action:", action)
print("Observation:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()

#confirmed I am able to launch environment