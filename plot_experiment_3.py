import pandas as pd
import matplotlib.pyplot as plt

# Load Experiment 3 results
df = pd.read_csv(
    "results/experiments/more_episodes/dqn_results.csv"
)

# Plot reward over episodes
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["reward"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Acrobot DQN - Experiment 3 Reward")
plt.grid(True)

# Save the graph
plt.savefig(
    "results/experiments/more_episodes/reward_curve.png"
)

plt.show()