import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/experiments/more_episodes/dqn_results.csv")


# =========================
# Reward vs Episode
# =========================

plt.figure()

plt.plot(df["episode"], df["reward"])

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Reward During Training")

plt.savefig("results/baseline_reward_plot.png")
plt.show()


# =========================
# Steps vs Episode
# =========================

plt.figure()

plt.plot(df["episode"], df["steps"])

plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("DQN Steps During Training")

plt.savefig("results/baseline_steps_plot.png")
plt.show()


# =========================
# Loss vs Episode
# =========================

plt.figure()

plt.plot(df["episode"], df["loss"])

plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("DQN Loss During Training")

plt.savefig("results/baseline_loss_plot.png")
plt.show()