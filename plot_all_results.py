import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# File locations
# ==========================================

baseline_file = "results/experiments/baseline/dqn_results.csv"
lr_file = "results/experiments/lr_0005/dqn_results.csv"
epsilon_file = "results/experiments/epsilon_099/dqn_results.csv"
episodes_file = "results/experiments/more_episodes/dqn_results.csv"

# ==========================================
# Load results
# ==========================================

baseline = pd.read_csv(baseline_file)
lr = pd.read_csv(lr_file)
epsilon = pd.read_csv(epsilon_file)
episodes = pd.read_csv(episodes_file)

# ==========================================
# Create output folders
# ==========================================

os.makedirs("results/graphs", exist_ok=True)

# ==========================================
# 1. Baseline Reward
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    baseline["episode"],
    baseline["reward"]
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Baseline DQN Reward")
plt.grid(True)

plt.savefig(
    "results/graphs/baseline_reward.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 2. Learning Rate Experiment
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    lr["episode"],
    lr["reward"]
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Experiment 1: Learning Rate = 0.0005")
plt.grid(True)

plt.savefig(
    "results/graphs/experiment1_learning_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 3. Epsilon Decay Experiment
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    epsilon["episode"],
    epsilon["reward"]
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Experiment 2: Epsilon Decay = 0.99")
plt.grid(True)

plt.savefig(
    "results/graphs/experiment2_epsilon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 4. 1000 Episode Experiment
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    episodes["episode"],
    episodes["reward"],
    alpha=0.3,
    label="Episode reward"
)

# 50-episode moving average
episodes["moving_average"] = (
    episodes["reward"].rolling(50).mean()
)

plt.plot(
    episodes["episode"],
    episodes["moving_average"],
    linewidth=2,
    label="50-episode moving average"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Experiment 3: 1000 Episodes")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/graphs/experiment3_1000_episodes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 5. Final Evaluation Comparison
# ==========================================

experiments = [
    "Baseline",
    "Learning Rate 0.0005",
    "Epsilon Decay 0.99",
    "1000 Episodes"
]

average_rewards = [
    -79.06,
    -108.03,
    -107.63,
    -181.03
]

plt.figure(figsize=(10, 5))

plt.bar(
    experiments,
    average_rewards
)

plt.xlabel("Experiment")
plt.ylabel("Average Evaluation Reward")
plt.title("DQN Experiment Comparison")
plt.xticks(rotation=15)
plt.grid(axis="y")

plt.savefig(
    "results/graphs/experiment_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()