import pandas as pd

file1 = "step_by_step_test_result_ep_800.csv"
df1 = pd.read_csv(file1)

df = pd.DataFrame(columns=["step", "myopic", "myopic_reward_rl_transition"])

num_of_episode = 30
myopic_cumulative_penalty = 0
rl_cumulative_penalty = 0

for step in range(300):
    total_myopic_penalty = 0
    total_rl_penalty = 0
    for episode in range(num_of_episode):
        index = episode * 300 + step

        penalty = float(df1.iloc[index]["myopic"])
        total_myopic_penalty += penalty

        penalty = float(df1.iloc[index]["myopic_reward_rl_transition"])
        total_rl_penalty += penalty

    average_myopic_penalty = total_myopic_penalty/num_of_episode
    average_rl_penalty = total_rl_penalty/num_of_episode

    myopic_cumulative_penalty += average_myopic_penalty
    rl_cumulative_penalty += average_rl_penalty

    row = {"step": step, "myopic": myopic_cumulative_penalty, "myopic_reward_rl_transition": rl_cumulative_penalty}
    df = df.append(row, ignore_index=True)

df.to_csv("cumulative_average_step_by_step_test_result.csv")






