import pandas as pd

load_episode = 1920
file1 = f"step_by_step_test_result_ep_{load_episode}.csv"
df1 = pd.read_csv(file1)

num_of_episode = 101
for episode in range(num_of_episode):
    df = pd.DataFrame(columns=["episode","step","myopic","myopic_reward_rl_transition","rl","myopic_reward_rl_transition-myopic","rl-myopic"])

    for step in range(300):
        index = episode * 300 + step
        row = df1.iloc[index]
        df = df.append(row, ignore_index=True)

    df.to_csv(f"episode_info/step_info_for_episode_{episode}.csv")


