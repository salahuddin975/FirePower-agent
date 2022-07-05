import pandas as pd

file1 = "step_by_step_test_result_ep_800.csv"
df1 = pd.read_csv (file1)

df = pd.DataFrame(columns=["episode", "step", "myopic", "myopic_reward_rl_transition", "rl"])

total_myopic = 0
total_rl_transition = 0
total_rl = 0

for index in range(len(df1)):
    row = df1.iloc[index]

    if int(row["step"]) == 0:
        total_myopic = 0
        total_rl_transition = 0
        total_rl = 0

    myopic = abs(float(row["myopic"]))
    rl_transition = abs(float(row["myopic_reward_rl_transition"]))
    rl = abs(float(row["rl"]))

    total_myopic += myopic
    total_rl_transition += rl_transition
    total_rl += rl

    val = {"episode":row["episode"], "step":row["step"], "myopic":total_myopic,
           "myopic_reward_rl_transition":total_rl_transition, "rl":total_rl}
    df = df.append(val, ignore_index=True)
df.to_csv("step_by_step_cumulative_positive_penalty.csv")






