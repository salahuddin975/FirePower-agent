import pandas as pd

step_by_step = True
episode = 800

if step_by_step:
    file1 = f"step_by_step_test_result_ep_{episode}.csv"
    df = pd.DataFrame(columns=["episode", "step", "myopic", "myopic_reward_rl_transition", "rl"])
else:
    file1 = f"episodic_test_result_ep_{episode}.csv"
    df = pd.DataFrame(columns=["episode", "myopic", "myopic_reward_rl_transition", "rl"])

total_myopic = 0
total_rl_transition = 0
total_rl = 0
df1 = pd.read_csv (file1)

for index in range(len(df1)):
    row = df1.iloc[index]

    if step_by_step and int(row["step"]) == 0:
        total_myopic = 0
        total_rl_transition = 0
        total_rl = 0

    myopic = float(row["myopic"])
    rl_transition = float(row["myopic_reward_rl_transition"])
    rl = float(row["rl"])

    total_myopic += myopic
    total_rl_transition += rl_transition
    total_rl += rl

    if step_by_step:
        val = {"episode":row["episode"], "step":row["step"], "myopic":total_myopic,
               "myopic_reward_rl_transition":total_rl_transition, "rl":total_rl}
    else:
        val = {"episode":row["episode"], "myopic":total_myopic,
               "myopic_reward_rl_transition":total_rl_transition, "rl":total_rl}

    df = df.append(val, ignore_index=True)

if step_by_step:
    df.to_csv(f"step_by_step_cumulative_penalty_ep_{episode}.csv")
else:
    df.to_csv(f"episodic_cumulative_penalty_ep_{episode}.csv")

