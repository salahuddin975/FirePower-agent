import pandas as pd


seed = 101
num_of_result = 25
path = "rnslab2/"

test_result = []
file_name = "fire_power_reward_list_v0.csv"

for i in range(num_of_result):
    file_path = path + "seed_" + str(seed + i) + "_" + file_name
    print(file_path)

    df = pd.read_csv(file_path, header=0)

    penalty_ep_no = 0
    min_penalty = -2700000.00
    step_count = 0

    for i in range(len(df)):
        if i >= 50 and df.loc[i, "episode_number"] % 20 == 0 and i+4 < len(df):
            penalty = (df.loc[i+1, "reward"] + df.loc[i+2, "reward"] + df.loc[i+3, "reward"] + df.loc[i+4, "reward"]) / 4
            if min_penalty < penalty:
                min_penalty = penalty
                penalty_ep_no = i
                step_count = min(df.loc[i+1, "max_reached_step"], df.loc[i+2, "max_reached_step"], df.loc[i+3, "max_reached_step"], df.loc[i+4, "max_reached_step"])
                print("min_penalty: ", min_penalty, "; at: ", penalty_ep_no)

    test_result.append((penalty_ep_no, step_count, min_penalty))
    print("episode_no: ", penalty_ep_no, "; max_reached_step: ", step_count, "; minimum_penalty: ", min_penalty)


penalty = []
for i, result in enumerate(test_result):
    print(f" - Test result{i+1}: episode_no: {result[0]}; max_reached_step: {result[1]}; minimum_penalty: {result[2]}")
    penalty.append(result[2])

print(f"\n - Summary: average minimum penalty: {sum(penalty)/5}; lowest: {max(penalty)}; highest: {min(penalty)}")