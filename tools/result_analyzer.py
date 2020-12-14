import pandas as pd


seed = 101
num_of_result = 17
path = "rnslab1/"

test_result = []
file_name = "fire_power_reward_list_v0.csv"

for itr1 in range(num_of_result):
    file_path = path + "seed_" + str(seed + itr1) + "_" + file_name
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

    test_result.append((seed+itr1, penalty_ep_no, step_count, min_penalty))
    print("seed: ", seed+itr1, "episode_no: ", penalty_ep_no, "; max_reached_step: ", step_count, "; minimum_penalty: ", min_penalty)


penalty = []
for result in test_result:
    print(f" - Seed: {result[0]}; episode_no: {result[1]}; min_reached_step: {result[2]}; minimum_penalty: {result[3]}")
    penalty.append(result[3])

print(f"\n - Summary: average minimum penalty: {sum(penalty)/5}; lowest: {max(penalty)}; highest: {min(penalty)}")