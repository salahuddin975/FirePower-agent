import math

import pandas as pd

file1 = "episodic_test_result_ep_800.csv"
df1 = pd.read_csv(file1)

df = pd.DataFrame(columns=["step", "myopic", "myopic_reward_rl_transition"])

num_of_episodes = 30
myopic_penalties = []
rl_penalties = []

for i in range(num_of_episodes):
    penalty = float(df1.iloc[i]["myopic"])
    myopic_penalties.append(penalty)

    penalty = float(df1.iloc[i]["myopic_reward_rl_transition"])
    rl_penalties.append(penalty)

myopic_penalties.sort()
rl_penalties.sort()
print("Myopic_penalties: ", myopic_penalties)
print("RL_penalties: ", rl_penalties)

print("myopic_min: ", myopic_penalties[0], ", rl_min: ", rl_penalties[0])

pos = math.floor(num_of_episodes/4)
if num_of_episodes % 4 == 0:
    myopic_1st_quartile = (myopic_penalties[pos] + myopic_penalties[pos-1])/2
    rl_1st_quartile = (rl_penalties[pos] + rl_penalties[pos-1])/2
else:
    myopic_1st_quartile = myopic_penalties[pos]
    rl_1st_quartile = rl_penalties[pos]
print("myopic_1st_quartile: ", myopic_1st_quartile, ", rl_1st_quartile: ", rl_1st_quartile)

pos = math.floor(num_of_episodes/2)
if num_of_episodes % 2 == 0:
    myopic_median = (myopic_penalties[pos] + myopic_penalties[pos-1])/2
    rl_median = (rl_penalties[pos] + rl_penalties[pos-1])/2
else:
    myopic_median = myopic_penalties[pos]
    rl_median = rl_penalties[pos]
print("myopic_median: ", myopic_median, ", rl_median: ", rl_median)

pos = math.floor(num_of_episodes*3/4)
if num_of_episodes % 4 == 0:
    myopic_3rd_quartile = (myopic_penalties[pos] + myopic_penalties[pos-1])/2
    rl_3rd_quartile = (rl_penalties[pos] + rl_penalties[pos-1])/2
else:
    myopic_3rd_quartile = myopic_penalties[pos]
    rl_3rd_quartile = rl_penalties[pos]
print("myopic_3rd_quartile: ", myopic_3rd_quartile, ", rl_3rd_quartile: ", rl_3rd_quartile)

print("myopic_max: ", myopic_penalties[num_of_episodes-1], ", rl_max: ", rl_penalties[num_of_episodes-1])

