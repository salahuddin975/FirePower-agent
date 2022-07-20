import pandas as pd
import matplotlib.pyplot as plt

load_episode = 1920
file1 = f"step_by_step_test_result_ep_{load_episode}.csv"
file2 = f"perfect_control_load_loss.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# episode = 10
for episode in range(10, 100):
    plt.cla()
    myopic_penalty = []
    rl_penalty = []
    perfect_control_penalty = []
    for step in range(300):
        index = episode * 300 + step

        penalty = float(df1.iloc[index]["myopic"])
        myopic_penalty.append(penalty)

        penalty = float(df1.iloc[index]["myopic_reward_rl_transition"])
        rl_penalty.append(penalty)

        penalty = -1 * float(df2.iloc[step][f"episode_{episode}"])/100
        perfect_control_penalty.append(penalty)

    plt.plot(myopic_penalty, color="blue", label="myopic")
    # plt.plot(rl_penalty, label="rl")
    plt.plot(perfect_control_penalty, color="red", label="perfect_control")
    plt.legend(loc="lower left")
    # plt.show()
    plt.savefig(f"plots/episode_{episode}.png")

