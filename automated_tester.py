import sys
import pandas as pd
import automated_tester_main
from automated_tester_main import ResultWriter

end_episode = 10
check_at_episode = 20
max_penalty = 400000

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python automated_tester.py <seed> <num_of_generator>")
        print("Ex. python automated_tester.py 50 11")

    seed_value = int(sys.argv[1])
    num_of_generator = int(sys.argv[2])

    model_version = 0
    path = f"./database_seed_{seed_value}/test_result/fire_power_reward_list_v0.csv"

    base_path = "database_seed_" + str(seed_value)
    result_writer = ResultWriter(base_path, model_version, seed_value, "_summary", True)

    df = pd.read_csv(path, header=0)
    max_episode = len(df)

    flag = True
    for i in range(max_episode):
        episode = max_episode - i - 1
        if episode < end_episode:
            print(f"Seed: {seed_value}, Finished at: {end_episode}")
            break

        if episode % check_at_episode != 0:
            continue

        if flag:
            result_writer.add_info(episode, 0, 0, 0)
            flag = False

        # if episode+4 < len(df):
        #     penalty = (df.loc[episode + 1, "total_penalty"] + df.loc[episode + 2, "total_penalty"] + df.loc[episode + 3, "total_penalty"] + df.loc[episode + 4, "total_penalty"]) / 4
        #     #penalty = (df.loc[episode + 1, "reward"] + df.loc[episode + 2, "reward"] + df.loc[episode + 3, "reward"] + df.loc[episode + 4, "reward"]) / 4
        #     if penalty > max_penalty:
        #         continue

        print("Start testing at checkpoint: ", episode)
        avg_score, violation_count, avg_load_loss = automated_tester_main.main(seed_value, num_of_generator,
                                                                               model_version, episode)
        if avg_score != 0:
            result_writer.add_info(episode, violation_count, avg_score, avg_load_loss)
        print(
            f"checkpoint: {episode}, violation_count_episodes: {violation_count}, average_score: {avg_score}, average_load_loss: {avg_load_loss}")

