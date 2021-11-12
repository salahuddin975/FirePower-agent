import sys
import pandas as pd
import automated_tester_main
from automated_tester_main import ResultWriter



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python automated_tester.py <seed> <num_of_generator>")
        print("Ex. python automated_tester.py 50 11")

    seed_value = int(sys.argv[1])
    num_of_generator = int(sys.argv[2])

    model_version = 0
    path = f"./database_seed_{seed_value}/test_result/test_result/{seed_value}_summary_v0.csv"

    base_path = "database_seed_" + str(seed_value)
    result_writer = ResultWriter(base_path, model_version, seed_value, "_final_result")

    df = pd.read_csv(path, header=0)
    max_episode = len(df)

    flag = True
    for i in range(max_episode):
        episode = df.loc[i, "episode_number"]
        print(f"============== Test agent at: {episode} ==============")

        if flag:
            result_writer.add_info(episode, 0, 0)
            flag = False
            

        print("Start testing episode: ", episode)
        avg_score = automated_tester_main.main(seed_value, num_of_generator, model_version, episode)
        if avg_score != 0:
            result_writer.add_info(episode, 299, avg_score)
        print(f"episode: {episode}, avg_score: {avg_score}")

