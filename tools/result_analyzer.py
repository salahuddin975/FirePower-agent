import sys
import csv

# seed = int(sys.argv[1])

seed=50
num_of_result=5

path = "rnslab1/"
file_name = "fire_power_reward_list_v0.csv"

test_result = []
for i in range(num_of_result):
    file_path = path + "seed_" + str(seed+i) + "_" + file_name
    print(file_path)

    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        penalty_ep_no = 0
        min_penalty = -2700000.00
        step_ep_no = 0
        max_step = 0

        prev_row = ""
        flag = False
        for row in csv_reader:
            if flag:
                if int(row[1])%20 == 2 and int(row[1]) > 50:
                    if (float(row[3]) + float(prev_row[3]))/2 > min_penalty:
                        penalty_ep_no = row[1]
                        min_penalty = (float(row[3]) + float(prev_row[3]))/2
                        print("min_penalty: ", min_penalty, "; at: ", penalty_ep_no)

                if (int(row[2]) >= max_step) and (int(row[1])%20 == 1 or int(row[1])%20 == 2) and int(row[1]) > 50:
                        step_ep_no = row[1]
                        max_step = int(row[2])

            flag = True
            prev_row = row

        test_result.append((penalty_ep_no, max_step, min_penalty))
        print("max_reached_step_episode: ", step_ep_no)
        print("episode_no: ", penalty_ep_no, "; max_reached_step: ", max_step, "; minimum_penalty: ", min_penalty)

penalty = []
for i, result in enumerate(test_result):
    print(f" - Test result{i+1}: episode_no: {result[0]}; max_reached_step: {result[1]}; minimum_penalty: {result[2]}")
    penalty.append(result[2])

print(f"\n - Summary: average minimum penalty: {sum(penalty)/5}; lowest: {max(penalty)}; highest: {min(penalty)}")

