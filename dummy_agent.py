import gym
import argparse
from gym import wrappers, logger
import os
from pprint import PrettyPrinter
import numpy as np
# import matplotlib.pyplot as plt
# import pickle

pp = PrettyPrinter(compact=True, depth=3)
gym.logger.set_level(20)


class CleverAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done, itr):
        action = {"generator_injection": np.zeros(10, int),
                  "branch_status": np.ones(34, int),
                  "bus_status": np.ones(24, int),
                  "generator_selector": np.array([24]*10)}

        if itr >= 40:
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [-0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181])

        if itr >= 45:
            # opening line (15,18)
            action["branch_status"][27] = 0
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [0]*10)

        if itr >= 46:
            action["generator_selector"] = np.array(
                [24]*10)

        # if itr >= 114:
        #     action["generator_selector"] = np.array(
        #         [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
        #     action["generator_injection"] = np.array(
        #         [0, 0, 0, 0, 0, -1.11, -1.983, 1.96327, -0.82327, 1.516])

        if itr >= 115:
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [0, 0, 0, 0, 0, -1.0325, -1.983, 1.91327, -0.82327, 0.771])

        if itr >= 116:
            # opening line (13,15)
            action["branch_status"][22] = 0
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [0]*10)
        
        if itr > 120:
            action["generator_selector"] = np.array(
                [24]*10)
            

        if itr >= 164:
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [-0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181, -0.181])

        if itr >= 165:
            # opening line (18,19)
            action["branch_status"][31] = 0
            action["bus_status"][18] = 0
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [0]*10)
        
        if itr >= 170:
            action["generator_selector"] = np.array(
                [24]*10)
        
        if itr >= 171:
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 21, 22])
            action["generator_injection"] = np.array(
                [-0.215, -0.215, -0.22722, -0.22722, -0.22722, 0, 0, -0.22722, -0.22722, -0.22722])

        if itr >= 172:
            # opening line (10,13)
            action["branch_status"][18] = 0
            action["bus_status"][13] = 0
            action["generator_selector"] = np.array(
                [0, 1, 6, 12, 14, 15, 17, 20, 24, 24])
            action["generator_injection"] = np.array(
                [0]*10)

        if itr >= 174:
            action["generator_selector"] = np.array(
                [24]*10)

        if itr >= 249:
            # opening line (14,20)
            action["branch_status"][24] = 0

        return action


class DummyAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return {"generator_injection": np.zeros(10, int),
                "branch_status": np.ones(34, int),
                "bus_status": np.ones(24, int),
                "generator_selector": np.array([24]*10)}


class SneakyAgent(object):

    def __init__(self, action_space):
        self.acttion_space = action_space

    def act(self, observation, reward, done, itr):
        action = {"generator_injection": np.zeros(10, int),
                  "branch_status": np.ones(34, int),
                  "bus_status": np.ones(24, int),
                  "generator_selector": np.array([24]*10)}
        if itr >= 45:
            # opening line (15,18)
            action["branch_status"][27] = 0
        if itr >= 117:
            # opening line (13,15)
            action["branch_status"][22] = 0
        if itr >= 165:
            # opening line (18,19)
            action["branch_status"][31] = 0
            action["bus_status"][18] = 0
        if itr >= 171:
            # opening line (10,13)
            action["branch_status"][18] = 0
            action["bus_status"][13] = 0
        if itr >= 249:
            # opening line (14,20)
            action["branch_status"][24] = 0
        return action


def main(args):
    env = gym.envs.make("gym_firepower:firepower-v0",
                        geo_file=args.path_geo, network_file=args.path_power)

    # agent = DummyAgent(env.action_space)
    # agent = SneakyAgent(env.action_space)
    agent = CleverAgent(env.action_space)
    episode_count = args.num_episodes
    iterations = args.num_iterations
    reward = 0
    done = False

    if args.enable_recording:
        env = wrappers.Monitor(env, directory=args.path_output, force=True)
    reward_list = []
    for i in range(episode_count):
        ob = env.reset()
        itr_count = 0
        for t in range(iterations):
            itr_count += 1
            # action = agent.act(ob, reward, done)
            action = agent.act(ob, reward, done, itr_count)
            ob, reward, done, _ = env.step(action)
            reward_list.append(reward)
            logger.info("Episode #{}, Iteration #{}, Reward {}, Load loss {}".format(
                i, itr_count, reward[0], reward[1]))
            if done:
                logger.info("Game over!!!")
                break
        else:
            if args.enable_recording:
                env.stats_recorder.save_complete()
                env.stats_recorder.done = True
    env.close()

    # pickle.dump(reward_list, open(os.path.join(
    #     args.path_output, "reward_clever.p"), 'wb'))
    # plt.plot(range(1, len(reward_list)+1), reward_list, '--')
    # plt.ylabel("Reward")
    # plt.xlabel("Iterations")
    # plt.title("Sneaky Agent #1 (Live Equipment Removal Action Penalty)")
    # plt.savefig(os.path.join(args.path_output, "rewards.png"))


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Dummy Agent for gym_firepower")
   parser.add_argument('-g', '--path-geo',
                       help="Full path to geo file", required=True)
   parser.add_argument('-p', '--path-power',
                       help="Full path to power systems file", required=True)
   parser.add_argument('-f', '--scale-factor',
                       help="Scaling factor", type=int, default=6)
   parser.add_argument('-n', '--nonconvergence-penalty',
                       help="Non-convergence penalty", type=float)
   parser.add_argument('-a', '--protectionaction-penalty',
                       help="Protection action penalty", type=float)
   parser.add_argument('-l', '--activelineremoval-penalty',
                       help="Active line removal action penalty", type=float)
   parser.add_argument(
       '-s', '--seed', help="Seed for random number generator", type=int)
   parser.add_argument('-o', '--path-output',
                       help="Output directory for dumping environment data")
   parser.add_argument('-e', '--num-episodes',
                       help="Number of episodes to run", type=int, default=1)
   parser.add_argument('-i', '--num-iterations',
                       help="Max number of iterations in an episode", type=int, default=300)
   parser.add_argument('-r', '--enable-recording',
                       help="Disable recording", action='store_true')
   args = parser.parse_args()
   print(args)
   main(args)

