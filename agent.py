import gym
import argparse
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from gym import wrappers, logger
import os

gym.logger.set_level(40)



def get_actor(num_bus, num_branch, num_fire_status, num_gen_inj, num_load_demand, num_theta, num_gen_selector, num_gen_output):
    # bus -> MultiBinary(24)
    bus_input = layers.Input(shape=(num_bus,))
    bus_input1 = layers.Dense(30, activation="relu") (bus_input)

    # num_branch -> MultiBinary(34)
    branch_input = layers.Input(shape=(num_branch,))
    branch_input1 = layers.Dense(30, activation="relu") (branch_input)

    # fire_status -> Box(350, 350)
    fire_input = layers.Input(shape=(num_fire_status, num_fire_status))
    fire_input1 = layers.Flatten()(fire_input)
    fire_input1 = layers.Dense(500, activation="relu") (fire_input1)

    # generator_injection -> Box(24, )
    gen_inj_input = layers.Input(shape=(num_gen_inj,))
    gen_inj_input1 = layers.Dense(30, activation="relu") (gen_inj_input)

    # load_demand -> Box(24, )
    load_demand_input = layers.Input(shape=(num_load_demand, ))
    load_demand_input1 = layers.Dense(30, activation="relu") (load_demand_input)

    # theta -> Box(24, )
    theta_input = layers.Input(shape=(num_theta, ))
    theta_input1 = layers.Dense(30, activation="relu") (theta_input)

    state = layers.Concatenate() ([bus_input1, branch_input1, fire_input1, gen_inj_input1, load_demand_input1, theta_input1])
    hidden = layers.Dense(512, activation="relu") (state)
    hidden = layers.Dense(512, activation="relu") (hidden)
    hidden = layers.Dense(512, activation="relu") (hidden)

    # bus -> MultiBinary(24)
    bus_output = layers.Dense(num_bus) (hidden)

    # num_branch -> MultiBinary(34)
    branch_output = layers.Dense(num_branch) (hidden)

    # generator_selector -> MultiDiscrete([12 12 12 12 12])
    gen_selector_output = layers.Dense(num_gen_selector) (hidden)

    # generator_injection (generator output) -> Box(5, )
    gen_inj_output = layers.Dense(num_gen_output) (hidden)

    model = tf.keras.Model([bus_input, branch_input, fire_input, gen_inj_input, load_demand_input, theta_input], [bus_output, branch_output, gen_selector_output, gen_inj_output])
    return model


def get_tf_state(obj):
    tf_bus_status = tf.expand_dims(tf.convert_to_tensor(obj["bus_status"]), 0)
    tf_branch_status = tf.expand_dims(tf.convert_to_tensor(obj["branch_status"]), 0)
    tf_fire_state = tf.expand_dims(tf.convert_to_tensor(obj["fire_state"]), 0)
    tf_generator_injection = tf.expand_dims(tf.convert_to_tensor(obj["generator_injection"]), 0)
    tf_load_demand = tf.expand_dims(tf.convert_to_tensor(obj["load_demand"]), 0)
    tf_theta = tf.expand_dims(tf.convert_to_tensor(obj["theta"]), 0)

    return [tf_bus_status, tf_branch_status, tf_fire_state, tf_generator_injection, tf_load_demand, tf_theta]


def get_action(tf_action):
    bus_status = np.array(tf_action[0])
    bus_status[: 1] = bus_status[:] > 0
    bus_status = np.squeeze(bus_status.astype(int))
    print ("bus status: ", bus_status)

    branch_status = np.array(tf_action[1])
    branch_status[: 1] = branch_status[:] > 0
    branch_status = np.squeeze(branch_status.astype(int))
    print ("branch status: ", branch_status)

    gen_selector = np.array(tf.squeeze(tf_action[2]))
    gen_selector = np.abs(gen_selector * 22)
    gen_selector = gen_selector.astype(int)
    print("gen selector: ", gen_selector)

    gen_injection = np.array(tf.squeeze(tf_action[3]))
    print("gen injection: ", gen_injection)

    action = {"generator_injection": gen_injection,
        "branch_status": branch_status,
        "bus_status": bus_status,
        "generator_selector": gen_selector}

    return action


def main(args):
    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power)

    print("action_space: ", env.action_space)
    print("observation space: ", env.observation_space)

    action_space = env.action_space
    num_branch = action_space["branch_status"].shape[0]
    num_bus = action_space["bus_status"].shape[0]
    num_generator_output = action_space["generator_injection"].shape[0]
    num_generator_selector = action_space["generator_selector"].shape[0]
    print (f"num branch: {num_branch}, num bus: {num_bus}, num generator injection: {num_generator_output}, num_generator_selector: {num_generator_selector}")

    observation_space = env.observation_space
    num_fire_status = observation_space["fire_status"].shape[0]
    num_gen_injection = observation_space["generator_injection"].shape[0]
    num_load_demand = observation_space["load_demand"].shape[0]
    num_theta = observation_space["theta"].shape[0]
    print(f"fire status: {num_fire_status}, num_gen_injection: {num_gen_injection}, num_load_demand: {num_load_demand}, num_theta: {num_theta}")

    actor = get_actor(num_bus, num_branch, num_fire_status, num_gen_injection, num_load_demand, num_theta,num_generator_selector, num_generator_output)

    state = env.reset()
    state = get_tf_state(state)

    tf_action = actor(state)
    action = get_action(tf_action)
    print ("action: ", action)

    # action = env.action_space.sample()
    ob, reward, done, _ = env.step(action)
    print("reward: ", reward)
    # print("sample action: ", action)
    # print("obs: ", ob)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy Agent for gym_firepower")
    parser.add_argument('-g', '--path-geo', help="Full path to geo file", required=True)
    parser.add_argument('-p', '--path-power', help="Full path to power systems file", required=False)
    parser.add_argument('-f', '--scale-factor', help="Scaling factor", type=int, default=6)
    parser.add_argument('-n', '--nonconvergence-penalty', help="Non-convergence penalty", type=float)
    parser.add_argument('-a', '--protectionaction-penalty', help="Protection action penalty", type=float)
    parser.add_argument('-s', '--seed', help="Seed for random number generator", type=int)
    parser.add_argument('-o', '--path-output', help="Output directory for dumping environment data")
    args = parser.parse_args()
    print(args)
    main(args)

