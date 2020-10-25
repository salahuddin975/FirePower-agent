import os
import gym
import random
import math
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int

seed_value = 50
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

gym.logger.set_level(25)

class ReplayBuffer:
    def __init__(self, state_spaces, action_spaces, buffer_capacity=20000, batch_size=64):
        self.counter = 0
        self.gamma = 0.99      # discount factor
        self.tau = 0.005       # used to update target network
        actor_lr = 0.001
        critic_lr = 0.002
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.capacity = buffer_capacity
        self.batch_size = batch_size

        self.st_bus = np.zeros((self.capacity, state_spaces[0]))
        self.st_branch = np.zeros((self.capacity, state_spaces[1]))
        # self.st_fire = np.zeros((self.capacity, state_spaces[2], state_spaces[2]))
        self.st_fire_distance = np.zeros((self.capacity, state_spaces[2]))
        self.st_gen_output = np.zeros((self.capacity, state_spaces[3]))
        self.st_load_demand = np.zeros((self.capacity, state_spaces[4]))
        self.st_theta = np.zeros((self.capacity, state_spaces[5]))

        self.act_bus = np.zeros((self.capacity, action_spaces[0]))
        self.act_branch = np.zeros((self.capacity, action_spaces[1]))
        self.act_gen_selector = np.zeros((self.capacity, action_spaces[2]))
        self.act_gen_injection = np.zeros((self.capacity, action_spaces[3]))

        self.rewards = np.zeros((self.capacity, 1))

        self.next_st_bus = np.zeros((self.capacity, state_spaces[0]))
        self.next_st_branch = np.zeros((self.capacity, state_spaces[1]))
        # self.next_st_fire = np.zeros((self.capacity, state_spaces[2], state_spaces[2]))
        self.next_st_fire_distance = np.zeros((self.capacity, state_spaces[2]))
        self.next_st_gen_output = np.zeros((self.capacity, state_spaces[3]))
        self.next_st_load_demand = np.zeros((self.capacity, state_spaces[4]))
        self.next_st_theta = np.zeros((self.capacity, state_spaces[5]))


    def current_record_size(self):
        record_size = min(self.capacity, self.counter)
        return record_size


    def add_record(self, record):
        index = self.counter % self.capacity

        self.st_bus[index] = np.copy(record[0]["bus_status"])
        self.st_branch[index] = np.copy(record[0]["branch_status"])
        # self.st_fire[index] = np.copy(record[0]["fire_state"])
        self.st_fire_distance[index] = np.copy(record[0]["fire_distance"])
        self.st_gen_output[index] = np.copy(record[0]["generator_injection"])
        self.st_load_demand[index] = np.copy(record[0]["load_demand"])
        self.st_theta[index] = np.copy(record[0]["theta"])

        self.act_bus[index] = np.copy(record[1]["bus_status"])
        self.act_branch[index] = np.copy(record[1]["branch_status"])
        self.act_gen_selector[index] = np.copy(record[1]["generator_selector"])
        self.act_gen_injection[index] = np.copy(record[1]["generator_injection"])

        self.rewards[index] = record[2][0]

        self.next_st_bus[index] = np.copy(record[3]["bus_status"])
        self.next_st_branch[index] = np.copy(record[3]["branch_status"])
        # self.next_st_fire[index] = np.copy(record[3]["fire_state"])
        self.next_st_fire_distance[index] = np.copy(record[3]["fire_distance"])
        self.next_st_gen_output[index] = np.copy(record[3]["generator_injection"])
        self.next_st_load_demand[index] = np.copy(record[3]["load_demand"])
        self.next_st_theta[index] = np.copy(record[3]["theta"])

        self.counter = self.counter + 1


    def learn(self):
        record_size = min(self.capacity, self.counter)
        batch_indices = np.random.choice(record_size, self.batch_size)

        st_tf_bus = tf.convert_to_tensor(self.st_bus[batch_indices])
        st_tf_branch = tf.convert_to_tensor(self.st_branch[batch_indices])
        # st_tf_fire = tf.convert_to_tensor(self.st_fire[batch_indices])
        st_tf_fire_distance = tf.convert_to_tensor(self.st_fire_distance[batch_indices])
        st_tf_gen_output = tf.convert_to_tensor(self.st_gen_output[batch_indices])
        st_tf_load_demand = tf.convert_to_tensor(self.st_load_demand[batch_indices])
        st_tf_theta = tf.convert_to_tensor(self.st_theta[batch_indices])

        act_tf_bus = tf.convert_to_tensor(self.act_bus[batch_indices])
        act_tf_branch = tf.convert_to_tensor(self.act_branch[batch_indices])
        act_tf_gen_selector = tf.convert_to_tensor(self.act_gen_selector[batch_indices])
        act_tf_gen_injection = tf.convert_to_tensor(self.act_gen_injection[batch_indices])

        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)

        next_st_tf_bus = tf.convert_to_tensor(self.next_st_bus[batch_indices])
        next_st_tf_branch = tf.convert_to_tensor(self.next_st_branch[batch_indices])
        # next_st_tf_fire = tf.convert_to_tensor(self.next_st_fire[batch_indices])
        next_st_tf_fire_distance = tf.convert_to_tensor(self.next_st_fire_distance[batch_indices])
        next_st_tf_gen_output = tf.convert_to_tensor(self.next_st_gen_output[batch_indices])
        next_st_tf_load_demand = tf.convert_to_tensor(self.next_st_load_demand[batch_indices])
        next_st_tf_theta = tf.convert_to_tensor(self.next_st_theta[batch_indices])

        # update critic network
        with tf.GradientTape() as tape:
            target_actions = target_actor([next_st_tf_bus, next_st_tf_branch, next_st_tf_fire_distance, next_st_tf_gen_output,
                                    next_st_tf_load_demand, next_st_tf_theta])
            # need to check if target action needs to be converted
            y = reward_batch + self.gamma * target_critic([next_st_tf_bus, next_st_tf_branch, next_st_tf_fire_distance,
                                    next_st_tf_gen_output, next_st_tf_load_demand, next_st_tf_theta, target_actions])
            critic_value = critic([st_tf_bus, st_tf_branch, st_tf_fire_distance, st_tf_gen_output, st_tf_load_demand,
                                   st_tf_theta, act_tf_bus, act_tf_branch, act_tf_gen_selector, act_tf_gen_injection])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))

        # update actor network
        with tf.GradientTape() as tape:
            actions = actor([st_tf_bus, st_tf_branch, st_tf_fire_distance, st_tf_gen_output, st_tf_load_demand, st_tf_theta])
            # need to check if target action needs to be converted
            critic_value = critic([st_tf_bus, st_tf_branch, st_tf_fire_distance, st_tf_gen_output, st_tf_load_demand, st_tf_theta, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))


    def update_target(self):
        # update target critic network
        new_weights = []
        target_critic_weights = target_critic.weights
        for i, critic_weight in enumerate(critic.weights):
            new_weights.append(self.tau * critic_weight + (1 - self.tau) * target_critic_weights[i])
        target_critic.set_weights(new_weights)

        # update target actor network
        new_weights = []
        target_actor_weights = target_actor.weights
        for i, actor_weight in enumerate(actor.weights):
            new_weights.append(self.tau * actor_weight + (1 - self.tau) * target_actor_weights[i])
        target_actor.set_weights(new_weights)


def get_actor(state_space, action_space):
    # bus -> MultiBinary(24)
    bus_input = layers.Input(shape=(state_space[0],))
    bus_input1 = layers.Dense(30, activation="relu") (bus_input)

    # num_branch -> MultiBinary(34)
    branch_input = layers.Input(shape=(state_space[1],))
    branch_input1 = layers.Dense(30, activation="relu") (branch_input)

    # # fire_status -> Box(350, 350)
    # fire_input = layers.Input(shape=(state_space[2], state_space[2]))
    # fire_input1 = layers.Flatten()(fire_input)
    # fire_input1 = layers.Dense(500, activation="relu") (fire_input1)

    # fire_distance -> Box(58, )
    fire_distance_input = layers.Input(shape=(state_space[2],))
    fire_distance_input1 = layers.Dense(75, activation="relu") (fire_distance_input)

    # generator_injection -> Box(24, )
    gen_inj_input = layers.Input(shape=(state_space[3],))
    gen_inj_input1 = layers.Dense(30, activation="relu") (gen_inj_input)

    # load_demand -> Box(24, )
    load_demand_input = layers.Input(shape=(state_space[4], ))
    load_demand_input1 = layers.Dense(30, activation="relu") (load_demand_input)

    # theta -> Box(24, )
    theta_input = layers.Input(shape=(state_space[5], ))
    theta_input1 = layers.Dense(30, activation="relu") (theta_input)

    state = layers.Concatenate() ([bus_input1, branch_input1, fire_distance_input1, gen_inj_input1, load_demand_input1, theta_input1])
    hidden = layers.Dense(512, activation="relu") (state)
    hidden = layers.Dense(512, activation="relu") (hidden)
    hidden = layers.Dense(512, activation="relu") (hidden)

    # bus -> MultiBinary(24)
    bus_output = layers.Dense(action_space[0], activation="sigmoid") (hidden)

    # num_branch -> MultiBinary(34)
    branch_output = layers.Dense(action_space[1], activation="sigmoid") (hidden)

    # generator_selector -> MultiDiscrete([12 12 12 12 12])
    gen_selector_output = layers.Dense(action_space[2], activation="sigmoid") (hidden)

    # generator_injection (generator output) -> Box(5, )
    gen_inj_output = layers.Dense(action_space[3], activation="tanh") (hidden)

    model = tf.keras.Model([bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input],
                           [bus_output, branch_output, gen_selector_output, gen_inj_output])
    return model


def get_critic(state_spaces, action_spaces):
    # bus -> MultiBinary(24)
    st_bus = layers.Input(shape=(state_spaces[0],))
    st_bus1 = layers.Dense(30, activation="relu") (st_bus)

    # num_branch -> MultiBinary(34)
    st_branch = layers.Input(shape=(state_spaces[1],))
    st_branch1 = layers.Dense(30, activation="relu") (st_branch)

    # # fire_status -> Box(350, 350)
    # st_fire = layers.Input(shape=(state_spaces[2], state_spaces[2]))
    # st_fire1 = layers.Flatten()(st_fire)
    # st_fire1 = layers.Dense(500, activation="relu") (st_fire1)

    # fire_distance -> Box(58, )
    st_fire_distance = layers.Input(shape=(state_spaces[2],))
    st_fire_distance1 = layers.Dense(60, activation="relu") (st_fire_distance)

    # generator_injection (output) -> Box(24, )
    st_gen_output = layers.Input(shape=(state_spaces[3],))                     # Generator current total output
    st_gen_output1 = layers.Dense(30, activation="relu") (st_gen_output)

    # load_demand -> Box(24, )
    st_load_demand = layers.Input(shape=(state_spaces[4], ))
    st_load_demand1 = layers.Dense(30, activation="relu") (st_load_demand)

    # theta -> Box(24, )
    st_theta = layers.Input(shape=(state_spaces[5], ))
    st_theta1 = layers.Dense(30, activation="relu") (st_theta)

    # bus -> MultiBinary(24)
    act_bus = layers.Input(shape=(action_spaces[0],))
    act_bus1 = layers.Dense(30, activation="relu") (act_bus)

    # num_branch -> MultiBinary(34)
    act_branch = layers.Input(shape=(action_spaces[1],))
    act_branch1 = layers.Dense(30, activation="relu") (act_branch)

    # generator_selector -> MultiDiscrete([12 12 12 12 12])
    act_gen_selector = layers.Input(shape=(action_spaces[2],))
    act_gen_selector1 = layers.Dense(30, activation="relu") (act_gen_selector)

    # generator_injection -> Box(5, )
    act_gen_injection = layers.Input(shape=(action_spaces[3],))
    act_gen_injection1 = layers.Dense(30, activation="relu") (act_gen_injection)          # power ramping up/down

    state = layers.Concatenate() ([st_bus1, st_branch1, st_fire_distance1, st_gen_output1, st_load_demand1, st_theta1])
    action = layers.Concatenate() ([act_bus1, act_branch1, act_gen_selector1, act_gen_injection1])
    hidden = layers.Concatenate() ([state, action])

    hidden = layers.Dense(512, activation="relu") (hidden)
    hidden = layers.Dense(512, activation="relu") (hidden)
    reward = layers.Dense(1, activation="linear") (hidden)

    model = tf.keras.Model([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta,
                            act_bus, act_branch, act_gen_selector, act_gen_injection], reward)
    return model


def  get_tf_critic_input(state, action):
    st_bus_status = tf.expand_dims(tf.convert_to_tensor(state["bus_status"]), 0)
    st_branch_status = tf.expand_dims(tf.convert_to_tensor(state["branch_status"]), 0)
    st_fire_state = tf.expand_dims(tf.convert_to_tensor(state["fire_state"]), 0)
    st_generator_output = tf.expand_dims(tf.convert_to_tensor(state["generator_injection"]), 0)
    st_load_demand = tf.expand_dims(tf.convert_to_tensor(state["load_demand"]), 0)
    st_theta = tf.expand_dims(tf.convert_to_tensor(state["theta"]), 0)

    act_bus_status = tf.expand_dims(tf.convert_to_tensor(action["bus_status"]), 0)
    act_branch_status = tf.expand_dims(tf.convert_to_tensor(action["branch_status"]), 0)
    act_generator_selector = tf.expand_dims(tf.convert_to_tensor(action["generator_selector"]), 0)
    act_generator_injection = tf.expand_dims(tf.convert_to_tensor(action["generator_injection"]), 0)

    return [st_bus_status, st_branch_status, st_fire_state, st_generator_output, st_load_demand, st_theta,
            act_bus_status, act_branch_status, act_generator_selector, act_generator_injection]


def get_tf_state(state):
    tf_bus_status = tf.expand_dims(tf.convert_to_tensor(state["bus_status"]), 0)
    tf_branch_status = tf.expand_dims(tf.convert_to_tensor(state["branch_status"]), 0)
    # tf_fire_state = tf.expand_dims(tf.convert_to_tensor(state["fire_state"]), 0)
    tf_fire_distance = tf.expand_dims(tf.convert_to_tensor(state["fire_distance"]), 0)
    tf_generator_injection = tf.expand_dims(tf.convert_to_tensor(state["generator_injection"]), 0)
    tf_load_demand = tf.expand_dims(tf.convert_to_tensor(state["load_demand"]), 0)
    tf_theta = tf.expand_dims(tf.convert_to_tensor(state["theta"]), 0)

    return [tf_bus_status, tf_branch_status, tf_fire_distance, tf_generator_injection, tf_load_demand, tf_theta]


def check_network_violations(bus_status, branch_status):
    from_buses = ppc["branch"][:, F_BUS].astype('int')
    to_buses = ppc["branch"][:, T_BUS].astype('int')

    for bus in range(bus_status.size):
        is_active = bus_status[bus]
        for branch in range(branch_status.size):
            if bus in [from_buses[branch], to_buses[branch]]:
                if is_active == 0:
                    branch_status[branch] = 0

    return branch_status


def get_selected_generators_with_ramp(generators_current_output, indices_prob, ramp_ratio):
    # print("generators current output: ", generators_current_output)

    selected_indices = indices_prob * (generators.size)
    selected_indices = selected_indices.astype(int)
    # print("selected indices: ", selected_indices, "; generators_size: ", generators.size)
    selected_generators = generators[selected_indices]
    # print("selected generators: ", selected_generators)

    gene_current_output = np.zeros(generators.size)
    for i in range(generators.size):
        gene_current_output[i] = generators_current_output[generators[i]]
    # print("all generators current output: ", gene_current_output)

    selected_generators_current_output = gene_current_output[selected_indices]
    selected_generators_max_output = generators_max_output[selected_indices]
    selected_generators_max_ramp = generators_max_ramp[selected_indices]
    selected_generators_initial_ramp = selected_generators_max_ramp * ramp_ratio
    # print("selected generators max ramp: ", selected_generators_max_ramp)
    # print("selected generators ramp: ", selected_generators_initial_ramp)

    decimal = 10000
    selected_generators_ramp = np.zeros(selected_generators.size)
    for i in range(selected_generators.size):
        index = selected_generators[i]
        # print("index: ", index, "; ramp: ", selected_generators_ramp[i], "; cur: ", generators_current_output[index],
        #       "; total: ",selected_generators_ramp[i] + generators_current_output[index], "; max_output: ", selected_generators_max_output[i])
        if selected_generators_initial_ramp[i] == 0:
            selected_generators_ramp[i] = 0
        elif selected_generators_initial_ramp[i] > 0:
            if generators_current_output[index] == selected_generators_max_output[i]:
                selected_generators_ramp[i] = 0
            elif selected_generators_max_output[i] >= (selected_generators_initial_ramp[i] + generators_current_output[index]):
                selected_generators_ramp[i] = math.floor(selected_generators_initial_ramp[i] * decimal) / decimal
                generators_current_output[index] = generators_current_output[index] + selected_generators_ramp[i]
            else:
                selected_generators_ramp[i] = math.floor((selected_generators_max_output[i] - generators_current_output[index]) * decimal) / decimal
                generators_current_output[index] = selected_generators_max_output[i]
                # print("ramp: ", selected_generators_set_ramp[i], "; curr: ", generators_current_output[index])
        else:
            if generators_current_output[index] == 0:
                selected_generators_ramp[i] = 0
            elif 0 < (selected_generators_initial_ramp[i] + generators_current_output[index]):
                selected_generators_ramp[i] = math.ceil(selected_generators_initial_ramp[i] * decimal)/decimal
                generators_current_output[index] = generators_current_output[index] + selected_generators_ramp[i]
            else:
                selected_generators_ramp[i] = math.ceil((0 - generators_current_output[index]) * decimal)/decimal
                generators_current_output[index] = 0.0

        # print("updated output: ", generators_current_output)

    # print("selected generators current output: ", selected_generators_current_output)
    # print("selected generators max output: ", selected_generators_max_output)
    # print("generators set ramp: ", selected_generators_ramp)

    return selected_generators, selected_generators_ramp


def check_bus_generator_violation(bus_status, selected_generators, generators_ramp):
    for bus in range(bus_status.size):
        flag = bus_status[bus]
        for j in range(selected_generators.size):
            gen_bus = selected_generators[j]
            if bus == gen_bus and flag == False:
                generators_ramp[j] = False

    return generators_ramp


def get_processed_action(tf_action, fire_distance, generators_current_output, bus_threshold=0.1, branch_threshold=0.1, explore_network = False):
    # print(f"explore network: {explore_network}")
    # print("fire distance: ", fire_distance)

    bus_status = np.ones(num_bus)
    for i in range(num_bus):
        if fire_distance[i] < 2.0:
            bus_status[i] = 0

    branch_status = np.ones(num_branch)
    for i in range(num_branch):
        if fire_distance[num_bus+i] < 2.0:
            branch_status[i] = 0

    branch_status = check_network_violations(bus_status, branch_status)

    # print("bus status: ", bus_status)
    # print("branch status: ", branch_status)

    # bus status
    # bus_status = np.squeeze(np.array(tf_action[0]))
    # if explore_network:
    #     for i in range(bus_status.size):
    #         total = bus_status[i] + noise_generator()
    #         bus_status[i] = total if 0 <= total and total >= 1 else bus_status[i]
    # bus_status= np.expand_dims(bus_status, axis=0)
    # bus_status[:1] = bus_status[:] > bus_threshold
    # bus_status = np.squeeze(bus_status.astype(int))
    # # print ("bus status: ", bus_status)
    #
    # # branch status
    # branch_status = np.squeeze(np.array(tf_action[1]))
    # if explore_network:
    #     for i in range(branch_status.size):
    #         total = branch_status[i] + noise_generator()
    #         branch_status[i] = total if total >= 0 and total <=1 else branch_status[i]
    # branch_status = np.expand_dims(branch_status, axis=0)
    # branch_status[: 1] = branch_status[:] > branch_threshold
    # branch_status = np.squeeze(branch_status.astype(int))
    # branch_status = check_network_violations(bus_status, branch_status)
    # print ("branch status: ", branch_status)

    # select generators for power ramping up/down
    indices_prob = np.array(tf.squeeze(tf_action[2]))
    if explore_network:
        for i, x in enumerate(indices_prob):
            indices_prob[i] = indices_prob[i] + noise_generator()
    for i, x in enumerate(indices_prob):
        indices_prob[i] = indices_prob[i] if indices_prob[i] < 1  else 0.999
        indices_prob[i] = indices_prob[i] if indices_prob[i] > 0 else 0
    # print ("indices prob: ", indices_prob)

    # amount of power for ramping up/down
    ramp_ratio = np.array(tf.squeeze(tf_action[3]))
    if explore_network:
        for i, x in enumerate(ramp_ratio):
            ramp_ratio[i] = ramp_ratio[i] + noise_generator()
    # print("ramp ratio: ", ramp_ratio)

    selected_generators, generators_ramp = get_selected_generators_with_ramp(generators_current_output, indices_prob, ramp_ratio)
    # print("selected generators: ", selected_generators)
    generators_ramp = check_bus_generator_violation(bus_status, selected_generators, generators_ramp)
    # print("generators ramp: ", generators_ramp)

    # bus_status = np.ones(24, int)          # overwrite by dummy bus status (need to remove)
    # branch_status = np.ones(34, int)       # overwrite by dummy branch status (need to remove)
    # selected_generators = np.array([24]*10)       # overwrite by dummy value (need to remove)
    # generators_ramp = np.zeros(10, int)      # overwrite by dummy value (need to remove)

    action = {
        "bus_status": bus_status,
        "branch_status": branch_status,
        "generator_selector": selected_generators,
        "generator_injection": generators_ramp,
    }

    return action


class NoiseGenerator:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.mean = mean
        self.std_deviation = std_deviation
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def reset(self):
        if self.x_initial is None:
            self.x_prev = np.zeros_like(self.mean)
        else:
            self.x_prev = self.x_initial

    def __call__(self):
        x = self.x_prev \
            + self.theta * (self.mean - self.x_prev) * self.dt \
            + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)

        self.x_prev = x
        return x


def merge_generators():
    ppc_gen_trim = []
    temp = ppc["gen"][0, :]
    ptr = 0
    ptr1 = 1
    while(ptr1 < ppc["gen"].shape[0]):
        if ppc["gen"][ptr, GEN_BUS] == ppc["gen"][ptr1, GEN_BUS]:
            temp[PG:QMIN+1] += ppc["gen"][ptr1, PG:QMIN+1]
            temp[PMAX:APF+1] += ppc["gen"][ptr1, PMAX:APF+1]
        else:
            ppc_gen_trim.append(temp)
            temp = ppc["gen"][ptr1, :]
            ptr = ptr1
        ptr1 += 1
    ppc_gen_trim.append(temp)
    ppc["gen"] = np.asarray(ppc_gen_trim)


def merge_branches():
    ppc_branch_trim = []
    temp = ppc["branch"][0, :]
    ptr = 0
    ptr1 = 1
    while(ptr1 < ppc["branch"].shape[0]):
        if np.all(ppc["branch"][ptr, F_BUS:T_BUS+1] == ppc["branch"][ptr1, GEN_BUS:T_BUS+1]):
            temp[BR_R: RATE_C+1] += ppc["branch"][ptr1, BR_R: RATE_C+1]
        else:
            ppc_branch_trim.append(temp)
            temp = ppc["branch"][ptr1, :]
            ptr = ptr1
        ptr1 += 1
    ppc_branch_trim.append(temp)
    ppc["branch"] = np.asarray(ppc_branch_trim)


def get_generators_info(ramp_frequency_in_hour = 6):
    # generators information from config file
    generators = ppc["gen"][:, GEN_BUS].astype("int")
    generators_min_output = np.zeros(generators.size)
    generators_max_output = ppc["gen"][:, PMAX]/ppc["baseMVA"]
    generators_max_ramp = (ppc["gen"][:, RAMP_10]/ppc["baseMVA"]) * (1/ramp_frequency_in_hour)

    # print ("generators: ", generators)
    # print ("generators max output: ", generators_max_output)
    # print ("generators max ramp: ", generators_max_ramp)
    #
    return generators, generators_min_output, generators_max_output, generators_max_ramp


def get_state_spaces(env):
    observation_space = env.observation_space
    print("observation space: ", observation_space)

    num_st_bus = observation_space["bus_status"].shape[0]
    num_st_branch = observation_space["branch_status"].shape[0]
    # num_fire_status = observation_space["fire_status"].shape[0]
    num_fire_distance = observation_space["fire_distance"].shape[0]
    num_gen_output = observation_space["generator_injection"].shape[0]
    num_load_demand = observation_space["load_demand"].shape[0]
    num_theta = observation_space["theta"].shape[0]
    state_spaces = [num_st_bus, num_st_branch, num_fire_distance, num_gen_output, num_load_demand, num_theta]
    print(f"State Spaces: num bus: {num_st_bus}, num branch: {num_st_branch}, fire distance: {num_fire_distance}, "
          f"num_gen_injection: {num_gen_output}, num_load_demand: {num_load_demand}, num_theta: {num_theta}")

    return state_spaces


def get_action_spaces(env):
    action_space = env.action_space
    num_bus = action_space["bus_status"].shape[0]
    num_branch = action_space["branch_status"].shape[0]
    num_generator_selector = action_space["generator_selector"].shape[0]
    num_generator_injection = action_space["generator_injection"].shape[0]
    action_spaces = [num_bus, num_branch, num_generator_selector, num_generator_injection]
    print (f"Action Spaces: num bus: {num_bus}, num branch: {num_branch}, num_generator_selector: {num_generator_selector}, "
            f"num generator injection: {num_generator_injection}")

    return action_spaces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy Agent for gym_firepower")
    parser.add_argument('-g', '--path-geo', help="Full path to geo file", required=True)
    parser.add_argument('-p', '--path-power', help="Full path to power systems file", required=False)
    parser.add_argument('-f', '--scale-factor', help="Scali    actor_lr = 0.001ng factor", type=int, default=6)
    parser.add_argument('-n', '--nonconvergence-penalty', help="Non-convergence penalty", type=float)
    parser.add_argument('-a', '--protectionaction-penalty', help="Protection action penalty", type=float)
    parser.add_argument('-s', '--seed', help="Seed for random number generator", type=int)
    parser.add_argument('-o', '--path-output', help="Output directory for dumping environment data")
    args = parser.parse_args()
    print(args)

    ppc = loadcase(args.path_power)
    merge_generators()
    merge_branches()
    ppc = ext2int(ppc)
    generators, generators_min_output, generators_max_output, generators_max_ramp = get_generators_info(ramp_frequency_in_hour=6)

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power, num_tunable_gen=10)

    state_spaces = get_state_spaces(env)
    action_spaces = get_action_spaces(env)

    num_bus = state_spaces[0]
    num_branch = state_spaces[1]

    std_dev = 0.2
    noise_generator = NoiseGenerator(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor = get_actor(state_spaces, action_spaces)
    target_actor = get_actor(state_spaces, action_spaces)

    critic = get_critic(state_spaces, action_spaces)
    target_critic = get_critic(state_spaces, action_spaces)

    # save trained model to reuse
    save_model = False
    reload_model = True
    save_model_version = 0
    reload_model_version = 1
    reload_episode_num = 600
    if reload_model == False:
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())
    else:
        actor.load_weights(f"saved_model/agent_actor{reload_model_version}_{reload_episode_num}.h5")
        target_actor.load_weights(f"saved_model/agent_target_actor{reload_model_version}_{reload_episode_num}.h5")
        critic.load_weights(f"saved_model/agent_critic{reload_model_version}_{reload_episode_num}.h5")
        target_critic.load_weights(f"saved_model/agent_target_critic{reload_model_version}_{reload_episode_num}.h5")
        print("weights are loaded successfully!")

    total_episode = 5000
    max_steps_per_episode = 300
    train_agent_per_episode = 100
    buffer = ReplayBuffer(state_spaces, action_spaces, 15000, 64)

    epsilon = 0.5               # initial exploration rate
    max_epsilon = 0.5
    min_epsilon = 0.01
    decay_rate = 0.002          # exponential decay rate for exploration probability

    episodic_rewards = []
    dummy_agent_flag = False
    for episode in range(total_episode):
        state = env.reset()
        episodic_reward = 0

        for step in range(max_steps_per_episode):
            tf_state = get_tf_state(state)
            tf_action = actor(tf_state)

            tradeoff = random.uniform(0, 1)
            if tradeoff < epsilon:              # explore (add noise)
                action = get_processed_action(tf_action, state["fire_distance"], state["generator_injection"], bus_threshold=0.1, branch_threshold=0.1, explore_network=True)
            else:                               # exploit (use network)
                action = get_processed_action(tf_action, state["fire_distance"], state["generator_injection"], bus_threshold=0.1, branch_threshold=0.1, explore_network=False)

            next_state, reward, done, _ = env.step(action)
            print(f"Episode: {episode}, dummy_agent: {dummy_agent_flag}, at step: {step}, reward: {reward[0]}")

            episodic_reward += reward[0]
            buffer.add_record((state, action, reward, next_state))

            if done:
                print(f"Episode: V{save_model_version}_{episode}, dummy_agent: {dummy_agent_flag}, done at step: {step}, total reward: {episodic_reward}")
                break

            state = next_state

        print("Train agent, current number of records: ", buffer.current_record_size())
        for i in range(train_agent_per_episode):
            buffer.learn()
            buffer.update_target()

        # reduce epsilon as we need less and less exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        episodic_rewards.append(episodic_reward)
        avg_reward = np.mean(episodic_rewards[-25:])        # calculate moving average

        # save model weights
        if (episode % 100 == 0) and save_model:
            actor.save_weights(f"saved_model/agent_actor{save_model_version}_{episode}.h5")
            critic.save_weights(f"saved_model/agent_critic{save_model_version}_{episode}.h5")
            target_actor.save_weights(f"saved_model/agent_target_actor{save_model_version}_{episode}.h5")
            target_critic.save_weights(f"saved_model/agent_target_critic{save_model_version}_{episode}.h5")

        # save logs
        if (episode % 5 == 0) and save_model:
            log_file = open("saved_model/reward_log.txt", "a")
            log_file.write(f"Episode: V{save_model_version}_{episode}, dummy_agent: {dummy_agent_flag}, Reward: {episodic_reward}, Avg reward: {avg_reward}\n")
            log_file.close()

