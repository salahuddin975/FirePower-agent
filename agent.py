import os
import csv
import gym
import random
import copy
import time
import datetime
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.loadcase import loadcase
from pypower.ext2int import ext2int


gym.logger.set_level(25)
np.set_printoptions(linewidth=300)

seed_value = 50
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
agent_log_dir = 'logs/' + current_time + '/agent'
citic_log_dir = 'logs/' + current_time + '/critic'
agent_summary_writer = tf.summary.create_file_writer(agent_log_dir)
critic_summary_writer = tf.summary.create_file_writer(citic_log_dir)


class Agent:
    def __init__(self, state_spaces, action_spaces):
        self._gamma = 0.9      # discount factor
        self._tau = 0.05       # used to update target network
        actor_lr = 0.001
        critic_lr = 0.002
        self._actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self._critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self._state_spaces =  copy.deepcopy(state_spaces)
        self._action_spaces =  copy.deepcopy(action_spaces)

        self._actor = self._actor_model()
        self._target_actor = self._actor_model()
        self._target_actor.set_weights(self._actor.get_weights())

        self._critic = self._critic_model()
        self._target_critic = self._critic_model()
        self._target_critic.set_weights(self._critic.get_weights())

    def save_weight(self, version, episode_num):
        self._actor.save_weights(f"saved_model/agent_actor{version}_{episode_num}.h5")
        self._critic.save_weights(f"saved_model/agent_critic{version}_{episode_num}.h5")
        self._target_actor.save_weights(f"saved_model/agent_target_actor{version}_{episode_num}.h5")
        self._target_critic.save_weights(f"saved_model/agent_target_critic{version}_{episode_num}.h5")

    def load_weight(self, version, episode_num):
        self._actor.load_weights(f"saved_model/agent_actor{version}_{episode_num}.h5")
        self._target_actor.load_weights(f"saved_model/agent_target_actor{version}_{episode_num}.h5")
        self._critic.load_weights(f"saved_model/agent_critic{version}_{episode_num}.h5")
        self._target_critic.load_weights(f"saved_model/agent_target_critic{version}_{episode_num}.h5")
        print("weights are loaded successfully!")

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        action_batch1 = [action_batch[0], action_batch[1]]
        # update critic network
        with tf.GradientTape() as tape:
            target_actions = self._target_actor(next_state_batch)
            action_batch1.append(target_actions)
            y = reward_batch + self._gamma * self._target_critic([next_state_batch, action_batch1])
            critic_value = self._critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        action_batch1.pop()
        # update actor network
        with tf.GradientTape() as tape:
            actions = self._actor(state_batch)
            action_batch1.append(actions)
            critic_value1 = self._critic([state_batch, action_batch1])
            actor_loss = -1 * tf.math.reduce_mean(critic_value1)
        actor_grad = tape.gradient(actor_loss, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(actor_grad, self._actor.trainable_variables))

        self._update_target()
        return critic_loss, tf.math.reduce_mean(reward_batch), tf.math.reduce_mean(critic_value)

    def _update_target(self):
        # update target critic network
        new_weights = []
        target_critic_weights = self._target_critic.weights
        for i, critic_weight in enumerate(self._critic.weights):
            new_weights.append(self._tau * critic_weight + (1 - self._tau) * target_critic_weights[i])
        self._target_critic.set_weights(new_weights)

        # update target actor network
        new_weights = []
        target_actor_weights = self._target_actor.weights
        for i, actor_weight in enumerate(self._actor.weights):
            new_weights.append(self._tau * actor_weight + (1 - self._tau) * target_actor_weights[i])
        self._target_actor.set_weights(new_weights)

    def _actor_model(self):
        # bus -> MultiBinary(24)
        bus_input = layers.Input(shape=(self._state_spaces[0],))
        # bus_input1 = layers.Dense(32, activation="tanh") (bus_input)

        # num_branch -> MultiBinary(34)
        branch_input = layers.Input(shape=(self._state_spaces[1],))
        # branch_input1 = layers.Dense(32, activation="tanh") (branch_input)

        # fire_distance -> Box(58, )
        fire_distance_input = layers.Input(shape=(self._state_spaces[2],))
        # fire_distance_input1 = layers.Dense(64, activation="tanh") (fire_distance_input)

        # generator_injection -> Box(24, )
        gen_inj_input = layers.Input(shape=(self._state_spaces[3],))
        # gen_inj_input1 = layers.Dense(32, activation="tanh") (gen_inj_input)

        # load_demand -> Box(24, )
        load_demand_input = layers.Input(shape=(self._state_spaces[4], ))
        # load_demand_input1 = layers.Dense(32, activation="tanh") (load_demand_input)

        # theta -> Box(24, )
        theta_input = layers.Input(shape=(self._state_spaces[5], ))
        # theta_input1 = layers.Dense(32, activation="tanh") (theta_input)

        state = layers.Concatenate() ([bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input])
        # state = layers.Concatenate() ([bus_input1, branch_input1, fire_distance_input1, gen_inj_input1, load_demand_input1, theta_input1])
        hidden = layers.Dense(512, activation="tanh") (state)
        hidden = layers.Dense(128, activation="tanh") (hidden)

        # bus -> MultiBinary(24)
        # bus_output = layers.Dense(action_space[0], activation="sigmoid") (hidden)
        #
        # # num_branch -> MultiBinary(34)
        # branch_output = layers.Dense(action_space[1], activation="sigmoid") (hidden)

        # generator_injection (generator output) -> Box(5, )
        gen_inj_output = layers.Dense(self._action_spaces[3], activation="sigmoid") (hidden)

        model = tf.keras.Model([bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input],
                               [gen_inj_output])
        return model


    def _critic_model(self):
        # bus -> MultiBinary(24)
        st_bus = layers.Input(shape=(self._state_spaces[0],))
        # st_bus1 = layers.Dense(32, activation="relu") (st_bus)

        # num_branch -> MultiBinary(34)
        st_branch = layers.Input(shape=(self._state_spaces[1],))
        # st_branch1 = layers.Dense(32, activation="relu") (st_branch)

        # fire_distance -> Box(58, )
        st_fire_distance = layers.Input(shape=(self._state_spaces[2],))
        # st_fire_distance1 = layers.Dense(64, activation="relu") (st_fire_distance)

        # generator_injection (output) -> Box(24, )
        st_gen_output = layers.Input(shape=(self._state_spaces[3],))                     # Generator current total output
        # st_gen_output1 = layers.Dense(32, activation="relu") (st_gen_output)

        # load_demand -> Box(24, )
        st_load_demand = layers.Input(shape=(self._state_spaces[4], ))
        # st_load_demand1 = layers.Dense(32, activation="relu") (st_load_demand)

        # theta -> Box(24, )
        st_theta = layers.Input(shape=(self._state_spaces[5], ))
        # st_theta1 = layers.Dense(30, activation="relu") (st_theta)

        # bus -> MultiBinary(24)
        act_bus = layers.Input(shape=(action_spaces[0],))
        # act_bus1 = layers.Dense(30, activation="relu") (act_bus)
        #
        # # num_branch -> MultiBinary(34)
        act_branch = layers.Input(shape=(action_spaces[1],))
        # act_branch1 = layers.Dense(30, activation="relu") (act_branch)

        # generator_injection -> Box(5, )
        act_gen_injection = layers.Input(shape=(self._action_spaces[3],))
        # act_gen_injection1 = layers.Dense(32, activation="relu") (act_gen_injection)          # power ramping up/down

        # state = layers.Concatenate() ([st_bus, act_gen_injection])
        state = layers.Concatenate() ([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta,
                                       act_bus, act_branch, act_gen_injection])
        # state = layers.Concatenate() ([st_bus1, st_branch1, st_fire_distance1, st_gen_output1, st_load_demand1, st_theta1,
        #                                act_bus1, act_branch1, act_gen_injection1])

        hidden = layers.Dense(512, activation="relu") (state)
        hidden = layers.Dense(128, activation="relu") (hidden)
        reward = layers.Dense(1, activation="linear") (hidden)

        model = tf.keras.Model([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta,
                                act_bus, act_branch, act_gen_injection], reward)
        return model


class ReplayBuffer:
    def __init__(self, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_dir, load_replay_buffer_version=0, buffer_capacity=200000, batch_size=256):
        self._counter = 0
        self._capacity = buffer_capacity
        self._batch_size = batch_size

        if load_replay_buffer == False:
            self._initialize_buffer(state_spaces, action_spaces)
        else:
            self._load_buffer(load_replay_buffer_dir, load_replay_buffer_version)


    def _initialize_buffer(self, state_spaces, action_spaces):
        self.st_bus = np.zeros((self._capacity, state_spaces[0]))
        self.st_branch = np.zeros((self._capacity, state_spaces[1]))
        self.st_fire_distance = np.zeros((self._capacity, state_spaces[2]))
        self.st_gen_output = np.zeros((self._capacity, state_spaces[3]))
        self.st_load_demand = np.zeros((self._capacity, state_spaces[4]))
        self.st_theta = np.zeros((self._capacity, state_spaces[5]))

        self.act_bus = np.zeros((self._capacity, action_spaces[0]))
        self.act_branch = np.zeros((self._capacity, action_spaces[1]))
        self.act_gen_injection = np.zeros((self._capacity, action_spaces[3]))

        self.rewards = np.zeros((self._capacity, 1))

        self.next_st_bus = np.zeros((self._capacity, state_spaces[0]))
        self.next_st_branch = np.zeros((self._capacity, state_spaces[1]))
        self.next_st_fire_distance = np.zeros((self._capacity, state_spaces[2]))
        self.next_st_gen_output = np.zeros((self._capacity, state_spaces[3]))
        self.next_st_load_demand = np.zeros((self._capacity, state_spaces[4]))
        self.next_st_theta = np.zeros((self._capacity, state_spaces[5]))

        self.np_counter = np.zeros((1))


    def save_buffer(self, version):
        np.save(f'replay_buffer/st_bus_v{version}.npy', self.st_bus)
        np.save(f'replay_buffer/st_branch_v{version}.npy', self.st_branch)
        np.save(f'replay_buffer/st_fire_distance_v{version}.npy', self.st_fire_distance)
        np.save(f'replay_buffer/st_gen_output_v{version}.npy', self.st_gen_output)
        np.save(f'replay_buffer/st_load_demand_v{version}.npy', self.st_load_demand)
        np.save(f'replay_buffer/st_theta_v{version}.npy', self.st_theta)

        np.save(f'replay_buffer/act_bus_v{save_replay_buffer_version}.npy', self.act_bus)
        np.save(f'replay_buffer/act_branch_v{save_replay_buffer_version}.npy', self.act_branch)
        np.save(f'replay_buffer/act_gen_injection_v{version}.npy', self.act_gen_injection)

        np.save(f'replay_buffer/rewards_v{version}.npy', self.rewards)

        np.save(f'replay_buffer/next_st_bus_v{version}.npy', self.next_st_bus)
        np.save(f'replay_buffer/next_st_branch_v{version}.npy', self.next_st_branch)
        np.save(f'replay_buffer/next_st_fire_distance_v{version}.npy', self.next_st_fire_distance)
        np.save(f'replay_buffer/next_st_gen_output_v{version}.npy', self.next_st_gen_output)
        np.save(f'replay_buffer/next_st_load_demand_v{version}.npy', self.next_st_load_demand)
        np.save(f'replay_buffer/next_st_theta_v{version}.npy', self.next_st_theta)

        self.np_counter[0] = self._counter
        np.save(f'replay_buffer/counter_v{version}.npy', self.np_counter)


    def _load_buffer(self, load_buffer_dir, version):
        self.st_bus = np.load(f'{load_buffer_dir}/st_bus_v{version}.npy')
        self.st_branch = np.load(f'{load_buffer_dir}/st_branch_v{version}.npy')
        self.st_fire_distance = np.load(f'{load_buffer_dir}/st_fire_distance_v{version}.npy')
        self.st_gen_output = np.load(f'{load_buffer_dir}/st_gen_output_v{version}.npy')
        self.st_load_demand = np.load(f'{load_buffer_dir}/st_load_demand_v{version}.npy')
        self.st_theta = np.load(f'{load_buffer_dir}/st_theta_v{version}.npy')

        self.act_bus = np.load(f'replay_buffer/act_bus_v{load_replay_buffer_version}.npy')
        self.act_branch = np.load(f'replay_buffer/act_branch_v{load_replay_buffer_version}.npy')
        self.act_gen_injection = np.load(f'{load_buffer_dir}/act_gen_injection_v{version}.npy')

        self.rewards = np.load(f'{load_buffer_dir}/rewards_v{version}.npy')

        self.next_st_bus = np.load(f'{load_buffer_dir}/next_st_bus_v{version}.npy')
        self.next_st_branch = np.load(f'{load_buffer_dir}/next_st_branch_v{version}.npy')
        self.next_st_fire_distance = np.load(f'{load_buffer_dir}/next_st_fire_distance_v{version}.npy')
        self.next_st_gen_output = np.load(f'{load_buffer_dir}/next_st_gen_output_v{version}.npy')
        self.next_st_load_demand = np.load(f'{load_buffer_dir}/next_st_load_demand_v{version}.npy')
        self.next_st_theta = np.load(f'{load_buffer_dir}/next_st_theta_v{version}.npy')

        self.np_counter = np.load(f'{load_buffer_dir}/counter_v{version}.npy')
        self._counter = int(self.np_counter[0])
        print("Replay buffer loaded successfully!")
        print("Counter set at: ", self._counter)

    def get_num_records(self):
        record_size = min(self._capacity, self._counter)
        return record_size

    def add_record(self, record):
        index = self._counter % self._capacity

        self.st_bus[index] = np.copy(record[0]["bus_status"])
        self.st_branch[index] = np.copy(record[0]["branch_status"])
        self.st_fire_distance[index] = np.copy(record[0]["fire_distance"])
        self.st_fire_distance[index] = self.st_fire_distance[index] / 100
        self.st_gen_output[index] = np.copy(record[0]["generator_injection"])
        self.st_load_demand[index] = np.copy(record[0]["load_demand"])
        self.st_theta[index] = np.copy(record[0]["theta"])

        # use data from heuristic
        self.act_bus[index] = np.copy(record[4]["bus_status"])
        self.act_branch[index] = np.copy(record[4]["branch_status"])
        # use data from NN actor
        self.act_gen_injection[index] = np.copy(record[1]["generator_injection"])

        self.rewards[index] = record[2][0]

        self.next_st_bus[index] = np.copy(record[3]["bus_status"])
        self.next_st_branch[index] = np.copy(record[3]["branch_status"])
        self.next_st_fire_distance[index] = np.copy(record[3]["fire_distance"])
        self.next_st_fire_distance[index] = self.next_st_fire_distance[index] / 100
        self.next_st_gen_output[index] = np.copy(record[3]["generator_injection"])
        self.next_st_load_demand[index] = np.copy(record[3]["load_demand"])
        self.next_st_theta[index] = np.copy(record[3]["theta"])

        self._counter = self._counter + 1

    def get_batch(self):
        record_size = min(self._capacity, self._counter)
        batch_indices = np.random.choice(record_size, self._batch_size)

        st_tf_bus = tf.convert_to_tensor(self.st_bus[batch_indices])
        st_tf_branch = tf.convert_to_tensor(self.st_branch[batch_indices])
        st_tf_fire_distance = tf.convert_to_tensor(self.st_fire_distance[batch_indices])
        st_tf_gen_output = tf.convert_to_tensor(self.st_gen_output[batch_indices])
        st_tf_load_demand = tf.convert_to_tensor(self.st_load_demand[batch_indices])
        st_tf_theta = tf.convert_to_tensor(self.st_theta[batch_indices])

        act_tf_bus = tf.convert_to_tensor(self.act_bus[batch_indices])
        act_tf_branch = tf.convert_to_tensor(self.act_branch[batch_indices])
        act_tf_gen_injection = tf.convert_to_tensor(self.act_gen_injection[batch_indices])

        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)

        next_st_tf_bus = tf.convert_to_tensor(self.next_st_bus[batch_indices])
        next_st_tf_branch = tf.convert_to_tensor(self.next_st_branch[batch_indices])
        next_st_tf_fire_distance = tf.convert_to_tensor(self.next_st_fire_distance[batch_indices])
        next_st_tf_gen_output = tf.convert_to_tensor(self.next_st_gen_output[batch_indices])
        next_st_tf_load_demand = tf.convert_to_tensor(self.next_st_load_demand[batch_indices])
        next_st_tf_theta = tf.convert_to_tensor(self.next_st_theta[batch_indices])

        state_batch = [st_tf_bus, st_tf_branch, st_tf_fire_distance, st_tf_gen_output, st_tf_load_demand, st_tf_theta]
        action_batch = [act_tf_bus, act_tf_branch, act_tf_gen_injection]
        next_state_batch = [next_st_tf_bus, next_st_tf_branch, next_st_tf_fire_distance, next_st_tf_gen_output,
                                    next_st_tf_load_demand, next_st_tf_theta]

        return state_batch, action_batch, reward_batch, next_state_batch


class DataProcessor:
    def __init__(self, state_spaces, action_spaces):
        self._state_spaces = state_spaces
        self._action_spaces = action_spaces

    def _check_network_violations(self, bus_status, branch_status):
        from_buses = simulator_resources.ppc["branch"][:, F_BUS].astype('int')
        to_buses = simulator_resources.ppc["branch"][:, T_BUS].astype('int')

        for bus in range(bus_status.size):
            is_active = bus_status[bus]
            for branch in range(branch_status.size):
                if bus in [from_buses[branch], to_buses[branch]]:
                    if is_active == 0:
                        branch_status[branch] = 0

        return branch_status

    def _clip_ramp_values(self, nn_output, generators_output):
        # print("generators output: ", generators_output)
        # print("nn ratio output: ", nn_output)

        max_output = generators.get_max_outputs()
        net_output = nn_output * max_output
        # print ("network output: ", net_output)

        generators_current_output = np.zeros(generators.get_size())
        for i in range(generators.get_size()):
            generators_current_output[i] = generators_output[generators.get_generators()[i]]
        # print("generators current output: ", generators_current_output)

        # print("nn ramp: ", nn_ramp)

        generators_max_output = generators.get_max_outputs()
        generators_max_ramp = generators.get_max_ramps()
        ramp = net_output - generators_current_output
        # print("generators initial ramp: ", ramp)

        for i in range(ramp.size):
            if ramp[i] > 0:
                ramp[i] = ramp[i] if ramp[i] < generators_max_ramp[i] else generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] < generators_max_output[i] else generators_max_output[i] - generators_current_output[i]
            else:
                ramp[i] = ramp[i] if abs(ramp[i]) < generators_max_ramp[i] else -generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] > 0 else 0 - generators_current_output[i]

        # print("generators set ramp: ", ramp)
        return ramp

    def _check_bus_generator_violation(self, bus_status, generators_ramp):
        selected_generators = generators.get_generators()

        for bus in range(bus_status.size):
            flag = bus_status[bus]
            for j in range(selected_generators.size):
                gen_bus = selected_generators[j]
                if bus == gen_bus and flag == False:
                    generators_ramp[j] = False

        return generators_ramp


    def check_violations(self, np_action, fire_distance, generators_current_output, bus_threshold=0.1, branch_threshold=0.1):
        bus_status = np.ones(self._state_spaces[0])
        for i in range(self._state_spaces[0]):
            if fire_distance[i] < 2.0:
                bus_status[i] = 0

        branch_status = np.ones(self._state_spaces[1])
        for i in range(self._state_spaces[1]):
            if fire_distance[self._state_spaces[0] + i] < 2.0:
                branch_status[i] = 0

        branch_status = self._check_network_violations(bus_status, branch_status)
        # print("bus status: ", bus_status)
        # print("branch status: ", branch_status)

        nn_ramp = np_action["generator_injection"]
        ramp = self._clip_ramp_values(nn_ramp, generators_current_output)
        ramp = self._check_bus_generator_violation(bus_status, ramp)
        print("ramp: ", ramp)

        # generators_ramp = np.zeros(11, int)      # overwrite by dummy value (need to remove)

        action = {
            "bus_status": bus_status,
            "branch_status": branch_status,
            "generator_selector": generators.get_generators(),
            "generator_injection": ramp,
        }

        return action


    def explore_network(self, nn_action, explore_network = True, noise_range = 1.0):
        # bus status
        # bus_status = np.squeeze(np.array(tf_action[0]))
        # for i in range(bus_status.size):
        #     total = bus_status[i] + random.uniform(-1 * noise_range, noise_range)
        # print ("bus status: ", bus_status)

        # # branch status
        # branch_status = np.squeeze(np.array(tf_action[1]))
        # for i in range(branch_status.size):
        #     total = branch_status[i] + random.uniform(-1 * noise_range, noise_range)
        # print ("branch status: ", branch_status)

        # amount of power for ramping up/down
        nn_ramp = np.array(tf.squeeze(nn_action[0]))
        for i in range(nn_ramp.size):
            if explore_network:
                nn_ramp[i] = nn_ramp[i] + random.uniform(-noise_range, noise_range)
        nn_ramp = np.clip(nn_ramp, 0, 1)
        # print("ramp: ", nn_ramp)

        action = {
            "generator_injection": nn_ramp,
        }

        return action


    def get_tf_state(self, state):
        tf_bus_status = tf.expand_dims(tf.convert_to_tensor(state["bus_status"]), 0)
        tf_branch_status = tf.expand_dims(tf.convert_to_tensor(state["branch_status"]), 0)
        # tf_fire_state = tf.expand_dims(tf.convert_to_tensor(state["fire_state"]), 0)
        tf_fire_distance = tf.expand_dims(tf.convert_to_tensor(state["fire_distance"]), 0)
        tf_generator_injection = tf.expand_dims(tf.convert_to_tensor(state["generator_injection"]), 0)
        tf_load_demand = tf.expand_dims(tf.convert_to_tensor(state["load_demand"]), 0)
        tf_theta = tf.expand_dims(tf.convert_to_tensor(state["theta"]), 0)

        return [tf_bus_status, tf_branch_status, tf_fire_distance, tf_generator_injection, tf_load_demand, tf_theta]


    def get_tf_critic_input(self, state, action):
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


class Generators:
    def __init__(self, ppc, ramp_frequency_in_hour = 6):
        self.generators = np.copy(ppc["gen"][:, GEN_BUS].astype("int"))
        self.num_generators = self.generators.size
        self.generators_min_output = np.zeros(self.generators.size)
        self.generators_max_output = np.copy(ppc["gen"][:, PMAX] / ppc["baseMVA"])
        self.generators_max_ramp = np.copy((ppc["gen"][:, RAMP_10] / ppc["baseMVA"]) * (1 / ramp_frequency_in_hour))

    def get_generators(self):
        return self.generators

    def get_size(self):
        return self.num_generators

    def get_min_outputs(self):
        return self.generators_min_output

    def get_max_outputs(self):
        return self.generators_max_output

    def set_max_outputs(self, max_output):
        self.generators_max_output = np.copy(max_output[self.generators])

    def get_max_ramps(self):
        return  self.generators_max_ramp

    def print_info(self):
        print ("generators: ", self.generators)
        print ("generators max output: ", self.generators_max_output)
        print ("generators max ramp: ", self.generators_max_ramp)


class SimulatorResources():
    def __init__(self, power_file_path, geo_file_path):
        self._ppc = loadcase(power_file_path)
        self._merge_generators()
        self._merge_branches()
        self.ppc = ext2int(self._ppc)

    def _merge_generators(self):
        ppc_gen_trim = []
        temp = self._ppc["gen"][0, :]
        ptr = 0
        ptr1 = 1
        while(ptr1 < self._ppc["gen"].shape[0]):
            if self._ppc["gen"][ptr, GEN_BUS] == self._ppc["gen"][ptr1, GEN_BUS]:
                temp[PG:QMIN+1] += self._ppc["gen"][ptr1, PG:QMIN+1]
                temp[PMAX:APF+1] += self._ppc["gen"][ptr1, PMAX:APF+1]
            else:
                ppc_gen_trim.append(temp)
                temp = self._ppc["gen"][ptr1, :]
                ptr = ptr1
            ptr1 += 1
        ppc_gen_trim.append(temp)
        self._ppc["gen"] = np.asarray(ppc_gen_trim)

    def _merge_branches(self):
        ppc_branch_trim = []
        temp = self._ppc["branch"][0, :]
        ptr = 0
        ptr1 = 1
        while(ptr1 < self._ppc["branch"].shape[0]):
            if np.all(self._ppc["branch"][ptr, F_BUS:T_BUS+1] == self._ppc["branch"][ptr1, GEN_BUS:T_BUS+1]):
                temp[BR_R: RATE_C+1] += self._ppc["branch"][ptr1, BR_R: RATE_C+1]
            else:
                ppc_branch_trim.append(temp)
                temp = self._ppc["branch"][ptr1, :]
                ptr = ptr1
            ptr1 += 1
        ppc_branch_trim.append(temp)
        self._ppc["branch"] = np.asarray(ppc_branch_trim)

    def get_ppc(self):
        return self.ppc

    def print_ppc(self):
        print (self.ppc)


def get_state_spaces(observation_space):
    # print("observation space: ", observation_space)

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


def get_action_spaces(action_space):
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
    # print(args)

    simulator_resources = SimulatorResources(power_file_path = args.path_power, geo_file_path=args.path_geo)
    generators = Generators(ppc = simulator_resources.ppc, ramp_frequency_in_hour = 6)

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power, num_tunable_gen=11)

    state_spaces = get_state_spaces(env.observation_space)
    action_spaces = get_action_spaces(env.action_space)

    agent = Agent(state_spaces, action_spaces)
    data_processor = DataProcessor(state_spaces, action_spaces)

    # save trained model to reuse
    save_model = False
    load_model = False
    save_model_version = 0
    load_model_version = 0
    load_episode_num = 0

    if load_model:
        agent.load_weight(version=load_model_version, episode_num=load_episode_num)

    save_replay_buffer = False
    save_replay_buffer_version = 0

    load_replay_buffer = False
    load_replay_buffer_version = 0
    load_replay_buffer_dir = "replay_buffer"

    buffer = ReplayBuffer(state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_dir, load_replay_buffer_version,
                          buffer_capacity=200000, batch_size=1024)

    with open(f'fire_power_reward_list_v{save_model_version}.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(["model_version", "episode_number", "max_reached_step", "reward"])

    total_episode = 100001
    max_steps_per_episode = 300
    num_train_per_episode = 300
    episodic_rewards = []
    explore_network_flag = True

    for episode in range(total_episode):
        state = env.reset()

        episodic_reward = 0
        max_reached_step = 0
        generators.set_max_outputs(state["generator_injection"])

        for step in range(max_steps_per_episode):
            tf_state = data_processor.get_tf_state(state)
            nn_action = agent._actor(tf_state)
            print("NN generator output: ", nn_action[0])

            net_action = data_processor.explore_network(nn_action, explore_network=explore_network_flag, noise_range=.5)
            env_action = data_processor.check_violations(net_action, state["fire_distance"], state["generator_injection"])

            next_state, reward, done, _ =  env.step(env_action)
            print(f"Episode: {episode}, at step: {step}, reward: {reward[0]}")

            penalty = reward[0] # - 1 * np.sum(np.abs(net_action["generator_injection"])) * 100  # -1000 * net_action["generator_injection"][0] *  net_action["generator_injection"][0]
            reward1 = [penalty, reward[1]]
            buffer.add_record((state, net_action, reward1, next_state, env_action))

            episodic_reward += reward[0]
            state = next_state

            if done or (step == max_steps_per_episode-1):
                print(f"Episode: V{save_model_version}_{episode}, done at step: {step}, total reward: {episodic_reward}")
                max_reached_step = step
                break

        # if (buffer.get_num_records() > 300):
        # print ("Train at: ", episode)
        # for i in range(num_train_per_episode):
            state_batch, action_batch, reward_batch, next_state_batch = buffer.get_batch()
            critic_loss, reward_value, critic_value = agent.train(state_batch, action_batch, reward_batch, next_state_batch)   # magnitude of gradient
            i = step
            with critic_summary_writer.as_default():
                tf.summary.scalar('critic_loss', critic_loss, step=i + episode*num_train_per_episode)
                tf.summary.scalar('reward_value', reward_value, step=i + episode*num_train_per_episode)
                tf.summary.scalar('critic_value', critic_value, step=i + episode*num_train_per_episode)

        with agent_summary_writer.as_default():
            tf.summary.scalar("episodic_reward", episodic_reward, step=episode)

        # explore / Testing
        if episode and (episode % 10 == 0):
            print ("Start testing network at: ", episode)
            explore_network_flag = False
        if episode and (episode % 10 == 2):
            print ("Start exploring network at: ", episode)
            explore_network_flag = True

        # save update in csv file
        with open(f'fire_power_reward_list_v{save_model_version}.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(save_model_version), str(episode), str(max_reached_step), str(episodic_reward)])

        # save logs
        if (episode % 5 == 0) and save_model:
            log_file = open("saved_model/reward_log.txt", "a")
            log_file.write(f"Episode: V{save_model_version}_{episode}, Reward: {episodic_reward}\n")
            log_file.close()

        # save model weights
        if (episode % 50 == 0) and save_model:
            agent.save_weight(version=save_model_version, episode_num=episode)

        # save replay buffer
        if (episode % 10 == 0) and save_replay_buffer:
            print(f"Saving replay buffer at: {episode}")
            buffer.save_buffer(save_replay_buffer_version)
