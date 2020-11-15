import csv
import random
import datetime
import numpy as np
import tensorflow as tf
from pypower.idx_brch import *


class DataProcessor:
    def __init__(self, simulator_resources, generators, state_spaces, action_spaces):
        self.simulator_resources = simulator_resources
        self.generators = generators
        self._state_spaces = state_spaces
        self._action_spaces = action_spaces

    def _check_network_violations(self, bus_status, branch_status):
        from_buses = self.simulator_resources.ppc["branch"][:, F_BUS].astype('int')
        to_buses = self.simulator_resources.ppc["branch"][:, T_BUS].astype('int')

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

        max_output = self.generators.get_max_outputs()
        net_output = nn_output * max_output
        # print ("network output: ", net_output)

        generators_current_output = np.zeros(self.generators.get_size())
        for i in range(self.generators.get_size()):
            generators_current_output[i] = generators_output[self.generators.get_generators()[i]]
        # print("generators current output: ", generators_current_output)

        # print("nn ramp: ", nn_ramp)

        generators_max_output = self.generators.get_max_outputs()
        generators_min_output = self.generators.get_min_outputs()
        generators_max_ramp = self.generators.get_max_ramps()
        ramp = net_output - generators_current_output
        # print("generators initial ramp: ", ramp)

        for i in range(ramp.size):
            if ramp[i] > 0:
                ramp[i] = ramp[i] if ramp[i] < generators_max_ramp[i] else generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] < generators_max_output[i] else generators_max_output[i] - generators_current_output[i]
            else:
                ramp[i] = ramp[i] if abs(ramp[i]) < generators_max_ramp[i] else -generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] > generators_min_output[i] else generators_min_output[i] - generators_current_output[i]

        # print("generators set ramp: ", ramp)
        return ramp

    def _check_bus_generator_violation(self, bus_status, generators_ramp):
        selected_generators = self.generators.get_generators()

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

        nn_output = np_action["generator_injection"]
        ramp = self._clip_ramp_values(nn_output, generators_current_output)
        ramp = self._check_bus_generator_violation(bus_status, ramp)
        print("ramp: ", ramp)

        # generators_ramp = np.zeros(11, int)      # overwrite by dummy value (need to remove)

        action = {
            "bus_status": bus_status,
            "branch_status": branch_status,
            "generator_selector": self.generators.get_generators(),
            "generator_injection": ramp,
        }

        return action


    def explore_network(self, nn_action, explore_network = True, noise_range = 0.5):
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
        nn_output = np.array(tf.squeeze(nn_action[0]))
        for i in range(nn_output.size):
            if explore_network:
                nn_output[i] = nn_output[i] + random.uniform(-noise_range, noise_range)
        nn_output = np.clip(nn_output, 0, 1)
        # print("nn output: ", nn_output)

        action = {
            "generator_injection": nn_output,
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
        tf_line_flow = tf.expand_dims(tf.convert_to_tensor(state["line_flow"]), 0)

        return [tf_bus_status, tf_branch_status, tf_fire_distance, tf_generator_injection, tf_load_demand, tf_theta, tf_line_flow]


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


class Tensorboard:                 # $ tensorboard --logdir ./logs
    def __init__(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._agent_counter = 0
        self._critic_counter = 0

        agent_log_dir = 'logs/' + current_time + '/agent'
        citic_log_dir = 'logs/' + current_time + '/critic'

        self._agent_summary_writer = tf.summary.create_file_writer(agent_log_dir)
        self._critic_summary_writer = tf.summary.create_file_writer(citic_log_dir)


    def add_episodic_info(self, episodic_reward):
        with self._agent_summary_writer.as_default():
            tf.summary.scalar("episodic_reward", episodic_reward, step=self._agent_counter)
        self._agent_counter += 1


    def add_critic_network_info(self, critic_loss, reward_value, critic_value):
        with self._critic_summary_writer.as_default():
            tf.summary.scalar('critic_loss', critic_loss, step=self._critic_counter)
            tf.summary.scalar('reward_value', reward_value, step=self._critic_counter)
            tf.summary.scalar('critic_value', critic_value, step=self._critic_counter)
        self._critic_counter += 1


class SummaryWriter:
    def __init__(self, model_version):
        self._model_version = model_version
        self._file_name = "test_result/fire_power_reward_list"

        self._initialize()

    def _initialize(self):
        with open(f'{self._file_name}_v{self._model_version}.csv', 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(["model_version", "episode_number", "max_reached_step", "reward"])


    def add_info(self, episode, max_reached_step, episodic_reward):
        with open(f'{self._file_name}_v{self._model_version}.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(self._model_version), str(episode), str(max_reached_step), str(episodic_reward)])