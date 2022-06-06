import os
import csv
import random
import datetime
import numpy as np
import tensorflow as tf
from pypower.idx_brch import *

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x  # make next noise depended on current one
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DataProcessor:
    def __init__(self, simulator_resources, generators, state_spaces, action_spaces):
        self.simulator_resources = simulator_resources
        self.generators = generators
        self._state_spaces = state_spaces
        self._action_spaces = action_spaces
        self._considerable_fire_distance = 10

        std_dev = 0.2
        self._ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        self._branches = [(0, 1),(0, 2),(0, 4),(1, 3),(1, 5),(2, 8),(2, 23),(3, 8),(4, 9),(5, 9),(6, 7),(7, 8),(7, 9),(8, 10),
            (8, 11),(9, 10),(9, 11),(10, 12),(10, 13),(11, 12),(11, 22),(12, 22),(13, 15),(14, 15),(14, 20),
            (14, 23),(15, 16),(15, 18),(16, 17),(16, 21),(17, 20),(18, 19),(19, 22),(20, 21)]
        self._generators = [0,  1,  6, 12, 13, 14, 15, 17, 20, 21, 22]

    def _check_network_violations_branch(self, bus_status, branch_status):
        from_buses = self.simulator_resources.ppc["branch"][:, F_BUS].astype('int')
        to_buses = self.simulator_resources.ppc["branch"][:, T_BUS].astype('int')

        for bus in range(bus_status.size):
            is_active = bus_status[bus]
            for branch in range(branch_status.size):
                if bus in [from_buses[branch], to_buses[branch]]:
                    if is_active == 0:
                        branch_status[branch] = 0

        return branch_status

    def _check_network_violations_bus(self, bus_status, branch_status):
        generator_sets = [set() for _ in range(24)]
        for branch in self._branches:
            generator_sets[branch[0]].add(branch)
            generator_sets[branch[1]].add(branch)

        for i in range(34):
            if branch_status[i] == 0:
                x, y = self._branches[i]
                generator_sets[x].remove((x,y))
                generator_sets[y].remove((x,y))

                if len(generator_sets[x]) == 0 and x in self._generators:
                    index = self._generators.index(x)
                    bus_status[index] = 0

                if len(generator_sets[y]) == 0 and y in self._generators:
                    index = self._generators.index(y)
                    bus_status[index] = 0

        # print("bus_status:", bus_status)
        # print("branch_status:", branch_status)
        return bus_status

    def add_heuristic_ramp(self, ramp, load_loss, num_generators, generators_current_output, generators_max_output, generators_max_ramp):
        for i in range(num_generators):
            ramp[i] = 0
            if load_loss > 0:
                if generators_current_output[i] < generators_max_output[i]:
                    ramp[i] = generators_max_output[i] - generators_current_output[i]
                    if ramp[i] > generators_max_ramp[i]:
                        ramp[i] = generators_max_ramp[i]
                    if ramp[i] > load_loss:
                        ramp[i] = load_loss
                    load_loss = load_loss - ramp[i]

    def _clip_ramp_values(self, nn_output, generators_output):
        # print("generators output: ", generators_output)
        # print("nn ratio output: ", nn_output)

        num_generators = self.generators.get_num_generators()
        generators_current_output = np.zeros(num_generators)
        for i in range(num_generators):
            generators_current_output[i] = generators_output[self.generators.get_generators()[i]]
        # print("generators current output: ", generators_current_output)

        # print("nn ramp: ", nn_ramp)

        generators_max_output = self.generators.get_max_outputs()
        generators_min_output = self.generators.get_min_outputs()
        generators_max_ramp = self.generators.get_max_ramps()

        # net_output =  nn_output * generators_max_output
        net_output = generators_min_output + nn_output * (generators_max_output - generators_min_output)
        # print ("network output: ", net_output)

        ramp = net_output - generators_current_output
        # print("generators initial ramp: ", ramp)

        for i in range(ramp.size):
            if ramp[i] > 0:
                ramp[i] = ramp[i] if ramp[i] < generators_max_ramp[i] else generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] < generators_max_output[i] else generators_max_output[i] - generators_current_output[i]
            else:
                ramp[i] = ramp[i] if abs(ramp[i]) < generators_max_ramp[i] else -generators_max_ramp[i]
                ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] > generators_min_output[i] else generators_min_output[i] - generators_current_output[i]

            if abs(ramp[i]) < 0.00001:
                ramp[i] = 0.0

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

    def check_violations(self, np_action, state, ramp_scale):
        bus_status = np.deepcopy(state["bus_status"])
        branch_status = np.deepcopy(state["branch_status"])
        generators_current_output = state["generator_injection"],

        # fire_distance = state["fire_distance"]
        # bus_status = np.ones(self._state_spaces[0])
        # for i in range(self._state_spaces[0]):
        #     if fire_distance[i] == 1:
        #         bus_status[i] = 0

        # branch_status = np.ones(self._state_spaces[1])
        # for i in range(self._state_spaces[1]):
        #     if fire_distance[self._state_spaces[0] + i] == 1:
        #         branch_status[i] = 0

        branch_status = self._check_network_violations_branch(bus_status, branch_status)
        bus_status = self._check_network_violations_bus(bus_status, branch_status)
        # print("bus status: ", bus_status)
        # print("branch status: ", branch_status)

        nn_output = np_action["generator_injection"]
        ramp = self._clip_ramp_values(nn_output, generators_current_output)
        ramp = self._check_bus_generator_violation(bus_status, ramp)
        # print("ramp: ", ramp)

        # generators_ramp = np.zeros(11, int)      # overwrite by dummy value (need to remove)

        action = {
            "bus_status": bus_status,
            "branch_status": branch_status,
            "generator_selector": self.generators.get_generators(),
            "generator_injection": ramp * ramp_scale,
        }

        # action = {
        #     "bus_status": np.ones(24),
        #     "branch_status": np.ones(34),
        #     "generator_selector": np.array([24] * 11),
        #     "generator_injection": np.zeros(11, int),
        # }

        return action

    def explore_network(self, nn_action, explore_network, noise_range=0.5):
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
                nn_output[i] = nn_output[i] + self._ou_noise() # random.uniform(-noise_range, noise_range)
        nn_output = np.clip(nn_output, 0, 1)
        # print("nn output: ", nn_output)

        action = {
            "generator_injection": nn_output,
        }

        return action

    def preprocess(self, state, power_generation_scale, explore_network_flag):
        state["generator_injection"] = np.array([output / power_generation_scale for output in state["generator_injection"]])
        state["load_demand"] = np.array([load_output / power_generation_scale for load_output in state["load_demand"]])

        # state["fire_distance"] = [1 - dist/self._considerable_fire_distance if dist < self._considerable_fire_distance else 0 for dist in state["fire_distance"]]

        fire_distance = []
        vulnerable_equipment = {}
        for i, dist in enumerate(state["fire_distance"]):
            if dist < self._considerable_fire_distance:
                val = round(1 - dist/self._considerable_fire_distance, 3)
                # if dist < 2.0:
                # if dist == 0.0:
                #     val = 1
                fire_distance.append(val)
                vulnerable_equipment[i] = val
            else:
                fire_distance.append(0)

        state["fire_distance"] = fire_distance

        num_bus = len(state["bus_status"])
        for i in range(num_bus):
            if state["bus_status"][i] == 0:
                fire_distance[i] = 1
                vulnerable_equipment[i] = 1

        for i in range(len(state["branch_status"])):
            if state["branch_status"][i] == 0:
                fire_distance[num_bus + i] = 1
                vulnerable_equipment[num_bus + i] = 1

        # print("bus_status:", state["bus_status"])
        # print("branch_status:", state["branch_status"])
        # print("generator_output:", state["generator_injection"])
        # print("load_demand:", state["load_demand"])
        # print("line_flow:", state["line_flow"])
        # print("theta:", state["theta"])
        # print("fire_distance:", state["fire_distance"])
        # print("fire_state:", state["fire_state"])

        if explore_network_flag == False:
            print("vulnerable equipment: ", vulnerable_equipment)

        return state


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
    def __init__(self, base_path):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._train_info_counter = 0
        self._step_info_counter = 0
        self._episodic_info_counter = 0
        self._generator_output_counter = 0
        self._load_demand_counter = 0
        self._line_flow_counter = 0

        train_info_dir = os.path.join(base_path, 'logs', current_time, '1_train_info')
        step_info_log_dir = os.path.join(base_path, 'logs', current_time, '2_step_info')
        episodic_info_dir = os.path.join(base_path, 'logs', current_time, '3_episodic_info')
        generator_output_log_dir = os.path.join(base_path, 'logs', current_time, '4_generator_output')
        load_demand_log_dir = os.path.join(base_path, 'logs', current_time, '5_load_demand')
        line_flow_log_dir = os.path.join(base_path, 'logs', current_time, '6_line_flow')

        self._train_info_writer = tf.summary.create_file_writer(train_info_dir)
        self._step_info_summary_writer = tf.summary.create_file_writer(step_info_log_dir)
        self._episodic_info_summary_writer = tf.summary.create_file_writer(episodic_info_dir)
        self._generator_output_summary_writer = tf.summary.create_file_writer(generator_output_log_dir)
        self._load_demand_summary_writer = tf.summary.create_file_writer(load_demand_log_dir)
        self._line_flow_summary_writer = tf.summary.create_file_writer(line_flow_log_dir)

    def train_info(self, info):
        with self._train_info_writer.as_default():
            tf.summary.scalar('train_info_1/1_reward_value', info.reward_value, step=self._train_info_counter)
            tf.summary.scalar('train_info_1/2_target_actor_actions', info.target_actor_actions, step=self._train_info_counter)
            tf.summary.scalar('train_info_1/3_target_critic_value_with_target_actor_actions', info.target_critic_value_with_target_actor_actions, step=self._train_info_counter)
            tf.summary.scalar('train_info_1/4_return_y', info.return_y, step=self._train_info_counter)
            tf.summary.scalar('train_info_2/1_original_actions', info.original_actions, step=self._train_info_counter)
            tf.summary.scalar('train_info_2/2_critic_value_with_original_actions', info.critic_value_with_original_actions, step=self._train_info_counter)
            tf.summary.scalar('train_info_2/3_critic_loss', info.critic_loss, step=self._train_info_counter)
            tf.summary.scalar('train_info_3/1_actor_actions', info.actor_actions, step=self._train_info_counter)
            tf.summary.scalar('train_info_3/2_critic_value_with_actor_actions', info.critic_value_with_actor_actions, step=self._train_info_counter)
            # tf.summary.scalar('train_info_3/3_load_loss', info.load_loss, step=self._train_info_counter)
            tf.summary.scalar('train_info_3/3_actor_loss', info.actor_loss, step=self._train_info_counter)
        self._train_info_counter += 1

    def step_info(self, info, reward_info):
        with self._step_info_summary_writer.as_default():
            tf.summary.scalar("step_info_1/1_nn_actions", info.nn_actions, step=self._step_info_counter)
            tf.summary.scalar("step_info_1/2_nn_actions_with_noise", info.nn_actions_with_noise, step=self._step_info_counter)
            tf.summary.scalar("step_info_1/3_env_actions", info.env_actions, step=self._step_info_counter)

            tf.summary.scalar("step_info_2/1_nn_critic_value", info.nn_critic_value, step=self._step_info_counter)
            tf.summary.scalar("step_info_2/2_nn_noise_critic_value", info.nn_noise_critic_value, step=self._step_info_counter)
            tf.summary.scalar("step_info_2/3_env_critic_value", info.env_critic_value, step=self._step_info_counter)

            tf.summary.scalar('step_info_3/1_total_load_demand', reward_info[0] * 10, step=self._step_info_counter)
            tf.summary.scalar('step_info_3/2_total_generator_output', reward_info[1] * 10, step=self._step_info_counter)
            tf.summary.scalar('step_info_3/3_reward', reward_info[2] * 10, step=self._step_info_counter)
            tf.summary.scalar("step_info_3/2_done", reward_info[3], step=self._step_info_counter)
        self._step_info_counter += 1

    def episodic_info(self, episodic_reward):
        with self._episodic_info_summary_writer.as_default():
            tf.summary.scalar("episodic_info/episodic_reward", episodic_reward, step=self._episodic_info_counter)
        self._episodic_info_counter += 1

    def generator_output_info(self, info):
        with self._generator_output_summary_writer.as_default():
            tf.summary.scalar('generator_output/generator_0', info[0] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_1', info[1] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_6', info[6] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_12', info[12] * 10, step=self._generator_output_counter)
            # tf.summary.scalar('generator_output/generator_13', info[13] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_14', info[14] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_15', info[15] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_17', info[17] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_20', info[20] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_21', info[21] * 10, step=self._generator_output_counter)
            tf.summary.scalar('generator_output/generator_22', info[22] * 10, step=self._generator_output_counter)
        self._generator_output_counter += 1

    def load_demand_info(self, info):
        with self._load_demand_summary_writer.as_default():
            tf.summary.scalar('load_demand/bus_0', info[0] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_1', info[1] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_2', info[2] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_3', info[3] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_4', info[4] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_5', info[5] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_6', info[6] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_7', info[7] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_8', info[8] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_9', info[9] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_10', info[10] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_11', info[11] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_12', info[12] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_13', info[13] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_14', info[14] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_15', info[15] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_16', info[16] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_17', info[17] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_18', info[18] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_19', info[19] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_20', info[20] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_21', info[21] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_22', info[22] * 10, step=self._load_demand_counter)
            tf.summary.scalar('load_demand/bus_23', info[23] * 10, step=self._load_demand_counter)
        self._load_demand_counter += 1

    def line_flow_info(self, info):
        with self._line_flow_summary_writer.as_default():
            tf.summary.scalar('line_flow/branch_0', info[0], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_1', info[1], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_2', info[2], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_3', info[3], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_4', info[4], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_5', info[5], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_6', info[6], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_7', info[7], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_8', info[8], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_9', info[9], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_10', info[10], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_11', info[11], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_12', info[12], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_13', info[13], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_14', info[14], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_15', info[15], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_16', info[16], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_17', info[17], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_18', info[18], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_19', info[19], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_20', info[20], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_21', info[21], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_22', info[22], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_23', info[23], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_24', info[24], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_25', info[25], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_26', info[26], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_27', info[27], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_28', info[28], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_29', info[29], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_30', info[30], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_31', info[31], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_32', info[32], step=self._line_flow_counter)
            tf.summary.scalar('line_flow/branch_33', info[33], step=self._line_flow_counter)
        self._line_flow_counter += 1

class SummaryWriter:
    def __init__(self, base_path, model_version, load_episode_num = 0, reactive_control = False):
        self._model_version = model_version
        self._reactive_control = reactive_control
        self._dir_name = os.path.join(base_path, "test_result")
        if load_episode_num == 0:
            self._file_name = os.path.join(self._dir_name, "fire_power_reward_list")
        else:
            self._file_name = os.path.join(self._dir_name, "fire_power_reward_list_ep_" + str(load_episode_num))

        self._create_dir()
        self._initialize()

    def _create_dir(self):
        try:
            os.makedirs(self._dir_name)
        except OSError as error:
            print(error)

    def _initialize(self):
        with open(f'{self._file_name}_v{self._model_version}.csv', 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(["model_version", "episode_number", "max_reached_step", "total_penalty", "load_loss",
                             "active_line_removal", "no_action_penalty", "violation_penalty"])

    def add_info(self, episode, max_reached_step, episodic_penalty, load_loss):
        active_line_removal_penalty = 0
        no_action_penalty = 0
        violation_penalty = 0

        if max_reached_step < 299:
            violation_penalty = -1653000

        if self._reactive_control:
            no_action_penalty = episodic_penalty - load_loss - violation_penalty
        else:
            active_line_removal_penalty = episodic_penalty - load_loss - violation_penalty

        with open(f'{self._file_name}_v{self._model_version}.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(self._model_version), str(episode), str(max_reached_step), str(episodic_penalty),
                             str(load_loss), str(active_line_removal_penalty), str(no_action_penalty), str(violation_penalty)])

