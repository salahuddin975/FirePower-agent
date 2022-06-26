import os
import csv
import random
import datetime
import copy
import numpy as np
import tensorflow as tf
from pypower.idx_brch import *
from scipy.optimize import Bounds, minimize, LinearConstraint
from scipy import stats, optimize


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
    def __init__(self, simulator_resources, generators, connected_components, state_spaces, action_spaces, power_generation_preprocess_scale):
        self.simulator_resources = simulator_resources
        self.generators = generators
        self._connected_components = connected_components
        self._state_spaces = state_spaces
        self._action_spaces = action_spaces
        self._power_generation_preprocess_scale = power_generation_preprocess_scale
        self._considerable_fire_distance = 5

        std_dev = .2
        self._ou_noise = OUActionNoise(mean=np.zeros(self.generators.get_num_generators()), std_deviation=np.ones(self.generators.get_num_generators()) * std_dev)

        self._branches = [(0, 1),(0, 2),(0, 4),(1, 3),(1, 5),(2, 8),(2, 23),(3, 8),(4, 9),(5, 9),(6, 7),(7, 8),(7, 9),(8, 10),
                          (8, 11),(9, 10),(9, 11),(10, 12),(10, 13),(11, 12),(11, 22),(12, 22),(13, 15),(14, 15),(14, 20),
                          (14, 23),(15, 16),(15, 18),(16, 17),(16, 21),(17, 20),(18, 19),(19, 22),(20, 21)]

        self.custom_writer = CustomWriter()

    # def _check_network_violations_branch(self, bus_status, branch_status):
    #     from_buses = self.simulator_resources.ppc["branch"][:, F_BUS].astype('int')
    #     to_buses = self.simulator_resources.ppc["branch"][:, T_BUS].astype('int')
    #
    #     for bus in range(bus_status.size):
    #         is_active = bus_status[bus]
    #         for branch in range(branch_status.size):
    #             if bus in [from_buses[branch], to_buses[branch]]:
    #                 if is_active == 0:
    #                     branch_status[branch] = 0
    #
    #     return branch_status
    #
    # def _adjust_load_demand_if_all_branches_out(self, branch_status, load_demand, current_output):
    #     bus_sets = [set() for _ in range(24)]
    #     for branch in self._branches:
    #         bus_sets[branch[0]].add(branch)
    #         bus_sets[branch[1]].add(branch)
    #
    #     for i in range(34):
    #         if branch_status[i] == 0:
    #             x, y = self._branches[i]
    #             bus_sets[x].remove((x,y))
    #             bus_sets[y].remove((x,y))
    #
    #             if len(bus_sets[x]) == 0 and x in self.generators.get_generators() and current_output[x] == 0:
    #                 load_demand[x] = 0
    #             elif len(bus_sets[x]) == 0 and x in self.generators.get_generators():
    #                 current_output[x] = load_demand[x]
    #                 self.generators.set_max_output(x, load_demand[x])
    #                 self.generators.set_min_output(x, load_demand[x])
    #                 self.generators.set_max_ramp(x, 0)
    #             elif len(bus_sets[x]) == 0:
    #                 load_demand[x] = 0
    #
    #             if len(bus_sets[y]) == 0 and y in self.generators.get_generators() and current_output[y] == 0:
    #                 load_demand[y] = 0
    #             elif len(bus_sets[y]) == 0 and y in self.generators.get_generators():
    #                 current_output[y] = load_demand[y]
    #                 self.generators.set_max_output(y, load_demand[y])
    #                 self.generators.set_min_output(y, load_demand[y])
    #                 self.generators.set_max_ramp(y, 0)
    #             elif len(bus_sets[y]) == 0:
    #                 load_demand[y] = 0

    # def add_heuristic_ramp(self, ramp, load_loss, num_generators, generators_current_output, generators_max_output, generators_max_ramp):
    #     for i in range(num_generators):
    #         ramp[i] = 0
    #         if load_loss > 0:
    #             if generators_current_output[i] < generators_max_output[i]:
    #                 ramp[i] = generators_max_output[i] - generators_current_output[i]
    #                 if ramp[i] > generators_max_ramp[i]:
    #                     ramp[i] = generators_max_ramp[i]
    #                 if ramp[i] > load_loss:
    #                     ramp[i] = load_loss
    #                 load_loss = load_loss - ramp[i]

    # def _clip_ramp_values1(self, nn_output, generators_output):     # previous way of calculating ramp
    #     # print("generators output: ", generators_output)
    #     # print("nn ratio output: ", nn_output)
    #
    #     num_generators = self.generators.get_num_generators()
    #     generators_current_output = np.zeros(num_generators)
    #     for i in range(num_generators):
    #         generators_current_output[i] = generators_output[self.generators.get_generators()[i]]
    #     # print("generators current output: ", generators_current_output)
    #
    #     # print("nn ramp: ", nn_ramp)
    #
    #     generators_max_output = self.generators.get_max_outputs()
    #     generators_min_output = self.generators.get_min_outputs()
    #     generators_max_ramp = self.generators.get_max_ramps()
    #
    #     # net_output =  nn_output * generators_max_output
    #     net_output = generators_min_output + nn_output * (generators_max_output - generators_min_output)
    #     # print ("network output: ", net_output)
    #
    #     ramp = net_output - generators_current_output
    #     # print("generators initial ramp: ", ramp)
    #
    #     for i in range(ramp.size):
    #         if ramp[i] > 0:
    #             ramp[i] = ramp[i] if ramp[i] < generators_max_ramp[i] else generators_max_ramp[i]
    #             ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] < generators_max_output[i] else generators_max_output[i] - generators_current_output[i]
    #         else:
    #             ramp[i] = ramp[i] if abs(ramp[i]) < generators_max_ramp[i] else -generators_max_ramp[i]
    #             ramp[i] = ramp[i] if ramp[i] + generators_current_output[i] > generators_min_output[i] else generators_min_output[i] - generators_current_output[i]
    #
    #         if abs(ramp[i]) < 0.00001:
    #             ramp[i] = 0.0
    #
    #     # print("generators set ramp: ", ramp)
    #     return ramp

    # def _check_bus_generator_violation(self, bus_status, nn_output, generators_current_output):
    #     selected_generators = self.generators.get_generators()
    #
    #     for bus in range(bus_status.size):
    #         flag = bus_status[bus]
    #         for j in range(selected_generators.size):
    #             gen_bus = selected_generators[j]
    #             if bus == gen_bus and flag == 0:
    #                 nn_output[j] = 0
    #                 self.generators.set_zero_for_generator(j)
    #                 generators_current_output[j] = 0

    # def check_violations(self, np_action, state, ramp_scale):
    #     bus_status = copy.deepcopy(state["bus_status"])
    #     branch_status = copy.deepcopy(state["branch_status"])
    #     load_demand = copy.deepcopy(state["load_demand"])
    #     generators_current_output = copy.deepcopy(state["generator_injection"])
    #     nn_output = np_action["generator_injection"]
    #
    #     # fire_distance = state["fire_distance"]
    #     # bus_status = np.ones(self._state_spaces[0])
    #     # for i in range(self._state_spaces[0]):
    #     #     if fire_distance[i] == 1:
    #     #         bus_status[i] = 0
    #
    #     # branch_status = np.ones(self._state_spaces[1])
    #     # for i in range(self._state_spaces[1]):
    #     #     if fire_distance[self._state_spaces[0] + i] == 1:
    #     #         branch_status[i] = 0
    #
    #     branch_status = self._check_network_violations_branch(bus_status, branch_status)
    #     bus_status = self._check_network_violations_bus(bus_status, branch_status)
    #     # print("bus status: ", bus_status)
    #     # print("branch status: ", branch_status)
    #
    #     ramp = self._clip_ramp_values(load_demand, generators_current_output, nn_output)
    #
    #     # ramp = self._clip_ramp_values(nn_output, generators_current_output)
    #     # ramp = self._check_bus_generator_violation(bus_status, ramp)
    #     # print("ramp: ", ramp)
    #
    #
    #     action = {
    #         "bus_status": bus_status,
    #         "branch_status": branch_status,
    #         "generator_selector": self.generators.get_generators(),
    #         "generator_injection": ramp,
    #     }
    #
    #     # action = {
    #     #     "bus_status": np.ones(24),
    #     #     "branch_status": np.ones(34),
    #     #     "generator_selector": np.array([24] * 10),
    #     #     "generator_injection": np.zeros(10, int),
    #     # }
    #
    #     return action

    def _clip_ramp_values(self, servable_load_demand, generators_current_output, nn_output):
        total_servable_load_demand = np.sum(servable_load_demand)
        generators_min_output = copy.deepcopy(self.generators.get_min_outputs())
        generators_max_output = copy.deepcopy(self.generators.get_max_outputs())
        generators_max_ramp = copy.deepcopy(self.generators.get_max_ramps())

        # print("nn_output_sum: ", np.sum(nn_output))
        epsilon_nn = 0.0001
        epsilon_total = 0.0001
        assert 1 + epsilon_nn > np.sum(nn_output) > 1-epsilon_nn, "Not total value is 1"
        assert np.min(nn_output) >= 0, "value is negative"

        for i in range(len(servable_load_demand)):
            if servable_load_demand[i] == 0.0:
                # print("generator ", self.generators.get_generators()[i], " output is 0; nn_output: ", nn_output[i])
                generators_max_ramp[i] = 0
                generators_min_output[i] = 0
                generators_max_output[i] = 0
                generators_current_output[i] = 0
                # nn_output[i] = 0

        lower = np.maximum(generators_current_output - generators_max_ramp, generators_min_output)
        upper = np.minimum(generators_current_output + generators_max_ramp, generators_max_output)

        # self.custom_writer.add_info(self.episode, self.step, total_servable_load_demand, np.sum(generators_current_output), np.sum(lower), np.sum(upper))

        # if np.sum(upper) - epsilon_total < total_servable_load_demand:
        #     # print("Adjust load demand: total_servable_load_demand: ", total_servable_load_demand, ", upper: ", np.sum(upper) - epsilon_total )
        #     total_servable_load_demand = np.sum(upper) - epsilon_total
        #
        # if np.sum(lower) + epsilon_total > total_servable_load_demand:
        #     # print("Adjust load demand: total_servable_load_demand: ", total_servable_load_demand, ", lower: ", np.sum(upper) - epsilon_total )
        #     total_servable_load_demand = np.sum(lower) + epsilon_total

        # print("lower: ", lower)
        # print("upper: ", upper)
        # print("generators max output: ", generators_max_output)
        # print("generators min output: ", generators_min_output)
        # print("generators max ramp: ", generators_max_ramp)
        # print("generators current output: ", generators_current_output)
        # print("servable load demand: ", servable_load_demand)
        # print("generators_current_output_total: ", np.sum(generators_current_output), "; lower_total: ", np.sum(lower),
        #       "; upper_total: ", np.sum(upper), "; total_servable_load_demand:", total_servable_load_demand)

        # if np.sum(nn_output):
        #     nn_output = nn_output / np.sum(nn_output)
        actor_output = nn_output * total_servable_load_demand * (1 - epsilon_total)

        total_load_demand_lower = np.array(total_servable_load_demand * (1 - epsilon_total))
        total_load_demand_upper = np.array(total_servable_load_demand * (1 - 0.5 * epsilon_total))
        linear_constraint = LinearConstraint(A=np.transpose(np.ones(len(generators_current_output))),
                                             lb=total_load_demand_lower, ub=total_load_demand_upper)
        # print("load_demand_total: ", total_servable_load_demand, "; load_demand_lower_total: ", total_load_demand_lower, "; load_demand_upper_total: ", total_load_demand_upper)

        assert (lower <= upper).all(), "lower, upper value constraint failed."
        assert np.sum(lower) <= total_load_demand_upper and total_load_demand_lower <= np.sum(upper), \
            f"{np.sum(lower)} <= {total_load_demand_upper} and {total_load_demand_lower} <= {np.sum(upper)} violation"

        feasible_output = minimize(lambda feasible_output: np.sum(np.power((actor_output - feasible_output), 2)),
                 generators_current_output, options={'verbose': 0},
                 bounds=[(lower[i], upper[i]) for i in range(len(upper))],
                 constraints=[linear_constraint], method='trust-constr')

        assert(total_load_demand_lower * (1 - epsilon_total) <= sum(feasible_output.x) < total_load_demand_upper), \
            f"feasible_output constraint violated: {total_load_demand_lower * (1 - epsilon_total)} <= {np.sum(feasible_output.x)} >= {total_load_demand_upper}"

        # print("feasible_output: ", np.sum(feasible_output.x))

        ramp = feasible_output.x - generators_current_output
        # print("generators set ramp: ", ramp)

        return ramp

    def process_nn_action(self, state, nn_action, explore_network, noise_range=0.5):
        self.episode = state["episode"]
        self.step = state["step"]
        bus_status = copy.deepcopy(state["bus_status"])
        branch_status = copy.deepcopy(state["branch_status"])
        current_output = state["generator_injection"]
        # print("current_output: ", current_output)

        # branch_status = self._check_network_violations_branch(bus_status, branch_status) # if bus is 0, then corresponding all branches are 0
        # self._adjust_load_demand_if_all_branches_out(branch_status, load_demand, current_output) # adjust load_demand and generation max output if all branches are 0

        # generators_current_output = np.zeros(self.generators.get_num_generators())
        # servable_load_demand = np.zeros(self.generators.get_num_generators())
        # for i in range(self.generators.get_num_generators()):
        #     generators_current_output[i] = current_output[self.generators.get_generators()[i]]
        #     servable_load_demand[i] = state["servable_load_demand"][self.generators.get_generators()[i]] / self._power_generation_preprocess_scale

        nn_output = np.array(tf.squeeze(nn_action))
        if explore_network:
            nn_output *= np.exp(self._ou_noise())
            nn_output = nn_output / np.sum(nn_output)
        # print("step: ", self.step, ", exploration: ", ((np.array(tf.squeeze(nn_action)) - nn_output)/nn_output) * 100)

        nn_noise_action = {
            "generator_injection": copy.deepcopy(nn_output),
        }

        connected_components = self._connected_components.get_connected_components()
        # print("connected_components: ", connected_components)

        ramp = np.zeros(nn_output.size)
        for connected_component in connected_components:
            servable_load_demand = np.zeros(self.generators.get_num_generators())
            generators_current_output = np.zeros(self.generators.get_num_generators())

            for i in connected_component:
                for j, gen in enumerate(self.generators.get_generators()):
                    if i == gen:
                        servable_load_demand[j] = state["servable_load_demand"][i] / self._power_generation_preprocess_scale
                        generators_current_output[j] = current_output[i]

            ramp_value = self._clip_ramp_values(servable_load_demand, generators_current_output, nn_output)
            # print("connected_component: ", connected_component)
            # print("ramp_value: ", ramp_value)
            ramp += ramp_value
        # print("ramp: ", ramp)

        env_action = {
            "episode": self.episode,
            "step_count": self.step,
            "action_type": "rl",
            "bus_status": bus_status,
            "branch_status": branch_status,
            "generator_selector": self.generators.get_generators(),
            "generator_injection": ramp * self._power_generation_preprocess_scale,
        }

        return nn_noise_action, env_action

    def get_myopic_action(self, episode, step):
        return {
            "episode": episode,
            "step_count": step,
            "action_type": "myopic",
            "bus_status": np.ones(24),
            "branch_status": np.ones(34),
            "generator_selector": [24] * 10,
            "generator_injection": np.zeros(10),
        }

    def get_target_myopic_action(self, episode, step):
        return {
            "episode": episode,
            "step_count": step,
            "action_type": "target_myopic",
            "bus_status": np.ones(24),
            "branch_status": np.ones(34),
            "generator_selector": [24] * 10,
            "generator_injection": np.zeros(10),
        }

    def preprocess(self, state, explore_network_flag):
        state["generator_injection"] = np.array([output / self._power_generation_preprocess_scale for output in state["generator_injection"]])
        state["load_demand"] = np.array([load_output / self._power_generation_preprocess_scale for load_output in state["load_demand"]])

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

        # if explore_network_flag == False:
        #     print("vulnerable equipment: ", vulnerable_equipment)

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

# class SummaryWriter:
#     def __init__(self, base_path, model_version, load_episode_num = 0, reactive_control = False):
#         self._model_version = model_version
#         self._reactive_control = reactive_control
#         self._dir_name = os.path.join(base_path, "test_result")
#         if load_episode_num == 0:
#             self._file_name = os.path.join(self._dir_name, "fire_power_reward_list")
#         else:
#             self._file_name = os.path.join(self._dir_name, "fire_power_reward_list_ep_" + str(load_episode_num))
#
#         self._create_dir()
#         self._initialize()
#
#     def _create_dir(self):
#         try:
#             os.makedirs(self._dir_name)
#         except OSError as error:
#             print(error)
#
#     def _initialize(self):
#         with open(f'{self._file_name}_v{self._model_version}.csv', 'w') as fd:
#             writer = csv.writer(fd)
#             writer.writerow(["model_version", "episode_number", "max_reached_step", "total_penalty", "load_loss",
#                              "active_line_removal", "no_action_penalty", "violation_penalty"])
#
#     def add_info(self, episode, max_reached_step, episodic_penalty, load_loss):
#         active_line_removal_penalty = 0
#         no_action_penalty = 0
#         violation_penalty = 0
#
#         if max_reached_step < 299:
#             violation_penalty = -1653000
#
#         if self._reactive_control:
#             no_action_penalty = episodic_penalty - load_loss - violation_penalty
#         else:
#             active_line_removal_penalty = episodic_penalty - load_loss - violation_penalty
#
#         with open(f'{self._file_name}_v{self._model_version}.csv', 'a') as fd:
#             writer = csv.writer(fd)
#             writer.writerow([str(self._model_version), str(episode), str(max_reached_step), str(episodic_penalty),
#                              str(load_loss), str(active_line_removal_penalty), str(no_action_penalty), str(violation_penalty)])


class EpisodicReward:
    def __init__(self, path, load_episode):
        self._path = os.path.join(path, "test_result")
        self._load_episode = load_episode
        self._create_dir()
        self._initialize()

    def _create_dir(self):
        try:
            os.makedirs(self._path)
        except OSError as error:
            print(error)

    def _initialize(self):
        with open(f'{self._path}/episodic_test_result_ep_{self._load_episode}', 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(["episode", "myopic", "myopic_reward_rl_transition",
                             "rl", "myopic_reward_rl_transition-myopic", "rl-myopic"])

    def add_info(self, episode, myopic, myopic_reward_rl_transition, rl):
        with open(f'{self._path}/episodic_test_result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(episode), str(myopic), str(myopic_reward_rl_transition), str(rl),
                             str(myopic_reward_rl_transition - myopic), str(rl - myopic)])


class StepByStepReward:
    def __init__(self, path, load_episode):
        self._path = os.path.join(path, "test_result")
        self._load_episode = load_episode
        self._create_dir()
        self._initialize()

    def _create_dir(self):
        try:
            os.makedirs(self._path)
        except OSError as error:
            print(error)

    def _initialize(self):
        with open(f'{self._path}/step_by_step_test_result_ep_{self._load_episode}', 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(["episode", "step", "myopic", "myopic_reward_rl_transition",
                             "rl", "myopic_reward_rl_transition-myopic", "rl-myopic"])

    def add_info(self, episode, step, myopic, myopic_reward_rl_transition, rl):
        with open(f'{self._path}/step_by_step_test_result.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(episode), str(step), str(myopic), str(myopic_reward_rl_transition), str(rl),
                             str(myopic_reward_rl_transition - myopic), str(rl - myopic)])



class CustomWriter:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        with open(f'gams_feasible.csv', 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(["episode", "step", "servable_load_demand", "generator_current_output",
                             "output_lower_bound", "output_upper_bound"])

    def add_info(self, episode, step, load_demand, current_output, lower_bound, upper_bound):
        with open(f'gams_feasible.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(episode), str(step), str(load_demand), str(current_output),
                             str(lower_bound), str(upper_bound)])

