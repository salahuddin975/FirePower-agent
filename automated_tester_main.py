import os
import gym
import random
import csv
import numpy as np
import tensorflow as tf
from parameters import Parameters
from agent import Agent
from replay_buffer import ReplayBuffer
from data_processor import DataProcessor
from simulator_resorces import SimulatorResources, Generators


gym.logger.set_level(50)
np.set_printoptions(linewidth=300)
power_path = "./assets/case24_ieee_rts.py"
geo_path = "./configurations/configuration.json"


class ResultWriter:
    def __init__(self, base_path, model_version, episode_num, file_name="_test_result", is_summary=False):
        self._is_summary = is_summary
        self._model_version = model_version
        self._dir_name = os.path.join(base_path, "test_result", "test_result")
        self._file_name = os.path.join(self._dir_name, f"{episode_num}{file_name}")

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
            if self._is_summary == False:
                writer.writerow(["model_version", "episode_number", "max_reached_step", "total_penalty", "total_load_loss"])
            else:
                writer.writerow(["model_version", "trained_agent_episode", "violation_count", "avg_penalty", "avg_load_loss"])

    def add_info(self, episode, max_reached_step_or_violation_count, episodic_reward, load_loss):
        with open(f'{self._file_name}_v{self._model_version}.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([str(self._model_version), str(episode), str(max_reached_step_or_violation_count), str(episodic_reward), str(load_loss)])

    def delete_file(self):
        os.remove(f'{self._file_name}_v{self._model_version}.csv')


def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print("Set seed: ", seed_value)


def set_gpu_memory_limit():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    except:
        print("Couldn't set GPU memory limit!")


def get_state_spaces(observation_space):
    # print("observation space: ", observation_space)

    num_st_bus = observation_space["bus_status"].shape[0]
    num_st_branch = observation_space["branch_status"].shape[0]
    # num_fire_status = observation_space["fire_status"].shape[0]
    num_fire_distance = observation_space["fire_distance"].shape[0]
    num_gen_output = observation_space["generator_injection"].shape[0]
    num_load_demand = observation_space["load_demand"].shape[0]
    num_theta = observation_space["theta"].shape[0]
    num_line_flow = observation_space["line_flow"].shape[0]
    state_spaces = [num_st_bus, num_st_branch, num_fire_distance, num_gen_output, num_load_demand, num_theta, num_line_flow]
    # print(f"State Spaces: num bus: {num_st_bus}, num branch: {num_st_branch}, fire distance: {num_fire_distance}, "
    #       f"num_gen_injection: {num_gen_output}, num_load_demand: {num_load_demand}, num_theta: {num_theta}, num_line_flow: {num_line_flow}")

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


def main(seed, num_of_generator, load_model_version=0, load_episode_num=0):
    seed_value = seed
    set_seed(50)

    set_gpu_memory_limit()
    base_path = "database_seed_" + str(seed_value)

    simulator_resources = SimulatorResources(power_file_path=power_path, geo_file_path=geo_path)
    generators = Generators(ppc=simulator_resources.ppc, ramp_frequency_in_hour=6)

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=geo_path, network_file=power_path,
                        num_tunable_gen=num_of_generator, scaling_factor=1, seed=50)

    state_spaces = get_state_spaces(env.observation_space)
    action_spaces = get_action_spaces(env.action_space)

    parameters = Parameters(base_path, load_model_version, geo_path)
    parameters.print_parameters()

    agent = Agent(base_path, state_spaces, action_spaces, simulator_resources)
    agent.load_weight(version=load_model_version, episode_num=load_episode_num)

    result_writer = ResultWriter(base_path, load_model_version, load_episode_num)
    data_processor = DataProcessor(simulator_resources, generators, state_spaces, action_spaces)

    # agent training
    total_episode = 7
    max_steps_per_episode = 300
    explore_network_flag = False
    episodic_rewards = []
    episodic_load_losses = []
    violation_count = 0

    for episode in range(total_episode):
        state = env.reset()

        max_reached_step = 0
        episodic_reward = 0
        episodic_load_loss = 0

        if not parameters.generator_max_output:
            generators.set_max_outputs(state["generator_injection"])

        for step in range(max_steps_per_episode):
            tf_state = data_processor.get_tf_state(state)
            nn_action = agent.actor(tf_state)

            net_action = data_processor.explore_network(nn_action, explore_network=explore_network_flag, noise_range=parameters.noise_rate)
            env_action = data_processor.check_violations(net_action, state["fire_distance"], state["generator_injection"])

            next_state, reward, done, _ = env.step(env_action)
            # print(f"Episode: {episode}, at step: {step}, reward: {reward[0]}")

            episodic_reward += reward[0]
            episodic_load_loss += reward[1]

            state = next_state

            if done: violation_count += 1
            if done or (step == max_steps_per_episode - 1):
                print(f"Episode: V{load_model_version}_{episode}, done at step: {step}, total reward: {episodic_reward}, total_load_loss: {episodic_load_loss}")
                max_reached_step = step
                break

        result_writer.add_info(episode, max_reached_step, episodic_reward, episodic_load_loss)

        if episode>1:
            episodic_rewards.append(episodic_reward)
            episodic_load_losses.append(episodic_load_loss)

    return sum(episodic_rewards)/5, violation_count, sum(episodic_load_losses)/5
