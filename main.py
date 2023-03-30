import os
import gym
import random
import argparse
import copy
import numpy as np
from datetime import datetime
import tensorflow as tf
from parameters import Parameters
from ddpg import DDPG
from replay_buffer import ReplayBuffer
from data_processor import DataProcessor, Tensorboard, EpisodicReward, StepByStepReward, GeneratorsOutput
from simulator_resorces import SimulatorResources, Generators
from fire_propagation_visualizer import Visualizer
from connected_components import ConnectedComponents
from collections import namedtuple


gym.logger.set_level(50)
np.set_printoptions(linewidth=300)


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


def get_arguments():
    argument_parser = argparse.ArgumentParser(description="Dummy Agent for gym_firepower")
    argument_parser.add_argument('-g', '--path-geo', help="Full path to geo file", required=True)
    argument_parser.add_argument('-p', '--path-power', help="Full path to power systems file", required=True)
    argument_parser.add_argument('-s', '--seed', help="Seed for agent", type=int, required=True)
    # argument_parser.add_argument('-b', '--replay-buffer-loading-seed', help="Replay buffer loading seed for agent", type=int, required=True)

    argument_parser.add_argument('-f', '--scale-factor', help="Scali    actor_lr = 0.001ng factor", type=int, default=6)
    argument_parser.add_argument('-n', '--nonconvergence-penalty', help="Non-convergence penalty", type=float)
    argument_parser.add_argument('-a', '--protectionaction-penalty', help="Protection action penalty", type=float)
    argument_parser.add_argument('-o', '--path-output', help="Output directory for dumping environment data")

    parsed_args = argument_parser.parse_args()
    # print(parsed_args)
    return parsed_args


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
    # print (f"Action Spaces: num bus: {num_bus}, num branch: {num_branch}, num_generator_selector: {num_generator_selector}, "
    #         f"num generator injection: {num_generator_injection}")

    return action_spaces


class LoadOutByFire:
    def __init__(self, simulator_resources, power_generation_preprocess_scale):
        self.total_load_out = 0
        self._bus_status = np.ones(24)
        self._power_generation_preprocess_scale = power_generation_preprocess_scale
        self._simulator_resources = simulator_resources

    def get_total_load_out_by_fire(self, bus_status):
        if (self._bus_status != bus_status).any():
            self._bus_status = copy.deepcopy(bus_status)
            self.total_load_out = 0
            for i, status in enumerate(bus_status):
                if status == 0:
                    self.total_load_out += self._simulator_resources.get_load_demand()[i]

        return self.total_load_out * self._power_generation_preprocess_scale


MainLoopInfo = namedtuple("MainLoopInfo", ["nn_actions", "nn_critic_value",
                                           "nn_actions_with_noise", "nn_noise_critic_value",
                                           "env_actions", "env_critic_value"])

if __name__ == "__main__":
    args = get_arguments()
    seed_value = args.seed
    print(args)

    train_network = True      # change during testing

    set_seed(seed_value if train_network else 50)
    set_gpu_memory_limit()
    base_path = "database_seed_" + str(seed_value)

    ramp_frequency_in_hour = 12
    power_generation_preprocess_scale = 10
    simulator_resources = SimulatorResources(power_file_path=args.path_power, power_generation_preprocess_scale=power_generation_preprocess_scale)
    generators = Generators(ppc=simulator_resources.ppc, power_generation_preprocess_scale=power_generation_preprocess_scale, ramp_frequency_in_hour=ramp_frequency_in_hour)
    connected_components = ConnectedComponents(generators)
    load_out_by_fire = LoadOutByFire(simulator_resources, power_generation_preprocess_scale)
    # generators.print_info()

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power,
                        num_tunable_gen=generators.get_num_generators(), scaling_factor=1, sampling_duration=1/ramp_frequency_in_hour, seed=seed_value if train_network else 50)
    state_spaces = get_state_spaces(env.observation_space)
    action_spaces = get_action_spaces(env.action_space)

    # agent model
    save_model = True if train_network else False
    load_model = False if train_network else True
    save_model_version = 0
    load_model_version = 0
    load_episode_num = 0

    parameters = Parameters(base_path, save_model_version, args.path_geo)
    parameters.save_parameters()
    parameters.print_parameters()

    ddpg = DDPG(base_path, state_spaces, action_spaces, generators)
    if load_model:
        ddpg.load_weight(version=load_model_version, episode_num=load_episode_num)

    # replay buffer
    save_replay_buffer = True if train_network else False
    load_replay_buffer = False
    save_replay_buffer_version = 0
    load_replay_buffer_version = 0

    buffer = ReplayBuffer(base_path, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_version,
                          buffer_capacity=1000000, batch_size=parameters.batch_size)

    tensorboard = Tensorboard(base_path)
    visualizer = Visualizer(generators, simulator_resources, args.path_geo)
    data_processor = DataProcessor(simulator_resources, generators, connected_components, state_spaces, action_spaces, power_generation_preprocess_scale)

    # agent training
    total_episode = 100001
    max_steps_per_episode = 300
    num_train_per_episode = 500         # canbe used by loading replay buffer
    episodic_rewards = []
    explore_network_flag = True if train_network else False

    episodic_reward = EpisodicReward(base_path, 0 if train_network else load_episode_num)
    step_by_step_reward = StepByStepReward(base_path, 0 if train_network else load_episode_num)
    if not train_network:
        generators_output_rl = GeneratorsOutput(base_path, "rl", 0 if train_network else load_episode_num)
        generators_output_myopic = GeneratorsOutput(base_path, "myopic", 0 if train_network else load_episode_num)

    for episode in range(total_episode):
        connected_components.reset()
        generators.reset()
        state = env.reset()

        max_reached_step = 0
        total_myopic_reward = 0
        total_myopic_reward_rl_transition = 0
        total_rl_reward = 0
        total_custom_reward = 0

        state = data_processor.preprocess(state, explore_network_flag)
        if not train_network:
            generators_output_myopic.add_info(episode, 0, state["generator_injection"] * power_generation_preprocess_scale)
            generators_output_rl.add_info(episode, 0, state["generator_injection"] * power_generation_preprocess_scale)

        for step in range(max_steps_per_episode):
            if not train_network:
                tensorboard.generator_output_info(state["generator_injection"])
                tensorboard.load_demand_info(state["load_demand"])
                tensorboard.line_flow_info(state["line_flow"])

            myopic_action = data_processor.get_myopic_action(episode, step)
            myopic_next_state, myopic_reward, myopic_done, _ = env.step(myopic_action)

            myopic_action_rl_transition = data_processor.get_target_myopic_action(episode, step)
            target_myopic_next_state, myopic_reward_rl_transition, target_myopic_done, _ = env.step(myopic_action_rl_transition)

            if not train_network:
                generators_output_myopic.add_info(episode, step+1, myopic_next_state["generator_injection"])
                generators_output_rl.add_info(episode, step+1, target_myopic_next_state["generator_injection"])

            # servable_load_demand = np.sum(target_myopic_state["generator_injection"])/power_generation_preprocess_scale
            # print(f"Episode: {episode}, at step: {step}, load_demand: {np.sum(state['load_demand'])}, generator_injection: {np.sum(state['generator_injection'])}, "
            #     f"servable_load_demand: {servable_load_demand}, diff: {(np.sum(state['load_demand']) - servable_load_demand) * 10}")

            tf_state = data_processor.get_tf_state(state)
            nn_action = ddpg.actor(tf_state)

            state["episode"] = episode
            state["step"] = step
            state["servable_load_demand"] = target_myopic_next_state["generator_injection"]
            connected_components.update_connected_components(state)
            nn_noise_action, env_action = data_processor.process_nn_action(state, nn_action, explore_network=explore_network_flag, noise_range=parameters.noise_rate)

            # print("ramp:", env_action['generator_injection'])
            next_state, rl_reward, done, cells_info = env.step(env_action)

            # image = visualizer.draw_map(episode, step, cells_info, next_state)
            # image.save(f"fire_propagation_{episode}_{step}.png")

            # main_loop_info = MainLoopInfo(tf.math.reduce_mean(nn_action), ddpg.get_critic_value(tf_state, nn_action),
            #                               tf.math.reduce_mean(tf.expand_dims(tf.convert_to_tensor(nn_noise_action["generator_injection"]), 0)),
            #                               ddpg.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(nn_noise_action["generator_injection"]), 0)),
            #                               tf.math.reduce_mean(tf.expand_dims(tf.convert_to_tensor(env_action["generator_injection"]), 0)),
            #                               ddpg.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(env_action["generator_injection"]), 0)))
            # reward_info = (np.sum(state["load_demand"]), np.sum(state["generator_injection"]), rl_reward[0], done)
            # tensorboard.step_info(main_loop_info, reward_info)

            reward = np.sum(next_state["generator_injection"]) - np.sum(myopic_next_state["generator_injection"])
            custom_reward = (reward, reward)

            total_load_out_by_fire = load_out_by_fire.get_total_load_out_by_fire(state["bus_status"])
            myopic_reward = myopic_reward[0] + total_load_out_by_fire
            myopic_reward_rl_transition = myopic_reward_rl_transition[0] + total_load_out_by_fire
            rl_reward = rl_reward[0] + total_load_out_by_fire

            total_myopic_reward += myopic_reward
            total_myopic_reward_rl_transition += myopic_reward_rl_transition
            total_rl_reward += rl_reward
            total_custom_reward += custom_reward[0]

            if True or explore_network_flag == False:
                print(f"Episode: {episode}, at step: {step}, myopic_reward: {myopic_reward}, target_myopic_reward: "
                      f"{myopic_reward_rl_transition}, rl_reward: {rl_reward}, custom_reward: {reward}, load_out_by_fire: {total_load_out_by_fire}")
            step_by_step_reward.add_info(episode, step, round(myopic_reward, 2), round(myopic_reward_rl_transition, 2), round(rl_reward, 2))

            next_state = data_processor.preprocess(next_state, explore_network_flag)
            buffer.add_record((state, nn_noise_action, custom_reward, next_state, env_action, done))
            state = next_state

            if done or (step == max_steps_per_episode - 1):
                print(f"Episode: V{save_model_version}_{episode}, done at step: {step}, total myopic_reward: {total_myopic_reward},"
                      f" total_target_myopic_reward: {total_myopic_reward_rl_transition}, total_rl_reward: {total_rl_reward}, total_custom_reward: {total_custom_reward}")
                max_reached_step = step
                break

            if train_network and episode >= 3:
                state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch = buffer.get_batch()
                tensorboard_info = ddpg.train(state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch)
                tensorboard.train_info(tensorboard_info)
                # print("Episode:", episode, ", step: ", step, ", critic_value:", tensorboard_info.critic_value_with_original_action, ", critic_loss:", tensorboard_info.critic_loss)

        tensorboard.episodic_info(total_rl_reward)
        episodic_reward.add_info(episode, round(total_myopic_reward, 2), round(total_myopic_reward_rl_transition, 2), round(total_rl_reward, 2))

        # explore / Testing
        if episode and (episode % parameters.test_after_episodes == 0):
            print("Start testing network at: ", episode)
            explore_network_flag = False
        if episode and (episode % parameters.test_after_episodes == 4):
            print("Start exploring network at: ", episode)
            explore_network_flag = True if train_network else False

        # save model weights
        if (episode % parameters.test_after_episodes == 0) and save_model and episode:
            ddpg.save_weight(version=save_model_version, episode_num=episode)

        # save replay buffer
        if (episode % parameters.test_after_episodes == 0) and save_replay_buffer and episode:
            print(f"Saving replay buffer at: {episode}")
            buffer.save_buffer(save_replay_buffer_version)