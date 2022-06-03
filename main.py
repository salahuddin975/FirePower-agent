import os
import gym
import random
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf
from parameters import Parameters
from agent import Agent
from replay_buffer import ReplayBuffer
from data_processor import DataProcessor, Tensorboard, SummaryWriter
from simulator_resorces import SimulatorResources, Generators
from fire_propagation_visualizer import Visualizer
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

    power_generation_preprocess_scale = 10
    simulator_resources = SimulatorResources(power_file_path=args.path_power, geo_file_path=args.path_geo)
    generators = Generators(ppc=simulator_resources.ppc, power_generation_preprocess_scale=power_generation_preprocess_scale, ramp_frequency_in_hour=6)
    # generators.print_info()

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power,
                        num_tunable_gen=generators.get_num_generators(), scaling_factor=1, seed=seed_value if train_network else 50)
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

    agent = Agent(base_path, state_spaces, action_spaces)
    if load_model:
        agent.load_weight(version=load_model_version, episode_num=load_episode_num)

    # replay buffer
    save_replay_buffer = True if train_network else False
    load_replay_buffer = False
    save_replay_buffer_version = 0
    load_replay_buffer_version = 0

    buffer = ReplayBuffer(base_path, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_version,
                          buffer_capacity=1000000, batch_size=parameters.batch_size)

    tensorboard = Tensorboard(base_path)
    visualizer = Visualizer(args.path_geo)
    summary_writer = SummaryWriter(base_path, save_model_version, load_episode_num)
    data_processor = DataProcessor(simulator_resources, generators, state_spaces, action_spaces)

    # agent training
    total_episode = 100001
    max_steps_per_episode = 300
    num_train_per_episode = 500         # canbe used by loading replay buffer
    episodic_rewards = []
    explore_network_flag = True if train_network else False

    for episode in range(total_episode):
        state = env.reset()

        max_reached_step = 0
        episodic_penalty = 0
        episodic_load_loss = 0

        state = data_processor.preprocess(state, power_generation_preprocess_scale, explore_network_flag)
        # if not parameters.generator_max_output:
        #     generators.set_max_outputs(state["generator_injection"])

        for step in range(max_steps_per_episode):
            # tensorboard.generator_output_info(state["generator_injection"])
            # tensorboard.load_demand_info(state["load_demand"])
            # tensorboard.line_flow_info(state["line_flow"])

            tf_state = data_processor.get_tf_state(state)
            nn_action = agent.actor(tf_state)
            # print("NN generator output: ", nn_action[0])
            # print("original:", agent.get_critic_value(tf_state, nn_action))

            nn_noise_action = data_processor.explore_network(nn_action, explore_network=explore_network_flag, noise_range=parameters.noise_rate)
            # print("original+noise:", agent.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(nn_noise_action["generator_injection"]), 0)))

            env_action = data_processor.check_violations(nn_noise_action, state["fire_distance"], state["generator_injection"], ramp_scale=power_generation_preprocess_scale)
            # print("original+noise+violation_check:", agent.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(env_action["generator_injection"]),0)))

            # print("ramp:", env_action['generator_injection'])
            env_action["episode"] = episode
            env_action["step_count"] = step

            next_state, reward, done, cells_info = env.step(env_action)

            image = visualizer.draw_map(episode, step, cells_info[0], cells_info[1], next_state["bus_status"], next_state["branch_status"], next_state["generator_injection"])
            image.save(f"fire_propagation_{episode}_{step}.png")

            main_loop_info = MainLoopInfo(tf.math.reduce_mean(nn_action), agent.get_critic_value(tf_state, nn_action),
                                          tf.math.reduce_mean(tf.expand_dims(tf.convert_to_tensor(nn_noise_action["generator_injection"]), 0)),
                                          agent.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(nn_noise_action["generator_injection"]), 0)),
                                          tf.math.reduce_mean(tf.expand_dims(tf.convert_to_tensor(env_action["generator_injection"]), 0)),
                                          agent.get_critic_value(tf_state, tf.expand_dims(tf.convert_to_tensor(env_action["generator_injection"]),0)))
            reward_info = (np.sum(state["load_demand"]), np.sum(state["generator_injection"]), reward[0], done)
            tensorboard.step_info(main_loop_info, reward_info)

            if explore_network_flag == False:
                print(f"Episode: {episode}, at step: {step}, load_demand: {np.sum(state['load_demand'])},"
                      f" generator_injection: {np.sum(state['generator_injection'])}, reward: {reward[0]}")

            next_state = data_processor.preprocess(next_state, power_generation_preprocess_scale, explore_network_flag)
            buffer.add_record((state, nn_noise_action, reward, next_state, env_action, done))

            episodic_penalty += reward[0]
            episodic_load_loss += reward[1]
            state = next_state

            if done or (step == max_steps_per_episode - 1):
                print(f"Episode: V{save_model_version}_{episode}, done at step: {step}, total reward: {episodic_penalty}, total_load_loss: {episodic_load_loss}")
                max_reached_step = step
                break

            if train_network and episode >= 3:
                state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch = buffer.get_batch()
                tensorboard_info = agent.train(state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch)
                tensorboard.train_info(tensorboard_info)
                # print("Episode:", episode, ", step: ", step, ", critic_value:", tensorboard_info.critic_value_with_original_action, ", critic_loss:", tensorboard_info.critic_loss)

        # if train_network and episode > 5:
        #     print ("Train at episode: ", episode)
        #     start_time = datetime.now()
        #     for i in range(num_train_per_episode):
        #         state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch = buffer.get_batch()
        #         critic_loss, reward_value, critic_value, action_quality = agent.train(state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch)
        #         tensorboard.add_critic_network_info(critic_loss, reward_value, critic_value, action_quality)
        #         if i % 1000 == 0:
        #             print("train at: ", i)
        #
        #     computation_time = (datetime.now() - start_time).total_seconds()
        #     print("Training_computation_time:", computation_time)

        tensorboard.episodic_info(episodic_penalty)
        summary_writer.add_info(episode, max_reached_step, episodic_penalty, episodic_load_loss)

        # explore / Testing
        if episode and (episode % parameters.test_after_episodes == 0):
            print("Start testing network at: ", episode)
            explore_network_flag = False
        if episode and (episode % parameters.test_after_episodes == 4):
            print("Start exploring network at: ", episode)
            explore_network_flag = True if train_network else False

        # save model weights
        if (episode % parameters.test_after_episodes == 0) and save_model and episode:
            agent.save_weight(version=save_model_version, episode_num=episode)

        # save replay buffer
        if (episode % parameters.test_after_episodes == 0) and save_replay_buffer and episode:
            print(f"Saving replay buffer at: {episode}")
            buffer.save_buffer(save_replay_buffer_version)