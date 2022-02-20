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


gym.logger.set_level(25)
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


if __name__ == "__main__":
    args = get_arguments()
    seed_value = args.seed
    print(args)

    set_seed(seed_value)
    set_gpu_memory_limit()
    base_path = "database_seed_" + str(seed_value)

    simulator_resources = SimulatorResources(power_file_path=args.path_power, geo_file_path=args.path_geo)
    generators = Generators(ppc=simulator_resources.ppc, ramp_frequency_in_hour=6)
    # generators.print_info()

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power,
                        num_tunable_gen=generators.get_num_generators(), scaling_factor=1, seed=seed_value)
    state_spaces = get_state_spaces(env.observation_space)
    action_spaces = get_action_spaces(env.action_space)

    # agent model
    save_model = True
    load_model = False
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
    save_replay_buffer = True
    load_replay_buffer = False
    save_replay_buffer_version = 0
    load_replay_buffer_version = 0

    buffer = ReplayBuffer(base_path, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_version,
                          buffer_capacity=2000000, batch_size=parameters.batch_size)

    tensorboard = Tensorboard(base_path)
    summary_writer = SummaryWriter(base_path, save_model_version)
    data_processor = DataProcessor(simulator_resources, generators, state_spaces, action_spaces)

    # agent training
    total_episode = 100001
    max_steps_per_episode = 300
    num_train_per_episode = 1000         # canbe used by loading replay buffer
    episodic_rewards = []
    train_network = True
    explore_network_flag = True

    for episode in range(total_episode):
        state = env.reset()

        max_reached_step = 0
        episodic_penalty = 0
        episodic_load_loss = 0

        if not parameters.generator_max_output:
            generators.set_max_outputs(state["generator_injection"])

        for step in range(max_steps_per_episode):
            print("load_demand:", np.sum(state["load_demand"]), ", generator_injection:", np.sum(state["generator_injection"]) )

            tf_state = data_processor.get_tf_state(state)
            nn_action = agent.actor(tf_state)
            # print("NN generator output: ", nn_action[0])

            net_action = data_processor.explore_network(nn_action, explore_network=explore_network_flag, noise_range=parameters.noise_rate)
            env_action = data_processor.check_violations(net_action, state["fire_distance"], state["generator_injection"])

            # print("ramp:", env_action['generator_injection'])
            next_state, reward, done, _ = env.step(env_action)

            # if done:
            #     new_reward = reward
            # else:
            #     new_reward = (reward[0] + (28.5-np.sum(state["load_demand"])) * 100, reward[1])
            #     print(f"Episode: {episode}, at step: {step}, reward: {reward[0]}", ", new:", new_reward[0])

            print(f"Episode: {episode}, at step: {step}, reward: {reward[0]}")
            buffer.add_record((state, net_action, reward, next_state, env_action, not done))

            episodic_penalty += reward[0]
            episodic_load_loss += reward[1]
            state = next_state

            if done or (step == max_steps_per_episode - 1):
                print(f"Episode: V{save_model_version}_{episode}, done at step: {step}, total reward: {episodic_penalty}")
                max_reached_step = step
                break

        if train_network and episode > 5:
            print ("Train at episode: ", episode)
            start_time = datetime.now()
            for i in range(num_train_per_episode):
                state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch = buffer.get_batch()
                critic_loss, reward_value, critic_value, action_quality = agent.train(state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch)
                tensorboard.add_critic_network_info(critic_loss, reward_value, critic_value, action_quality)
                if i % 2000 == 0:
                    print("train at: ", i)

            computation_time = (datetime.now() - start_time).total_seconds()
            print("Training_computation_time:", computation_time)

        tensorboard.add_episodic_info(episodic_penalty)
        summary_writer.add_info(episode, max_reached_step, episodic_penalty, episodic_load_loss)

        # explore / Testing
        if episode and (episode % parameters.test_after_episodes == 0):
            print("Start testing network at: ", episode)
            explore_network_flag = False
        if episode and (episode % parameters.test_after_episodes == 4):
            print("Start exploring network at: ", episode)
            explore_network_flag = True

        # save model weights
        if (episode % parameters.test_after_episodes == 0) and save_model and episode:
            agent.save_weight(version=save_model_version, episode_num=episode)

        # save replay buffer
        if (episode % parameters.test_after_episodes == 0) and save_replay_buffer and episode:
            print(f"Saving replay buffer at: {episode}")
            buffer.save_buffer(save_replay_buffer_version)