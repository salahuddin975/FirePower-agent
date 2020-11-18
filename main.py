import os
import gym
import random
import argparse
import numpy as np
import tensorflow as tf
from agent import Agent
from replay_buffer import ReplayBuffer
from data_processor import DataProcessor, Tensorboard, SummaryWriter
from simulator_resorces import SimulatorResources, Generators


gym.logger.set_level(25)
np.set_printoptions(linewidth=300)

seed_value = 50
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
except:
    pass


def get_arguments():
    argument_parser = argparse.ArgumentParser(description="Dummy Agent for gym_firepower")
    argument_parser.add_argument('-g', '--path-geo', help="Full path to geo file", required=True)
    argument_parser.add_argument('-p', '--path-power', help="Full path to power systems file", required=False)
    argument_parser.add_argument('-f', '--scale-factor', help="Scali    actor_lr = 0.001ng factor", type=int, default=6)
    argument_parser.add_argument('-n', '--nonconvergence-penalty', help="Non-convergence penalty", type=float)
    argument_parser.add_argument('-a', '--protectionaction-penalty', help="Protection action penalty", type=float)
    argument_parser.add_argument('-s', '--seed', help="Seed for random number generator", type=int)
    argument_parser.add_argument('-o', '--path-output', help="Output directory for dumping environment data")

    parsed_args = argument_parser.parse_args()
    # print(parsed_args)
    return parsed_args


def get_state_spaces(observation_space):
    print("observation space: ", observation_space)

    num_st_bus = observation_space["bus_status"].shape[0]
    num_st_branch = observation_space["branch_status"].shape[0]
    # num_fire_status = observation_space["fire_status"].shape[0]
    num_fire_distance = observation_space["fire_distance"].shape[0]
    num_gen_output = observation_space["generator_injection"].shape[0]
    num_load_demand = observation_space["load_demand"].shape[0]
    num_theta = observation_space["theta"].shape[0]
    num_line_flow = observation_space["line_flow"].shape[0]
    state_spaces = [num_st_bus, num_st_branch, num_fire_distance, num_gen_output, num_load_demand, num_theta, num_line_flow]
    print(f"State Spaces: num bus: {num_st_bus}, num branch: {num_st_branch}, fire distance: {num_fire_distance}, "
          f"num_gen_injection: {num_gen_output}, num_load_demand: {num_load_demand}, num_theta: {num_theta}, num_line_flow: {num_line_flow}")

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
    args = get_arguments()
    base_path = "database_seed_" + str(seed_value)

    simulator_resources = SimulatorResources(power_file_path=args.path_power, geo_file_path=args.path_geo)
    generators = Generators(ppc=simulator_resources.ppc, ramp_frequency_in_hour=6)
    # generators.print_info()

    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power, num_tunable_gen=generators.get_num_generators())
    state_spaces = get_state_spaces(env.observation_space)
    action_spaces = get_action_spaces(env.action_space)

    # agent model
    save_model = True
    load_model = False
    save_model_version = 0
    load_model_version = 0
    load_episode_num = 0

    agent = Agent(base_path, state_spaces, action_spaces)
    if load_model:
        agent.load_weight(version=load_model_version, episode_num=load_episode_num)

    # replay buffer
    save_replay_buffer = True
    load_replay_buffer = False
    save_replay_buffer_version = 0
    load_replay_buffer_version = 0

    buffer = ReplayBuffer(base_path, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_version,
                          buffer_capacity=200000, batch_size=1024)

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

        episodic_reward = 0
        max_reached_step = 0
        # generators.set_max_outputs(state["generator_injection"])

        for step in range(max_steps_per_episode):
            tf_state = data_processor.get_tf_state(state)
            nn_action = agent.actor(tf_state)
            print("NN generator output: ", nn_action[0])

            net_action = data_processor.explore_network(nn_action, explore_network=explore_network_flag, noise_range=0.5)
            env_action = data_processor.check_violations(net_action, state["fire_distance"], state["generator_injection"])

            next_state, reward, done, _ = env.step(env_action)
            print(f"Episode: {episode}, at step: {step}, reward: {reward[0]}")

            buffer.add_record((state, net_action, reward, next_state, env_action))

            episodic_reward += reward[0]
            state = next_state

            if done or (step == max_steps_per_episode - 1):
                print(f"Episode: V{save_model_version}_{episode}, done at step: {step}, total reward: {episodic_reward}")
                max_reached_step = step
                break

            if train_network:
                # print ("Train at: ", episode)
                # for i in range(num_train_per_episode):
                state_batch, action_batch, reward_batch, next_state_batch = buffer.get_batch()
                critic_loss, reward_value, critic_value = agent.train(state_batch, action_batch, reward_batch, next_state_batch)
                tensorboard.add_critic_network_info(critic_loss, reward_value, critic_value)

        tensorboard.add_episodic_info(episodic_reward)
        summary_writer.add_info(episode, max_reached_step, episodic_reward)

        # explore / Testing
        if episode and (episode % 20 == 0):
            print("Start testing network at: ", episode)
            explore_network_flag = False
        if episode and (episode % 20 == 2):
            print("Start exploring network at: ", episode)
            explore_network_flag = True

        # save model weights
        if (episode % 20 == 0) and save_model:
            agent.save_weight(version=save_model_version, episode_num=episode)

        # save replay buffer
        if (episode % 20 == 0) and save_replay_buffer:
            print(f"Saving replay buffer at: {episode}")
            buffer.save_buffer(save_replay_buffer_version)