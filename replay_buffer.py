import os
import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, base_path, state_spaces, action_spaces, load_replay_buffer, load_replay_buffer_version=0, buffer_capacity=200000, batch_size=256):
        self._counter = 0
        self._capacity = buffer_capacity
        self._batch_size = batch_size
        self._load_replay_buffer_dir = os.path.join(base_path, "replay_buffer")
        self._save_replay_buffer_dir = os.path.join(base_path, "replay_buffer")
        self._create_dir()

        if not load_replay_buffer:
            self._initialize_buffer(state_spaces, action_spaces)
        else:
            self._load_buffer(load_replay_buffer_version)

    def _create_dir(self):
        try:
            os.makedirs(self._load_replay_buffer_dir)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._save_replay_buffer_dir)
        except OSError as error:
            print(error)

    def _initialize_buffer(self, state_spaces, action_spaces):
        self.st_bus = np.zeros((self._capacity, state_spaces[0]))
        self.st_branch = np.zeros((self._capacity, state_spaces[1]))
        self.st_fire_distance = np.zeros((self._capacity, state_spaces[2]))
        self.st_gen_output = np.zeros((self._capacity, state_spaces[3]))
        self.st_load_demand = np.zeros((self._capacity, state_spaces[4]))
        self.st_theta = np.zeros((self._capacity, state_spaces[5]))
        self.st_line_flow = np.zeros((self._capacity, state_spaces[6]))

        self.act_bus = np.zeros((self._capacity, action_spaces[0]))
        self.act_branch = np.zeros((self._capacity, action_spaces[1]))
        self.act_gen_injection = np.zeros((self._capacity, action_spaces[3]))

        self.rewards = np.zeros((self._capacity, 1))
        self.episode_end_flag = np.zeros((self._capacity, 1), dtype=bool)

        self.next_st_bus = np.zeros((self._capacity, state_spaces[0]))
        self.next_st_branch = np.zeros((self._capacity, state_spaces[1]))
        self.next_st_fire_distance = np.zeros((self._capacity, state_spaces[2]))
        self.next_st_gen_output = np.zeros((self._capacity, state_spaces[3]))
        self.next_st_load_demand = np.zeros((self._capacity, state_spaces[4]))
        self.next_st_theta = np.zeros((self._capacity, state_spaces[5]))
        self.next_st_line_flow = np.zeros((self._capacity, state_spaces[6]))

        self.np_counter = np.zeros((1))

    def save_buffer(self, version):
        np.save(f'{self._save_replay_buffer_dir}/st_bus_v{version}.npy', self.st_bus)
        np.save(f'{self._save_replay_buffer_dir}/st_branch_v{version}.npy', self.st_branch)
        np.save(f'{self._save_replay_buffer_dir}/st_fire_distance_v{version}.npy', self.st_fire_distance)
        np.save(f'{self._save_replay_buffer_dir}/st_gen_output_v{version}.npy', self.st_gen_output)
        np.save(f'{self._save_replay_buffer_dir}/st_load_demand_v{version}.npy', self.st_load_demand)
        np.save(f'{self._save_replay_buffer_dir}/st_theta_v{version}.npy', self.st_theta)
        np.save(f'{self._save_replay_buffer_dir}/st_line_flow_v{version}.npy', self.st_line_flow)

        np.save(f'{self._save_replay_buffer_dir}/act_bus_v{version}.npy', self.act_bus)
        np.save(f'{self._save_replay_buffer_dir}/act_branch_v{version}.npy', self.act_branch)
        np.save(f'{self._save_replay_buffer_dir}/act_gen_injection_v{version}.npy', self.act_gen_injection)

        np.save(f'{self._save_replay_buffer_dir}/rewards_v{version}.npy', self.rewards)
        np.save(f'{self._save_replay_buffer_dir}/episode_end_flag_v{version}.npy', self.episode_end_flag)

        np.save(f'{self._save_replay_buffer_dir}/next_st_bus_v{version}.npy', self.next_st_bus)
        np.save(f'{self._save_replay_buffer_dir}/next_st_branch_v{version}.npy', self.next_st_branch)
        np.save(f'{self._save_replay_buffer_dir}/next_st_fire_distance_v{version}.npy', self.next_st_fire_distance)
        np.save(f'{self._save_replay_buffer_dir}/next_st_gen_output_v{version}.npy', self.next_st_gen_output)
        np.save(f'{self._save_replay_buffer_dir}/next_st_load_demand_v{version}.npy', self.next_st_load_demand)
        np.save(f'{self._save_replay_buffer_dir}/next_st_theta_v{version}.npy', self.next_st_theta)
        np.save(f'{self._save_replay_buffer_dir}/next_st_line_flow_v{version}.npy', self.next_st_line_flow)

        self.np_counter[0] = self._counter
        np.save(f'{self._save_replay_buffer_dir}/counter_v{version}.npy', self.np_counter)

    def _load_buffer(self, version):
        print("Loading replay buffer!")
        self.st_bus = np.load(f'{self._load_replay_buffer_dir}/st_bus_v{version}.npy')
        self.st_branch = np.load(f'{self._load_replay_buffer_dir}/st_branch_v{version}.npy')
        self.st_fire_distance = np.load(f'{self._load_replay_buffer_dir}/st_fire_distance_v{version}.npy')
        self.st_gen_output = np.load(f'{self._load_replay_buffer_dir}/st_gen_output_v{version}.npy')
        self.st_load_demand = np.load(f'{self._load_replay_buffer_dir}/st_load_demand_v{version}.npy')
        self.st_theta = np.load(f'{self._load_replay_buffer_dir}/st_theta_v{version}.npy')
        self.st_line_flow = np.load(f'{self._load_replay_buffer_dir}/st_line_flow_v{version}.npy')

        self.act_bus = np.load(f'{self._load_replay_buffer_dir}/act_bus_v{version}.npy')
        self.act_branch = np.load(f'{self._load_replay_buffer_dir}/act_branch_v{version}.npy')
        self.act_gen_injection = np.load(f'{self._load_replay_buffer_dir}/act_gen_injection_v{version}.npy')

        self.rewards = np.load(f'{self._load_replay_buffer_dir}/rewards_v{version}.npy')
        self.episode_end_flag = np.load(f'{self._load_replay_buffer_dir}/episode_end_flag_v{version}.npy')

        self.next_st_bus = np.load(f'{self._load_replay_buffer_dir}/next_st_bus_v{version}.npy')
        self.next_st_branch = np.load(f'{self._load_replay_buffer_dir}/next_st_branch_v{version}.npy')
        self.next_st_fire_distance = np.load(f'{self._load_replay_buffer_dir}/next_st_fire_distance_v{version}.npy')
        self.next_st_gen_output = np.load(f'{self._load_replay_buffer_dir}/next_st_gen_output_v{version}.npy')
        self.next_st_load_demand = np.load(f'{self._load_replay_buffer_dir}/next_st_load_demand_v{version}.npy')
        self.next_st_theta = np.load(f'{self._load_replay_buffer_dir}/next_st_theta_v{version}.npy')
        self.next_st_line_flow = np.load(f'{self._load_replay_buffer_dir}/next_st_line_flow_v{version}.npy')

        self.np_counter = np.load(f'{self._load_replay_buffer_dir}/counter_v{version}.npy')
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
        self.st_line_flow[index] = np.copy(record[0]["line_flow"])

        # use data from heuristic
        self.act_bus[index] = np.copy(record[4]["bus_status"])
        self.act_branch[index] = np.copy(record[4]["branch_status"])
        # use data from NN actor
        self.act_gen_injection[index] = np.copy(record[1]["generator_injection"])

        self.rewards[index] = record[2][0]
        self.episode_end_flag[index] = record[5]

        self.next_st_bus[index] = np.copy(record[3]["bus_status"])
        self.next_st_branch[index] = np.copy(record[3]["branch_status"])
        self.next_st_fire_distance[index] = np.copy(record[3]["fire_distance"])
        self.next_st_fire_distance[index] = self.next_st_fire_distance[index] / 100
        self.next_st_gen_output[index] = np.copy(record[3]["generator_injection"])
        self.next_st_load_demand[index] = np.copy(record[3]["load_demand"])
        self.next_st_theta[index] = np.copy(record[3]["theta"])
        self.next_st_line_flow[index] = np.copy(record[3]["line_flow"])

        self._counter = self._counter + 1

    def get_batch(self):
        record_size = self.get_num_records()
        batch_indices = np.random.choice(record_size, self._batch_size)

        st_tf_bus = tf.convert_to_tensor(self.st_bus[batch_indices])
        st_tf_branch = tf.convert_to_tensor(self.st_branch[batch_indices])
        st_tf_fire_distance = tf.convert_to_tensor(self.st_fire_distance[batch_indices])
        st_tf_gen_output = tf.convert_to_tensor(self.st_gen_output[batch_indices])
        st_tf_load_demand = tf.convert_to_tensor(self.st_load_demand[batch_indices])
        st_tf_theta = tf.convert_to_tensor(self.st_theta[batch_indices])
        st_tf_line_flow = tf.convert_to_tensor(self.st_line_flow[batch_indices])

        act_tf_bus = tf.convert_to_tensor(self.act_bus[batch_indices])
        act_tf_branch = tf.convert_to_tensor(self.act_branch[batch_indices])
        act_tf_gen_injection = tf.convert_to_tensor(self.act_gen_injection[batch_indices])

        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        episode_end_flag_batch = tf.convert_to_tensor(self.episode_end_flag[batch_indices], dtype=tf.float32)

        next_st_tf_bus = tf.convert_to_tensor(self.next_st_bus[batch_indices])
        next_st_tf_branch = tf.convert_to_tensor(self.next_st_branch[batch_indices])
        next_st_tf_fire_distance = tf.convert_to_tensor(self.next_st_fire_distance[batch_indices])
        next_st_tf_gen_output = tf.convert_to_tensor(self.next_st_gen_output[batch_indices])
        next_st_tf_load_demand = tf.convert_to_tensor(self.next_st_load_demand[batch_indices])
        next_st_tf_theta = tf.convert_to_tensor(self.next_st_theta[batch_indices])
        next_st_tf_line_flow = tf.convert_to_tensor(self.next_st_line_flow[batch_indices])

        state_batch = [st_tf_bus, st_tf_branch, st_tf_fire_distance, st_tf_gen_output, st_tf_load_demand, st_tf_theta, st_tf_line_flow]
        action_batch = [act_tf_bus, act_tf_branch, act_tf_gen_injection]
        next_state_batch = [next_st_tf_bus, next_st_tf_branch, next_st_tf_fire_distance, next_st_tf_gen_output,
                                    next_st_tf_load_demand, next_st_tf_theta, next_st_tf_line_flow]

        return state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch
