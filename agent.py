import gym
import argparse
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from gym import wrappers, logger
import os

gym.logger.set_level(40)



class Memory:
    def __init__(self, state_spaces, action_spaces, buffer_capacity=100000, batch_size=64):
        self.counter = 0
        self.capacity = buffer_capacity
        self.batch_size = batch_size

        # current state buffer
        self.st_bus =np.zeros((state_spaces[0]))
        self.st_branch =np.zeros((state_spaces[1]))
        self.st_fire =np.zeros((state_spaces[2], state_spaces[2]))
        self.st_generator_output =np.zeros((state_spaces[3]))
        self.st_load_demand =np.zeros((state_spaces[4]))
        self.st_theta =np.zeros((state_spaces[5]))
        self.state = np.array((self.st_bus, self.st_branch, self.st_fire, self.st_generator_output, self.st_load_demand, self.st_theta))
        self.states = np.array([self.state]*self.capacity)

        # action buffer
        self.act_bus = np.zeros((action_spaces[0]))
        self.act_branch = np.zeros((action_spaces[1]))
        self.act_generator_selector = np.zeros((action_spaces[2]))
        self.act_generator_injection = np.zeros((action_spaces[3]))
        self.action = np.array((self.act_bus, self.act_branch, self.act_generator_selector, self.act_generator_injection))
        self.actions = np.array([self.action]*self.capacity)

        # reward buffer
        self.rewards = np.zeros((self.capacity, 1))

        # next state buffer
        self.next_states = np.array([self.state]*self.capacity)


    def add_record(self, record):
        index = self.counter % self.capacity

        self.states[index][0] = np.copy(record[0]["bus_status"])
        self.states[index][1] = np.copy(record[0]["branch_status"])
        self.states[index][2] = np.copy(record[0]["fire_state"])
        self.states[index][3] = np.copy(record[0]["generator_injection"])
        self.states[index][4] = np.copy(record[0]["load_demand"])
        self.states[index][5] = np.copy(record[0]["theta"])

        self.actions[index][0] = np.copy(record[1]["bus_status"])
        self.actions[index][1] = np.copy(record[1]["branch_status"])
        self.actions[index][2] = np.copy(record[1]["branch_status"])
        self.actions[index][3] = np.copy(record[1]["branch_status"])

        self.rewards[index] = record[2]

        self.next_states[index][0] = np.copy(record[3]["bus_status"])
        self.next_states[index][1] = np.copy(record[3]["branch_status"])
        self.next_states[index][2] = np.copy(record[3]["fire_state"])
        self.next_states[index][3] = np.copy(record[3]["generator_injection"])
        self.next_states[index][4] = np.copy(record[3]["load_demand"])
        self.next_states[index][5] = np.copy(record[3]["theta"])

        self.counter = self.counter + 1

    def learn(self):
        record_size = min(self.capacity, self.counter)
        batch_indices = np.random.choice(record_size, self.batch_size)

        bus_status_batch = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
        for i, index in enumerate(batch_indices):
            bus_tensor = tf.convert_to_tensor(self.states[index][0], dtype=tf.int8)
            print ("i: ", i, ", index: ", bus_tensor)
            # bus_status_batch = bus_status_batch.write(i, bus_tensor)

        # state_batch = tf.convert_to_tensor(self.states[batch_indices][0])
        # action_batch = tf.convert_to_tensor(self.actions[batch_indices])
        # reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        # next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices])






def get_actor(state_space, action_space):
    # bus -> MultiBinary(24)
    bus_input = layers.Input(shape=(state_space[0],))
    bus_input1 = layers.Dense(30, activation="relu") (bus_input)

    # num_branch -> MultiBinary(34)
    branch_input = layers.Input(shape=(state_space[1],))
    branch_input1 = layers.Dense(30, activation="relu") (branch_input)

    # fire_status -> Box(350, 350)
    fire_input = layers.Input(shape=(state_space[2], state_space[2]))
    fire_input1 = layers.Flatten()(fire_input)
    fire_input1 = layers.Dense(500, activation="relu") (fire_input1)

    # generator_injection -> Box(24, )
    gen_inj_input = layers.Input(shape=(state_space[3],))
    gen_inj_input1 = layers.Dense(30, activation="relu") (gen_inj_input)

    # load_demand -> Box(24, )
    load_demand_input = layers.Input(shape=(state_space[4], ))
    load_demand_input1 = layers.Dense(30, activation="relu") (load_demand_input)

    # theta -> Box(24, )
    theta_input = layers.Input(shape=(state_space[5], ))
    theta_input1 = layers.Dense(30, activation="relu") (theta_input)

    state = layers.Concatenate() ([bus_input1, branch_input1, fire_input1, gen_inj_input1, load_demand_input1, theta_input1])
    hidden = layers.Dense(512, activation="relu") (state)
    hidden = layers.Dense(512, activation="relu") (hidden)
    hidden = layers.Dense(512, activation="relu") (hidden)

    # bus -> MultiBinary(24)
    bus_output = layers.Dense(action_space[0]) (hidden)

    # num_branch -> MultiBinary(34)
    branch_output = layers.Dense(action_space[1]) (hidden)

    # generator_selector -> MultiDiscrete([12 12 12 12 12])
    gen_selector_output = layers.Dense(action_space[2]) (hidden)

    # generator_injection (generator output) -> Box(5, )
    gen_inj_output = layers.Dense(action_space[3]) (hidden)

    model = tf.keras.Model([bus_input, branch_input, fire_input, gen_inj_input, load_demand_input, theta_input], [bus_output, branch_output, gen_selector_output, gen_inj_output])
    return model


def get_tf_state(state):
    tf_bus_status = tf.expand_dims(tf.convert_to_tensor(state["bus_status"]), 0)
    tf_branch_status = tf.expand_dims(tf.convert_to_tensor(state["branch_status"]), 0)
    tf_fire_state = tf.expand_dims(tf.convert_to_tensor(state["fire_state"]), 0)
    tf_generator_injection = tf.expand_dims(tf.convert_to_tensor(state["generator_injection"]), 0)
    tf_load_demand = tf.expand_dims(tf.convert_to_tensor(state["load_demand"]), 0)
    tf_theta = tf.expand_dims(tf.convert_to_tensor(state["theta"]), 0)

    return [tf_bus_status, tf_branch_status, tf_fire_state, tf_generator_injection, tf_load_demand, tf_theta]


def get_np_action(tf_action):
    bus_status = np.array(tf_action[0])
    bus_status[: 1] = bus_status[:] > 0
    bus_status = np.squeeze(bus_status.astype(int))
    # print ("bus status: ", bus_status)

    branch_status = np.array(tf_action[1])
    branch_status[: 1] = branch_status[:] > 0
    branch_status = np.squeeze(branch_status.astype(int))
    # print ("branch status: ", branch_status)

    gen_selector = np.array(tf.squeeze(tf_action[2]))
    gen_selector = np.abs(gen_selector * 24)
    gen_selector = gen_selector.astype(int)
    # print("gen selector: ", gen_selector)

    gen_injection = np.array(tf.squeeze(tf_action[3]))      # need to take into range (need to talk Subir/Ajay)
    # print("gen injection: ", gen_injection)

    action = {"generator_injection": gen_injection,
        "branch_status": branch_status,
        "bus_status": bus_status,
        "generator_selector": gen_selector}

    return action


def get_critic(state_spaces, action_spaces):
    # bus -> MultiBinary(24)
    st_bus = layers.Input(shape=(state_spaces[0],))
    st_bus1 = layers.Dense(30, activation="relu") (st_bus)

    # num_branch -> MultiBinary(34)
    st_branch = layers.Input(shape=(state_spaces[1],))
    st_branch1 = layers.Dense(30, activation="relu") (st_branch)

    # fire_status -> Box(350, 350)
    st_fire = layers.Input(shape=(state_spaces[2], state_spaces[2]))
    st_fire1 = layers.Flatten()(st_fire)
    st_fire1 = layers.Dense(500, activation="relu") (st_fire1)

    # generator_injection (output) -> Box(24, )
    st_gen_output = layers.Input(shape=(state_spaces[3],))                     # Generator current total output
    st_gen_output1 = layers.Dense(30, activation="relu") (st_gen_output)

    # load_demand -> Box(24, )
    st_load_demand = layers.Input(shape=(state_spaces[4], ))
    st_load_demand1 = layers.Dense(30, activation="relu") (st_load_demand)

    # theta -> Box(24, )
    st_theta = layers.Input(shape=(state_spaces[5], ))
    st_theta1 = layers.Dense(30, activation="relu") (st_theta)

    # bus -> MultiBinary(24)
    act_bus = layers.Input(shape=(action_spaces[0],))
    act_bus1 = layers.Dense(30, activation="relu") (act_bus)

    # num_branch -> MultiBinary(34)
    act_branch = layers.Input(shape=(action_spaces[1],))
    act_branch1 = layers.Dense(30, activation="relu") (act_branch)

    # generator_selector -> MultiDiscrete([12 12 12 12 12])
    act_gen_selector = layers.Input(shape=(action_spaces[2],))
    act_gen_selector1 = layers.Dense(30, activation="relu") (act_gen_selector)

    # generator_injection -> Box(5, )
    act_gen_injection = layers.Input(shape=(action_spaces[3],))
    act_gen_injection1 = layers.Dense(30, activation="relu") (act_gen_injection)          # power ramping up/down

    state = layers.Concatenate() ([st_bus1, st_branch1, st_fire1, st_gen_output1, st_load_demand1, st_theta1])
    action = layers.Concatenate() ([act_bus1, act_branch1, act_gen_selector1, act_gen_injection1])
    hidden = layers.Concatenate() ([state, action])

    hidden = layers.Dense(512, activation="relu") (hidden)
    hidden = layers.Dense(512, activation="relu") (hidden)
    reward = layers.Dense(1) (hidden)

    model = tf.keras.Model([st_bus, st_branch, st_fire, st_gen_output, st_load_demand, st_theta, act_bus, act_branch, act_gen_selector, act_gen_injection], reward)
    return model


def  get_tf_critic_input(state, action):
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

    return [st_bus_status, st_branch_status, st_fire_state, st_generator_output, st_load_demand, st_theta, act_bus_status, act_branch_status, act_generator_selector, act_generator_injection]


class ReplayBuffer:
    def __init__(self, actor, target_actor, critic, target_critic, state_spaces, action_spaces, buffer_capacity=100000, batch_size=64):
        self.counter = 0
        self.gamma = 0.99      # discount factor
        actor_lr = 0.001
        critic_lr = 0.002
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.capacity = buffer_capacity
        self.batch_size = batch_size

        self.st_bus = np.zeros((self.capacity, state_spaces[0]))
        self.st_branch = np.zeros((self.capacity, state_spaces[1]))
        self.st_fire = np.zeros((self.capacity, state_spaces[2], state_spaces[2]))
        self.st_gen_output = np.zeros((self.capacity, state_spaces[3]))
        self.st_load_demand = np.zeros((self.capacity, state_spaces[4]))
        self.st_theta = np.zeros((self.capacity, state_spaces[5]))

        self.act_bus = np.zeros((self.capacity, action_spaces[0]))
        self.act_branch = np.zeros((self.capacity, action_spaces[1]))
        self.act_gen_selector = np.zeros((self.capacity, action_spaces[2]))
        self.act_gen_injection = np.zeros((self.capacity, action_spaces[3]))

        self.rewards = np.zeros((self.capacity, 1))

        self.next_st_bus = np.zeros((self.capacity, state_spaces[0]))
        self.next_st_branch = np.zeros((self.capacity, state_spaces[1]))
        self.next_st_fire = np.zeros((self.capacity, state_spaces[2], state_spaces[2]))
        self.next_st_gen_output = np.zeros((self.capacity, state_spaces[3]))
        self.next_st_load_demand = np.zeros((self.capacity, state_spaces[4]))
        self.next_st_theta = np.zeros((self.capacity, state_spaces[5]))


    def add_record(self, record):
        index = self.counter % self.capacity

        self.st_bus[index] = np.copy(record[0]["bus_status"])
        self.st_branch[index] = np.copy(record[0]["branch_status"])
        self.st_fire[index] = np.copy(record[0]["fire_state"])
        self.st_gen_output[index] = np.copy(record[0]["generator_injection"])
        self.st_load_demand[index] = np.copy(record[0]["load_demand"])
        self.st_theta[index] = np.copy(record[0]["theta"])

        self.act_bus[index] = np.copy(record[1]["bus_status"])
        self.act_branch[index] = np.copy(record[1]["branch_status"])
        self.act_gen_selector[index] = np.copy(record[1]["generator_selector"])
        self.act_gen_injection[index] = np.copy(record[1]["generator_injection"])

        self.rewards[index] = record[2]

        self.next_st_bus[index] = np.copy(record[3]["bus_status"])
        self.next_st_branch[index] = np.copy(record[3]["branch_status"])
        self.next_st_fire[index] = np.copy(record[3]["fire_state"])
        self.next_st_gen_output[index] = np.copy(record[3]["generator_injection"])
        self.next_st_load_demand[index] = np.copy(record[3]["load_demand"])
        self.next_st_theta[index] = np.copy(record[3]["theta"])

        self.counter = self.counter + 1


    def learn(self):
        record_size = min(self.capacity, self.counter)
        batch_indices = np.random.choice(record_size, self.batch_size)

        st_tf_bus = tf.convert_to_tensor(self.st_bus[batch_indices])
        st_tf_branch = tf.convert_to_tensor(self.st_branch[batch_indices])
        st_tf_fire = tf.convert_to_tensor(self.st_fire[batch_indices])
        st_tf_gen_output = tf.convert_to_tensor(self.st_gen_output[batch_indices])
        st_tf_load_demand = tf.convert_to_tensor(self.st_load_demand[batch_indices])
        st_tf_theta = tf.convert_to_tensor(self.st_theta[batch_indices])

        act_tf_bus = tf.convert_to_tensor(self.act_bus[batch_indices])
        act_tf_branch = tf.convert_to_tensor(self.act_branch[batch_indices])
        act_tf_gen_selector = tf.convert_to_tensor(self.act_gen_selector[batch_indices])
        act_tf_gen_injection = tf.convert_to_tensor(self.act_gen_injection[batch_indices])

        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)

        next_st_tf_bus = tf.convert_to_tensor(self.next_st_bus[batch_indices])
        next_st_tf_branch = tf.convert_to_tensor(self.next_st_branch[batch_indices])
        next_st_tf_fire = tf.convert_to_tensor(self.next_st_fire[batch_indices])
        next_st_tf_gen_output = tf.convert_to_tensor(self.next_st_gen_output[batch_indices])
        next_st_tf_load_demand = tf.convert_to_tensor(self.next_st_load_demand[batch_indices])
        next_st_tf_theta = tf.convert_to_tensor(self.next_st_theta[batch_indices])

        # update critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor([next_st_tf_bus, next_st_tf_branch, next_st_tf_fire, next_st_tf_gen_output, next_st_tf_load_demand, next_st_tf_theta])
            # need to check if target action needs to be converted
            y = reward_batch + self.gamma * self.target_critic([next_st_tf_bus, next_st_tf_branch, next_st_tf_fire, next_st_tf_gen_output, next_st_tf_load_demand, next_st_tf_theta, target_actions])
            critic_value = self.critic([st_tf_bus, st_tf_branch, st_tf_fire, st_tf_gen_output, st_tf_load_demand, st_tf_theta, act_tf_bus, act_tf_branch, act_tf_gen_selector, act_tf_gen_injection])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # update actor network
        with tf.GradientTape() as tape:
            actions = self.actor([st_tf_bus, st_tf_branch, st_tf_fire, st_tf_gen_output, st_tf_load_demand, st_tf_theta])
            # need to check if target action needs to be converted
            critic_value = self.critic([st_tf_bus, st_tf_branch, st_tf_fire, st_tf_gen_output, st_tf_load_demand, st_tf_theta, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))


def get_state_spaces(env):
    observation_space = env.observation_space
    num_st_bus = observation_space["bus_status"].shape[0]
    num_st_branch = observation_space["branch_status"].shape[0]
    num_fire_status = observation_space["fire_status"].shape[0]
    num_gen_output = observation_space["generator_injection"].shape[0]
    num_load_demand = observation_space["load_demand"].shape[0]
    num_theta = observation_space["theta"].shape[0]
    state_spaces = [num_st_bus, num_st_branch, num_fire_status, num_gen_output, num_load_demand, num_theta]
    # print(f"State Spaces: num bus: {num_st_bus}, num branch: {num_st_branch}, fire status: {num_fire_status}, num_gen_injection: {num_gen_output}, num_load_demand: {num_load_demand}, num_theta: {num_theta}")

    return state_spaces


def get_action_spaces(env):
    action_space = env.action_space
    num_bus = action_space["bus_status"].shape[0]
    num_branch = action_space["branch_status"].shape[0]
    num_generator_selector = action_space["generator_selector"].shape[0]
    num_generator_injection = action_space["generator_injection"].shape[0]
    action_spaces = [num_bus, num_branch, num_generator_selector, num_generator_injection]
    # print (f"Action Spaces: num bus: {num_bus}, num branch: {num_branch}, num_generator_selector: {num_generator_selector}, num generator injection: {num_generator_injection}")

    return action_spaces





def main(args):
    env = gym.envs.make("gym_firepower:firepower-v0", geo_file=args.path_geo, network_file=args.path_power)
    # print("action_space: ", env.action_space)
    # print("observation space: ", env.observation_space)

    state_spaces = get_state_spaces(env)
    action_spaces = get_action_spaces(env)

    actor = get_actor(state_spaces, action_spaces)
    critic = get_critic(state_spaces, action_spaces)

    state = env.reset()
    tf_state = get_tf_state(state)
    actor_action = actor(tf_state)
    action = get_np_action(actor_action)
    # print ("action: ", action)

    tf_critic_input = get_tf_critic_input(state, action)
    critic_reward = critic(tf_critic_input)
    # print("critic_reward: ", critic_reward)

    buffer = ReplayBuffer(actor, actor, critic, critic, state_spaces, action_spaces, 5, 5)
    buffer.add_record((state, action, 1, state))
    buffer.learn()


    # memory = Memory(state_spaces, action_spaces, 5, 5)
    # memory.add_record((state, action, 1, state))
    # memory.learn()


    # action = env.action_space.sample()
    # ob, reward, done, _ = env.step(action)
    # print("reward: ", reward)


    # print("sample action: ", action)
    # print("obs: ", ob)



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
    print(args)
    main(args)

