import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import namedtuple

class MixFeaturesLayer(layers.Layer):
    def __init__(self, num_features, feature_len):
        super(MixFeaturesLayer, self).__init__()
        self.num_features = num_features
        self.feature_len = feature_len

    def call(self, inputs):
        batch_size = len(inputs)
        t1 = tf.reshape(inputs, [batch_size, self.num_features, self.feature_len])
        t2 = tf.transpose(t1, perm=[0, 2, 1])
        return tf.reshape(t2, [batch_size, self.feature_len * self.num_features])

class SliceLayer(layers.Layer):
    def __init__(self, num_features, feature_len):
        super(SliceLayer, self).__init__()
        self.num_features = num_features
        self.feature_len = feature_len

    def call(self, inputs):
        all_sliced_inputs = []
        for i in range(self.feature_len):
            all_sliced_inputs.append(inputs[:, self.num_features * i : self.num_features * (i+1)])
        return all_sliced_inputs

TensorboardInfo = namedtuple("TensorboardInfo",
                             ["reward_value", "target_actor_actions", "target_critic_value_with_target_actor_actions",
                              "return_y", "original_actions", "critic_value_with_original_actions", "critic_loss",
                              "actor_actions", "critic_value_with_actor_actions", "actor_loss"])

class DDPG:
    def __init__(self, base_path, state_spaces, action_spaces, generators, seed):
        self._gamma = 0.9      # discount factor
        self._tau = 0.005       # used to update target network
        actor_lr = 0.001
        critic_lr = 0.002
        self._save_weight_directory = os.path.join(base_path, "trained_model")
        self._load_weight_directory = os.path.join(base_path, "trained_model")
        # self._load_weight_directory = os.path.join("../../FirePower-agent-private", f"database_seed_{seed}", "trained_model")
        self._create_dir()

        self._actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self._critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self._state_spaces = copy.deepcopy(state_spaces)
        self._action_spaces = copy.deepcopy(action_spaces)

        self._generators = generators
        self._num_of_active_generators = self._action_spaces[3]

        self.actor = self._actor_model()
        self._target_actor = self._actor_model()
        self._target_actor.set_weights(self.actor.get_weights())

        self._critic = self._critic_model()
        self._target_critic = self._critic_model()
        self._target_critic.set_weights(self._critic.get_weights())

    def get_critic_value(self, state, action):
        value = self._critic([state, action])
        return  np.array(value)[0][0]

    def _create_dir(self):
        try:
            os.makedirs(self._save_weight_directory)
        except OSError as error:
            print(error)

        try:
            os.makedirs(self._load_weight_directory)
        except OSError as error:
            print(error)

    def save_weight(self, version, episode_num):
        self.actor.save_weights(f"{self._save_weight_directory}/agent_actor{version}_{episode_num}.h5")
        self._critic.save_weights(f"{self._save_weight_directory}/agent_critic{version}_{episode_num}.h5")
        self._target_actor.save_weights(f"{self._save_weight_directory}/agent_target_actor{version}_{episode_num}.h5")
        self._target_critic.save_weights(f"{self._save_weight_directory}/agent_target_critic{version}_{episode_num}.h5")

    def load_weight(self, version, episode_num):
        self.actor.load_weights(f"{self._load_weight_directory}/agent_actor{version}_{episode_num}.h5")
        self._target_actor.load_weights(f"{self._load_weight_directory}/agent_target_actor{version}_{episode_num}.h5")
        self._critic.load_weights(f"{self._load_weight_directory}/agent_critic{version}_{episode_num}.h5")
        self._target_critic.load_weights(f"{self._load_weight_directory}/agent_target_critic{version}_{episode_num}.h5")
        print("weights are loaded successfully!")

    # @tf.function
    def train(self, state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch):
        # update critic network
        with tf.GradientTape() as tape:
            target_actor_actions = self._target_actor(next_state_batch)
            target_critic_values = self._target_critic([next_state_batch, target_actor_actions]) * (1 - episode_end_flag_batch)
            return_y = reward_batch + self._gamma * target_critic_values
            # y = reward_batch[0] + self._gamma * self._target_critic([next_state_batch, target_actor_actions]) * (1 - episode_end_flag_batch)
            # y = reward_batch[0] + reward_batch[1] +  reward_batch[2] +  reward_batch[3] +  reward_batch[4]

            critic_value_with_original_actions = self._critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(return_y - critic_value_with_original_actions))

        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        # update actor network
        with tf.GradientTape() as tape:
            actor_actions = self.actor(state_batch)
            critic_value_with_actor_actions = self._critic([state_batch, actor_actions])
            actor_loss = -1 * tf.math.reduce_mean(critic_value_with_actor_actions)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # self._update_target()
        self.update_target(self._target_actor.variables, self.actor.variables)
        self.update_target(self._target_critic.variables, self._critic.variables)

        return TensorboardInfo(tf.math.reduce_mean(reward_batch), tf.math.reduce_mean(target_actor_actions),
                tf.math.reduce_mean(target_critic_values), tf.math.reduce_mean(return_y),
                tf.math.reduce_mean(action_batch), tf.math.reduce_mean(critic_value_with_original_actions), critic_loss,
                tf.math.reduce_mean(actor_actions), tf.math.reduce_mean(critic_value_with_actor_actions), actor_loss)

    # @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self._tau + a * (1 - self._tau))

    # def _update_target(self):
    #     # update target critic network
    #     new_weights = []
    #     target_critic_weights = self._target_critic.weights
    #     for i, critic_weight in enumerate(self._critic.weights):
    #         new_weights.append(self._tau * critic_weight + (1 - self._tau) * target_critic_weights[i])
    #     self._target_critic.set_weights(new_weights)
    #
    #     # update target actor network
    #     new_weights = []
    #     target_actor_weights = self._target_actor.weights
    #     for i, actor_weight in enumerate(self.actor.weights):
    #         new_weights.append(self._tau * actor_weight + (1 - self._tau) * target_actor_weights[i])
    #     self._target_actor.set_weights(new_weights)

    def _actor_model(self):
        # bus -> MultiBinary(24)
        bus_input = layers.Input(shape=(self._state_spaces[0],))
        # bus_input1 = layers.Dense(32, activation="tanh") (bus_input)

        # num_branch -> MultiBinary(34)
        branch_input = layers.Input(shape=(self._state_spaces[1],))
        # branch_input1 = layers.Dense(32, activation="tanh") (branch_input)

        # fire_distance -> Box(58, )
        fire_distance_input = layers.Input(shape=(self._state_spaces[2],))
        # fire_distance_input1 = layers.Dense(64, activation="tanh") (fire_distance_input)

        # generator_injection -> Box(24, )
        gen_inj_input = layers.Input(shape=(self._state_spaces[3],))
        # gen_inj_input1 = layers.Dense(32, activation="tanh") (gen_inj_input)

        # load_demand -> Box(24, )
        load_demand_input = layers.Input(shape=(self._state_spaces[4], ))
        # load_demand_input1 = layers.Dense(32, activation="tanh") (load_demand_input)

        # theta -> Box(24, )
        theta_input = layers.Input(shape=(self._state_spaces[5], ))
        # theta_input1 = layers.Dense(32, activation="tanh") (theta_input)

        # line_flow -> Box(34, )
        line_flow_input = layers.Input(shape=(self._state_spaces[6], ))

        #-------------------------------------
        # st_bus_branch = layers.Concatenate() ([bus_input, branch_input])
        # st_bus_branch_layer1 = layers.Dense(64, activation="relu") (st_bus_branch)
        #
        # st_fire_distance_layer1 = layers.Dense(64, activation="relu") (fire_distance_input)
        #
        # st_bus_branch_fire_distance_comb = layers.Concatenate() ([st_bus_branch_layer1, st_fire_distance_layer1])
        # st_bus_branch_fire_distance_comb_layer1 = layers.Dense(128, activation="relu") (st_bus_branch_fire_distance_comb)
        #
        # # st_gen_combine = layers.Concatenate() ([st_gen_output, act_gen_injection])
        # st_gen_layer1 = layers.Dense(64, "relu") (gen_inj_input)
        # st_load_demand1 = layers.Dense(64, "relu") (load_demand_input)
        # st_line_flow_layer1 = layers.Dense(64, activation="relu") (line_flow_input)
        #
        # # st_gen_line_flow_combine = layers.Concatenate() ([st_gen_layer1, st_line_flow_layer1])
        # st_gen_line_flow_combine = layers.Concatenate() ([st_gen_layer1, st_load_demand1, st_line_flow_layer1])
        # st_gen_line_flow_layer1 = layers.Dense(128, activation="relu") (st_gen_line_flow_combine)

        # state = layers.Concatenate() ([st_bus_branch_fire_distance_comb_layer1, st_gen_line_flow_layer1])
        # -------------------------------------

        state = layers.Concatenate() ([fire_distance_input, gen_inj_input])
        # state = layers.Concatenate() ([fire_distance_input, gen_inj_input, load_demand_input])
        # state = layers.Concatenate() ([fire_distance_input, gen_inj_input, load_demand_input, line_flow_input])
        # state = layers.Concatenate() ([fire_distance_input, gen_inj_input, load_demand_input, line_flow_input, theta_input])

        # -------------------------------------

        hidden = layers.Dense(512, activation="relu") (state)
        hidden = layers.Dense(512, activation="relu") (hidden)
        gen_inj_output = layers.Dense(self._num_of_active_generators, activation="sigmoid") (hidden)

        model = tf.keras.Model([bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input, line_flow_input],
                               [gen_inj_output])
        return model

    def _critic_model(self):
        # bus -> MultiBinary(24)
        st_bus = layers.Input(shape=(self._state_spaces[0],))
        # st_bus1 = layers.Dense(32, activation="relu") (st_bus)

        # num_branch -> MultiBinary(34)
        st_branch = layers.Input(shape=(self._state_spaces[1],))
        # st_branch1 = layers.Dense(32, activation="relu") (st_branch)

        # fire_distance -> Box(58, )
        st_fire_distance = layers.Input(shape=(self._state_spaces[2],))
        # st_fire_distance1 = layers.Dense(64, activation="relu") (st_fire_distance)

        # generator_injection (output) -> Box(24, )
        st_gen_output = layers.Input(shape=(self._state_spaces[3],))                     # Generator current total output
        # st_gen_output1 = layers.Dense(32, activation="relu") (st_gen_output)

        # load_demand -> Box(24, )
        st_load_demand = layers.Input(shape=(self._state_spaces[4], ))
        # st_load_demand1 = layers.Dense(32, activation="relu") (st_load_demand)

        # theta -> Box(24, )
        st_theta = layers.Input(shape=(self._state_spaces[5], ))
        # st_theta1 = layers.Dense(30, activation="relu") (st_theta)

        # line_flow -> Box(34, )
        st_line_flow = layers.Input(shape=(self._state_spaces[6], ))

        # bus -> MultiBinary(24)
        act_bus = layers.Input(shape=(self._action_spaces[0],))
        # act_bus1 = layers.Dense(30, activation="relu") (act_bus)
        #
        # # num_branch -> MultiBinary(34)
        act_branch = layers.Input(shape=(self._action_spaces[1],))
        # act_branch1 = layers.Dense(30, activation="relu") (act_branch)

        # generator_injection -> Box(5, )
        act_gen_injection = layers.Input(shape=(self._action_spaces[3],))
        # act_gen_injection1 = layers.Dense(32, activation="relu") (act_gen_injection)          # power ramping up/down

        #-------------------------------------
        # st_bus_branch = layers.Concatenate() ([st_bus, st_branch])
        # st_bus_branch_layer1 = layers.Dense(64, activation="relu") (st_bus_branch)
        #
        # st_fire_distance_layer1 = layers.Dense(64, activation="relu") (st_fire_distance)
        #
        # st_bus_branch_fire_distance_comb = layers.Concatenate() ([st_bus_branch_layer1, st_fire_distance_layer1])
        # st_bus_branch_fire_distance_comb_layer1 = layers.Dense(128, activation="relu") (st_bus_branch_fire_distance_comb)
        #
        # st_gen_combine = layers.Concatenate() ([st_gen_output, act_gen_injection])
        # st_gen_layer1 = layers.Dense(64, "relu") (st_gen_combine)
        #
        # st_load_demand1 = layers.Dense(64, "relu") (st_load_demand)
        # st_line_flow_layer1 = layers.Dense(64, activation="relu") (st_line_flow)
        #
        # st_gen_line_flow_combine = layers.Concatenate() ([st_gen_layer1, st_load_demand1, st_line_flow_layer1])
        # st_gen_line_flow_layer1 = layers.Dense(128, activation="relu") (st_gen_line_flow_combine)

        # state = layers.Concatenate() ([st_bus_branch_fire_distance_comb_layer1, st_gen_line_flow_layer1])
        # -------------------------------------

        state = layers.Concatenate() ([st_fire_distance, st_gen_output, act_gen_injection])
        # state = layers.Concatenate() ([st_fire_distance, st_gen_output, st_load_demand, act_gen_injection])
        # state = layers.Concatenate() ([st_fire_distance, st_gen_output, st_load_demand, st_line_flow, act_gen_injection])
        # state = layers.Concatenate() ([st_fire_distance, st_gen_output, st_load_demand, st_line_flow, st_theta, act_gen_injection])

        # -------------------------------------

        hidden = layers.Dense(512, activation="relu") (state)
        hidden = layers.Dense(512, activation="relu") (hidden)
        reward = layers.Dense(1, activation="linear") (hidden)

        model = tf.keras.Model([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta, st_line_flow,
                                act_gen_injection], reward)

        return model