import os
import copy
import tensorflow as tf
from tensorflow.keras import layers

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

class SliceFireDistanceLayer(layers.Layer):
    def __init__(self, bus_size):
        super(SliceFireDistanceLayer, self).__init__()
        self.bus_size = bus_size

    def call(self, inputs):
        return inputs[:, :self.bus_size], inputs[:, self.bus_size:]

class SelectGeneratorsLayer(layers.Layer):
    def __init__(self):
        super(SelectGeneratorsLayer, self).__init__()
        self.indices = [0, 1, 6, 12, 13, 14, 15, 17, 20, 21, 22]

    def call(self, inputs):
        return tf.gather(inputs, indices=self.indices, axis=1)

class Agent:
    def __init__(self, base_path, state_spaces, action_spaces):
        self._gamma = 0.9      # discount factor
        self._tau = 0.01       # used to update target network
        actor_lr = 0.001
        critic_lr = 0.001
        self.mini_hidden_layer_size = 32
        self._save_weight_directory = os.path.join(base_path, "trained_model")
        self._load_weight_directory = os.path.join(base_path, "trained_model")
        # self._load_weight_directory = os.path.join("../../FirePower-agent-private", base_path, "trained_model")
        self._create_dir()

        self._actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self._critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self._state_spaces =  copy.deepcopy(state_spaces)
        self._action_spaces =  copy.deepcopy(action_spaces)

        self.actor = self._actor_model()
        self._target_actor = self._actor_model()
        self._target_actor.set_weights(self.actor.get_weights())

        self._critic = self._critic_model()
        self._target_critic = self._critic_model()
        self._target_critic.set_weights(self._critic.get_weights())

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

    def train(self, state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch):
        # update critic network
        with tf.GradientTape() as tape:
            target_actor_actions = self._target_actor(next_state_batch)
            y = reward_batch[0] + self._gamma * self._target_critic([next_state_batch, target_actor_actions]) * episode_end_flag_batch
            # y = reward_batch[0] + reward_batch[1] + reward_batch[2] + reward_batch[3] + reward_batch[4]
            critic_value = self._critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        # update actor network
        with tf.GradientTape() as tape:
            actor_actions = self.actor(state_batch)
            critic_value1 = self._critic([state_batch, actor_actions])
            actor_loss = -1 * tf.math.reduce_mean(critic_value1)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self._update_target()
        action_quality = tf.math.reduce_mean(critic_value1) - tf.math.reduce_mean(critic_value)
        return critic_loss, tf.math.reduce_mean(reward_batch[0]), tf.math.reduce_mean(critic_value), action_quality
        # return critic_loss, tf.math.reduce_mean(reward_batch[0]), tf.math.reduce_mean(critic_value), 0

    def _update_target(self):
        # update target critic network
        new_weights = []
        target_critic_weights = self._target_critic.weights
        for i, critic_weight in enumerate(self._critic.weights):
            new_weights.append(self._tau * critic_weight + (1 - self._tau) * target_critic_weights[i])
        self._target_critic.set_weights(new_weights)

        # update target actor network
        new_weights = []
        target_actor_weights = self._target_actor.weights
        for i, actor_weight in enumerate(self.actor.weights):
            new_weights.append(self._tau * actor_weight + (1 - self._tau) * target_actor_weights[i])
        self._target_actor.set_weights(new_weights)

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

        #--------------------- mini hidden layers ----------------------
        fire_distance_bus, fire_distance_branch = SliceFireDistanceLayer(24)(fire_distance_input)

        # --------- bus --------
        st_bus_fire_distance = layers.Concatenate() ([bus_input, fire_distance_bus])
        st_bus_fire_distance_mix = MixFeaturesLayer(2, 24)(st_bus_fire_distance)
        st_bus_sliced_input = SliceLayer(2, 24)(st_bus_fire_distance_mix)

        st_bus_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_bus_mini_dense1_output = [st_bus_weight_sharing_dense1 (st_bus_sliced_input[i]) for i in range(len(st_bus_sliced_input))]
        st_bus_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_bus_mini_dense2_output = [st_bus_weight_sharing_dense2 (st_bus_mini_dense1_output[i]) for i in range(len(st_bus_mini_dense1_output))]

        st_bus_mini_hidden_concat = layers.Concatenate() (st_bus_mini_dense2_output)

        # --------- branch --------
        st_branch_combine = layers.Concatenate() ([branch_input, fire_distance_branch, line_flow_input])     # for branch
        st_branch_mix = MixFeaturesLayer(3, 34)(st_branch_combine)
        st_branch_sliced_input = SliceLayer(3, 34)(st_branch_mix)

        st_branch_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_branch_mini_hidden1_output = [st_branch_weight_sharing_dense1 (st_branch_sliced_input[i]) for i in range(len(st_branch_sliced_input))]
        st_branch_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_branch_mini_hidden2_output = [st_branch_weight_sharing_dense2 (st_branch_mini_hidden1_output[i]) for i in range(len(st_branch_mini_hidden1_output))]

        st_branch_mini_hidden_concat = layers.Concatenate() (st_branch_mini_hidden2_output)

        # --------- generator --------
        st_load_demand_gen_only = SelectGeneratorsLayer() (load_demand_input)         # for generators
        st_generator_output_gen_only = SelectGeneratorsLayer()(gen_inj_input)

        st_gen_combine = layers.Concatenate() ([st_load_demand_gen_only, st_generator_output_gen_only])
        st_gen_mix_feature = MixFeaturesLayer(2, 11)(st_gen_combine)
        st_gen_sliced_input = SliceLayer(2, 11)(st_gen_mix_feature)

        st_gen_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_gen_mini_hidden1_output = [st_gen_weight_sharing_dense1 (st_gen_sliced_input[i]) for i in range(len(st_gen_sliced_input))]
        st_gen_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_gen_mini_hidden2_output = [st_gen_weight_sharing_dense2 (st_gen_mini_hidden1_output[i]) for i in range(len(st_gen_mini_hidden1_output))]

        st_gen_mini_hidden_concat = layers.Concatenate() (st_gen_mini_hidden2_output)

        # -------------------------------------------------------------------------------

        comb_merged_layer = layers.Concatenate() ([st_bus_mini_hidden_concat, st_branch_mini_hidden_concat, st_gen_mini_hidden_concat])
        comb_hidden_layer = layers.Dense(512, "relu") (comb_merged_layer)
        comb_hidden_layer = layers.Dense(512, "relu") (comb_hidden_layer)

        gen_inj_output = layers.Dense(self._action_spaces[3], activation="sigmoid") (comb_hidden_layer)
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

        # # num_branch -> MultiBinary(34)
        act_branch = layers.Input(shape=(self._action_spaces[1],))
        # act_branch1 = layers.Dense(30, activation="relu") (act_branch)

        # generator_injection -> Box(5, )
        act_gen_injection = layers.Input(shape=(self._action_spaces[3],))
        # act_gen_injection1 = layers.Dense(32, activation="relu") (act_gen_injection)          # power ramping up/down

        #--------------------- mini hidden layers ----------------------
        fire_distance_bus, fire_distance_branch = SliceFireDistanceLayer(24)(st_fire_distance)

        # ---------- bus ---------
        st_bus_fire_distance = layers.Concatenate() ([st_bus, fire_distance_bus])       # preprocessing
        st_bus_fire_distance_mix = MixFeaturesLayer(2, 24)(st_bus_fire_distance)
        st_bus_sliced_input = SliceLayer(2, 24)(st_bus_fire_distance_mix)

        st_bus_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_bus_mini_dense1_output = [st_bus_weight_sharing_dense1 (st_bus_sliced_input[i]) for i in range(len(st_bus_sliced_input))]
        st_bus_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_bus_mini_dense2_output = [st_bus_weight_sharing_dense2 (st_bus_mini_dense1_output[i]) for i in range(len(st_bus_mini_dense1_output))]

        st_bus_mini_hidden_concat = layers.Concatenate() (st_bus_mini_dense2_output)

        # ---------- branch ---------
        st_branch_combine = layers.Concatenate() ([st_branch, fire_distance_branch, st_line_flow])     # preprocessing
        st_branch_mix = MixFeaturesLayer(3, 34)(st_branch_combine)
        st_branch_sliced_input = SliceLayer(3, 34)(st_branch_mix)

        st_branch_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_branch_mini_hidden1_output = [st_branch_weight_sharing_dense1 (st_branch_sliced_input[i]) for i in range(len(st_branch_sliced_input))]
        st_branch_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_branch_mini_hidden2_output = [st_branch_weight_sharing_dense2 (st_branch_mini_hidden1_output[i]) for i in range(len(st_branch_mini_hidden1_output))]

        st_branch_mini_hidden_concat = layers.Concatenate() (st_branch_mini_hidden2_output)

        # ---------- generators ---------
        st_load_demand_gen_only = SelectGeneratorsLayer() (st_load_demand)      # for 11 generators
        st_generator_output_gen_only = SelectGeneratorsLayer()(st_gen_output)

        st_gen_combine = layers.Concatenate() ([st_load_demand_gen_only, st_generator_output_gen_only, act_gen_injection])
        st_gen_mix_feature = MixFeaturesLayer(3, 11)(st_gen_combine)
        st_gen_sliced_input = SliceLayer(3, 11)(st_gen_mix_feature)

        st_gen_weight_sharing_dense1 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_gen_mini_hidden1_output = [st_gen_weight_sharing_dense1 (st_gen_sliced_input[i]) for i in range(len(st_gen_sliced_input))]
        st_gen_weight_sharing_dense2 = layers.Dense(self.mini_hidden_layer_size, activation="relu")
        st_gen_mini_hidden2_output = [st_gen_weight_sharing_dense2 (st_gen_mini_hidden1_output[i]) for i in range(len(st_gen_mini_hidden1_output))]

        st_gen_mini_hidden_concat = layers.Concatenate() (st_gen_mini_hidden2_output)

        # -------------------------------------------------------------------------------

        comb_merged_layer = layers.Concatenate() ([st_bus_mini_hidden_concat, st_branch_mini_hidden_concat, st_gen_mini_hidden_concat])
        comb_hidden_layer = layers.Dense(512, "relu") (comb_merged_layer)
        comb_hidden_layer = layers.Dense(512, "relu") (comb_hidden_layer)

        reward = layers.Dense(1, activation="linear") (comb_hidden_layer)
        model = tf.keras.Model([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta, st_line_flow,
                                act_gen_injection], reward)
        return model
