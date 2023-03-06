import os
import copy
import json
import numpy as np
import pandas as pd
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

        # self.actor = self._actor_model()
        # self._target_actor = self._actor_model()
        # self._target_actor.set_weights(self.actor.get_weights())
        #
        # self._critic = self._critic_model()
        # self._target_critic = self._critic_model()
        # self._target_critic.set_weights(self._critic.get_weights())

        self.actor = GNN(True, self._save_weight_directory)
        self._target_actor = GNN(True, self._save_weight_directory)

        self._critic = GNN(False, self._save_weight_directory)
        self._target_critic = GNN(False, self._save_weight_directory)
        self.is_set_weight = 0

    def set_weights(self):
        self._target_actor.set_weights(self.actor)
        self._target_critic.set_weights(self._critic)
        return True

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
        self.actor.save_weight("actor", version, episode_num)
        self._target_actor.save_weight("target_actor", version, episode_num)
        self._critic.save_weight("critic", version, episode_num)
        self._target_critic.save_weight("target_critic", version, episode_num)

    def load_weight(self, version, episode_num):
        self.actor.load_weights(f"{self._load_weight_directory}/agent_actor{version}_{episode_num}.h5")
        self._target_actor.load_weights(f"{self._load_weight_directory}/agent_target_actor{version}_{episode_num}.h5")
        self._critic.load_weights(f"{self._load_weight_directory}/agent_critic{version}_{episode_num}.h5")
        self._target_critic.load_weights(f"{self._load_weight_directory}/agent_target_critic{version}_{episode_num}.h5")
        print("weights are loaded successfully!")

    # @tf.function
    def train(self, state_batch, action_batch, reward_batch, next_state_batch, episode_end_flag_batch):
        # update critic network
        self.is_set_weight += self.set_weights() if self.is_set_weight == 1 else 1

        with tf.GradientTape() as tape:
            target_actor_actions = self._target_actor(next_state_batch)
            target_critic_state = tf.concat([next_state_batch[0], target_actor_actions], axis=-1)
            target_critic_values = self._target_critic((target_critic_state, next_state_batch[1])) #* (1 - episode_end_flag_batch)
            return_y = reward_batch + self._gamma * target_critic_values

            critic_state = tf.concat([state_batch[0], action_batch], axis=-1)
            critic_value_with_original_actions = self._critic((critic_state, state_batch[1]))
            critic_loss = tf.math.reduce_mean(tf.math.square(return_y - critic_value_with_original_actions))

        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        # update actor network
        with tf.GradientTape() as tape:
            actor_actions = self.actor(state_batch)
            critic_state = tf.concat([state_batch[0], actor_actions], axis=-1)
            critic_value_with_actor_actions = self._critic((critic_state, state_batch[1]))
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
        gen_inj_output = layers.Dense(self._num_of_active_generators, activation="relu") (hidden)

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


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return tf.keras.Sequential(fnn_layers, name=name)

class GraphConvLayer(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, normalize, save_weight_dir, *args, **kwargs,):

        super().__init__(*args, **kwargs)
        self.aggregation_type = "sum"
        self.combination_type = "concat"
        self.normalize = normalize
        self._save_weight_directory = save_weight_dir

        self.prepare_neighbor_messages_ffn = create_ffn(hidden_units, dropout_rate)
        # if self.combination_type == "gated":
        #     self.node_embedding_fn = layers.GRU(units=hidden_units, activation="tanh", recurrent_activation="sigmoid",
        #                                         dropout=dropout_rate, return_state=True, recurrent_dropout=dropout_rate, )
        # else:
        self.node_embedding_fn = create_ffn(hidden_units, dropout_rate)

    def prepare_neighbor_messages(self, node_repesentations, branch_weights):
        messages = self.prepare_neighbor_messages_ffn(node_repesentations)        # node_repesentations shape is [num_edges, embedding_dim].
        messages = messages * branch_weights
        return messages

    def aggregate_neighbor_messages(self, node_indices, neighbour_messages, num_nodes):
        # node_indices shape is [num_edges], neighbour_messages shape: [num_edges, representation_dim].
        # if self.aggregation_type == "sum":
        agg = []
        for neigh in neighbour_messages:
            agg.append(tf.math.unsorted_segment_sum(neigh, node_indices, num_segments=num_nodes))
        aggregated_message = tf.stack(agg)
        # aggregated_message = tf.stack(tf.math.unsorted_segment_sum(neigh_message, node_indices, num_segments=num_nodes) for neigh_message in tf.unstack(neighbour_messages))
        # elif self.aggregation_type == "mean":
        #     aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        # elif self.aggregation_type == "max":
        #     aggregated_message = tf.math.unsorted_segment_max(neighbour_messages, node_indices, num_segments=num_nodes)
        # else:
        #     raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
        return aggregated_message

    def create_node_embedding(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim], aggregated_messages shape is [num_nodes, representation_dim].
        # if self.combination_type == "gru":
        #     h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        # elif self.combination_type == "concat":
        h = tf.concat([node_repesentations, aggregated_messages], axis=2)
        # elif self.combination_type == "add":
        #     h = node_repesentations + aggregated_messages
        # else:
        #     raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.node_embedding_fn(h)
        # if self.combination_type == "gru":
        #     node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        node_repesentations, branches, branch_weights = inputs
        node_indices, neighbour_indices = branches[0], branches[1]
        num_nodes = node_repesentations.shape[1]                # node_repesentations shape is [num_nodes, representation_dim]

        # print("node_indices:", node_indices.shape)
        # print("neighbour_indices:", neighbour_indices.shape)
        # print("node_representation:", node_repesentations.shape)
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices, axis=1, batch_dims=-1)     # neighbour_repesentations shape is [num_edges, representation_dim].
        # print("neighbour_representations:", neighbour_repesentations.shape)
        neighbour_messages = self.prepare_neighbor_messages(neighbour_repesentations, branch_weights)
        # print("neighbour_messages:", neighbour_messages.shape)
        aggregated_messages = self.aggregate_neighbor_messages(node_indices, neighbour_messages, num_nodes)
        # print("aggregated_messages:", aggregated_messages.shape)
        return self.create_node_embedding(node_repesentations, aggregated_messages)        # Update the node embedding with the neighbour messages; # Returns: node_embeddings of shape [num_nodes, representation_dim].

    def set_weights(self, conv_obj):
        self.prepare_neighbor_messages_ffn.set_weights(conv_obj.prepare_neighbor_messages_ffn.get_weights())
        self.node_embedding_fn.set_weights(conv_obj.node_embedding_fn.get_weights())

    def save_weight(self, obj_name, version, episode_num):
        self.prepare_neighbor_messages_ffn.save_weights(f"{self._save_weight_directory}/{obj_name}_conv_neighbor_messages-{version}_{episode_num}.h5")
        self.node_embedding_fn.save_weights(f"{self._save_weight_directory}/{obj_name}_conv_node_embedding-{version}_{episode_num}.h5")

class GNN(tf.keras.Model):
    def __init__(self, is_actor, save_weight_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_actor = is_actor
        hidden_units = [32, 32]
        dropout_rate = 0.2
        normalize = True
        num_classes = 10
        self._save_weight_directory = save_weight_dir

        branch_path = "configurations/branches.csv"
        branches = pd.read_csv(branch_path, header=None, names=["node_a", "node_b"])
        self.branches = branches[["node_a", "node_b"]].to_numpy().T
        # print(self.branches)

        self.node_feature_processing_ffn = create_ffn(hidden_units, dropout_rate, name="preprocess")
        self.conv1 = GraphConvLayer(hidden_units, dropout_rate, normalize, save_weight_dir, name="graph_conv1",)
        self.conv2 = GraphConvLayer(hidden_units, dropout_rate, normalize, save_weight_dir, name="graph_conv2",)
        self.node_embedding_processing_ffn = create_ffn(hidden_units, dropout_rate, name="postprocess")
        if is_actor:
            self.compute_output = layers.Dense(units=1, activation="relu", name="logits")    # logits layer for actor
        else:
            self.compute_output = layers.Dense(units=1, activation="linear", name="logits")    # output value layer for critic
            self.compute_critic_value = layers.Dense(units=1, activation="linear", name="logits")    # output value layer for critic

    def call(self, graph_info):
        node_info, branch_info = graph_info
        # print("node_info_shape:", node_info.shape, ", branch_info_shape:", branch_info.shape)
        x = self.node_feature_processing_ffn(node_info)      # process the node_features to produce node representations.
        # print("x_shape:", x.shape)
        x1 = self.conv1((x, self.branches, branch_info))
        # print("x1_shape:", x1.shape)
        x = x1 + x                                                    # Skip connection.
        x2 = self.conv2((x, self.branches, branch_info))
        # print("x2_shape:", x2.shape)
        x = x2 + x                                                    # Skip connection.
        x = self.node_embedding_processing_ffn(x)
        # print("x_shape1:", x.shape)
        output = self.compute_output(x)                       # Compute logits-actor, value-critic
        # print("Ouput_shape:", output.shape)

        if not self.is_actor:
            output = tf.squeeze(output, axis=-1)
            output = self.compute_critic_value(output)
            # print("critic_output_shape:", output.shape)
        return output

    def set_weights(self, gnn_obj):
        self.node_feature_processing_ffn.set_weights(gnn_obj.node_feature_processing_ffn.get_weights())
        self.conv1.set_weights(gnn_obj.conv1)
        self.conv2.set_weights(gnn_obj.conv2)
        self.node_embedding_processing_ffn.set_weights(gnn_obj.node_embedding_processing_ffn.get_weights())
        self.compute_output.set_weights(gnn_obj.compute_output.get_weights())
        if not self.is_actor:
            self.compute_critic_value.set_weights(gnn_obj.compute_critic_value.get_weights())

    def save_weight(self, obj_name, version, episode_num):
        self.node_feature_processing_ffn.save_weights(f"{self._save_weight_directory}/{obj_name}_node_feature-{version}_{episode_num}.h5")
        self.conv1.save_weight(obj_name, version, episode_num)
        self.conv2.save_weight(obj_name, version, episode_num)
        self.node_embedding_processing_ffn.save_weights(f"{self._save_weight_directory}/{obj_name}_node_embedding-{version}_{episode_num}.h5")
        self.compute_output.save_weights(f"{self._save_weight_directory}/{obj_name}_compute_output-{version}_{episode_num}.h5")
        if not self.is_actor:
            self.compute_critic_value.save_weights(f"{self._save_weight_directory}/{obj_name}_critic_value-{version}_{episode_num}.h5")
