import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class PaddedOutput(layers.Layer):
    def __init__(self):
        super(PaddedOutput, self).__init__()

    def call(self, inputs):
        zeros = tf.zeros(shape=(inputs.shape[0], 1), dtype=tf.float32)
        inputs = tf.concat([inputs, zeros], axis=-1)
        inputs_padded = tf.gather(inputs, [0, 1, 10, 10, 10, 10, 2, 10, 10, 10, 10, 10, 3, 10, 4, 5, 10, 6, 10, 10, 7, 8, 9, 10], axis=-1)
        return tf.reshape(inputs_padded, shape=(inputs.shape[0], 24, 1))


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    for units in hidden_units:
        # fnn_layers.append(layers.BatchNormalization())
        # fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    return tf.keras.Sequential(fnn_layers, name=name)

class GraphConvLayer(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, normalize, name, *args, **kwargs,):

        super().__init__(*args, **kwargs)
        self.aggregation_type = "sum"
        self.combination_type = "concat"
        self.normalize = normalize
        self.layer_name = name

        self.prepare_neighbor_messages_ffn = create_ffn(hidden_units, dropout_rate)
        # if self.combination_type == "gated":
        #     self.node_embedding_fn = layers.GRU(units=hidden_units, activation="tanh", recurrent_activation="sigmoid",
        #                                         dropout=dropout_rate, return_state=True, recurrent_dropout=dropout_rate, )
        # else:
        self.node_embedding_fn = create_ffn(hidden_units, dropout_rate)

    def prepare_neighbor_messages(self, neighbour_repesentations, branch_weights):
        messages = self.prepare_neighbor_messages_ffn(neighbour_repesentations)        # node_repesentations shape is [batch_size, num_edges, embedding_dim].
        messages = messages * branch_weights
        return messages

    def aggregate_neighbor_messages(self, node_indices, neighbour_messages, num_nodes):
        # node_indices shape is [num_edges], neighbour_messages shape: [num_edges, representation_dim].
        # if self.aggregation_type == "sum":
        stk = []
        for i in range(neighbour_messages.shape[0]):      # iterate over batch
            stk.append(tf.math.unsorted_segment_sum(neighbour_messages[i], node_indices, num_segments=num_nodes))
        aggregated_message = tf.stack(stk)
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
        # print("node_representation.shape:", node_repesentations.shape)
        # print("aggregated_messages.shape:", aggregated_messages.shape)
        stk = []
        for i in range(node_repesentations.shape[0]):
            stk.append(tf.concat([node_repesentations[i], aggregated_messages[i]], axis=1))
        h = tf.stack(stk)
        # print("h.shape:", h.shape)
        # elif self.combination_type == "add":
        #     h = node_repesentations + aggregated_messages
        # else:
        #     raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.node_embedding_fn(h)
        # print("node_embeddings:", node_embeddings.shape)
        # print(self.layer_name, ": node_embedding_fn weights:", self.node_embedding_fn.get_weights())
        # if self.combination_type == "gru":
        #     node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings)
        return node_embeddings

    # @tf.function
    def call(self, inputs):
        node_repesentations, branches, branch_weights = inputs
        node_indices, neighbour_indices = branches[0], branches[1]
        num_nodes = node_repesentations.shape[1]                # node_repesentations shape is [batch_size, num_nodes, representation_dim]

        # print("node_indices:", node_indices.shape)
        # print("neighbour_indices:", neighbour_indices.shape)
        # print("node_representation:", node_repesentations.shape)
        stk = []
        for i in range(node_repesentations.shape[0]):
            stk.append(tf.gather(node_repesentations[i], neighbour_indices))     # neighbour_repesentations shape is [batch_size, num_edges, representation_dim].
        neighbour_repesentations = tf.stack(stk)
        # print("neighbour_representations:", neighbour_repesentations.shape)
        neighbour_messages = self.prepare_neighbor_messages(neighbour_repesentations, branch_weights)
        # print("neighbour_messages:", neighbour_messages.shape)
        aggregated_messages = self.aggregate_neighbor_messages(node_indices, neighbour_messages, num_nodes)
        # print("aggregated_messages:", aggregated_messages.shape)
        return self.create_node_embedding(node_repesentations, aggregated_messages)        # Update the node embedding with the neighbour messages; # Returns: node_embeddings of shape [num_nodes, representation_dim].

class GNN_conv(tf.keras.Model):
    def __init__(self, generators, is_critic, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_critic = is_critic
        hidden_units = [8, 8]
        dropout_rate = 0.2
        normalize = True
        self.num_conv_layer = 3

        branch_path = "configurations/branches.csv"
        branches = pd.read_csv(branch_path, header=None, names=["node_a", "node_b"])
        self.branches = branches[["node_a", "node_b"]].to_numpy().T
        # print(self.branches)

        self.node_feature_processing_ffn = create_ffn(hidden_units, dropout_rate, name="preprocess")

        self.conv_layers = []
        for i in range(self.num_conv_layer):
            self.conv_layers.append(GraphConvLayer(hidden_units, dropout_rate, normalize, name="graph_conv" + str(i),))

        self.node_embedding_processing_ffn = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.compute_logits = layers.Dense(units=1, name="logits")    # logits layer for actor
        if self.is_critic:
            self.compute_critic_value = layers.Dense(units=1, activation="linear", name="critic_output")    # output value layer for critic
        else:
            self.compute_actor_values = layers.Dense(units=generators.get_num_generators(), activation="softmax", name="actor_output")
            self.padded_output = PaddedOutput()

    # @tf.function
    def call(self, graph_info):
        node_info, branch_info = graph_info
        # print("node_info_shape:", node_info.shape, ", branch_info_shape:", branch_info.shape)
        # print("begin: node_info:", node_info)
        # print("begin: branch_info:", branch_info)
        x = self.node_feature_processing_ffn(node_info)      # process the node_features to produce node representations.
        # print("begin: node_feature_processing_fnn weights:", self.node_feature_processing_ffn.get_weights())
        # print("begin: preprocessed: ", x.shape)

        for i in range(self.num_conv_layer):
            x_out = self.conv_layers[i]((x, self.branches, branch_info))
            x = x_out + x

        x = self.node_embedding_processing_ffn(x)
        # print("end: node_embedding_processing weights:", self.node_embedding_processing_ffn.get_weights())
        # print("end: x: ", x)

        # print("x_shape1:", x.shape)
        # print("x:", x)
        logits = self.compute_logits(x)                       # Compute logits-actor, value-critic
        # print("Ouput_shape:", logits.shape)
        logits = tf.squeeze(logits, axis=-1)
        # print("squeezed_output_shape:", logits.shape)
        if self.is_critic:
            output = self.compute_critic_value(logits)
            # print("critic_output_shape:", output.shape)
            # print("critic output:", output)
        else:
            output = self.compute_actor_values(logits)
            # print("output:", output.shape)
            output = self.padded_output(output)
            # print("output1:", output.shape)

        return output

