import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class PaddedOutput(layers.Layer):
    def __init__(self):
        super(PaddedOutput, self).__init__()

    def call(self, inputs):
        zeros = tf.zeros(shape=(inputs.shape[0], 1), dtype=tf.float32)
        inputs = tf.concat([inputs, zeros], axis=-1)
        inputs_padded = tf.gather(inputs, [0, 1, 10, 10, 10, 10, 2, 10, 10, 10, 10, 10, 3, 10, 4, 5, 10, 6, 10, 10, 7, 8, 9, 10], axis=-1)
        return tf.reshape(inputs_padded, shape=(inputs.shape[0], 24, 1))


class GraphAttention(layers.Layer):
    def __init__(self, units, kernel_initializer="glorot_uniform", kernel_regularizer=None, **kwargs,):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units), trainable=True,
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, name="kernel",)
        self.kernel_attention = self.add_weight(shape=(self.units * 2, 1), trainable=True,
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, name="kernel_attention",)
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs     # node_states: (2708, 800); edges: (5429, 2)
        print("ga:node_states_shape:", node_states.shape, ", edges_shape:", edges.shape)
        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)     # node_states_transformed: (2708, 100); node_states: (2708, 800); kernel: (800, 100)
        print("ga:node_states_transformed_shape:", node_states_transformed.shape)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)  # node_states_expanded: (5429, 2, 100)
        print("ga:node_states_expanded_shape1:", node_states_expanded.shape)
        node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))  # node_states_expanded: (5429, 200)
        print("ga:node_states_expanded_shape12:", node_states_expanded.shape)
        attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))   # attention_scores: (5429, 1); kernel_attention: (200, 1)
        print("ga:attention_scores_shape1:", attention_scores.shape)
        attention_scores = tf.squeeze(attention_scores, -1)    # attention_scores: (5429,)
        print("ga:attention_scores_shape2:", attention_scores.shape)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))  # attention_scores: (5429,)
        print("ga:attention_scores_shape3:", attention_scores.shape)
        attention_scores_sum = tf.math.unsorted_segment_sum(data=attention_scores, segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,)   # edges[:,0].shape: (5429,)
        print("ga:attention_scores_sum_shape1:", attention_scores_sum.shape)
        attention_scores_sum = tf.repeat(attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32")))
        print("ga:attention_scores_sum_shape2:", attention_scores_sum.shape)
        attention_scores_norm = attention_scores / attention_scores_sum    # attention_scores_norm: (5429,)
        print("ga:attention_scores_norm_shape:", attention_scores_norm.shape)

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])   # node_states_neighbors: (5429, 100)
        print("ga:node_states_neighbors_shape:", node_states_neighbors.shape)
        out = tf.math.unsorted_segment_sum(data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0], num_segments=tf.shape(node_states)[0],)     # out: (2708, 100)
        print("ga:out_shape:", out.shape)
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs
        print("mh:atom_features_shape:", atom_features.shape, ", pair_indices_shape:", pair_indices.shape)
        outputs = [attention_layer([atom_features, pair_indices]) for attention_layer in self.attention_layers]          # Obtain outputs from each attention head
        # print("outputs_shape:", outputs.shape)
        if self.merge_type == "concat":             # Concatenate or average the node states from each head
            outputs = tf.concat(outputs, axis=-1)
            print("mh:concat_outputs_shape:", outputs.shape)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
            print("mh:mean_outputs_shape:", outputs.shape)
        return tf.nn.relu(outputs)        # Activate and return node states


class GNN_gat(keras.Model):
    def __init__(self, generators, is_critic, **kwargs,):
        super().__init__(**kwargs)
        hidden_units = 100
        num_heads = 8
        num_layers = 3
        # output_dim = generators.get_num_generators() # len(class_values)
        self.is_critic = is_critic

        NUM_EPOCHS = 100
        BATCH_SIZE = 256
        VALIDATION_SPLIT = 0.1
        LEARNING_RATE = 3e-1
        MOMENTUM = 0.9

        branch_path = "configurations/branches.csv"
        branches = pd.read_csv(branch_path, header=None, names=["node_a", "node_b"])
        self.branches = branches[["node_a", "node_b"]].to_numpy()

        # self.node_states = node_states
        # self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        # self.output_layer = layers.Dense(output_dim)

        self.compute_logits = layers.Dense(units=1, name="logits")    # logits layer for actor
        if self.is_critic:
            self.compute_critic_value = layers.Dense(units=1, activation="linear", name="critic_output")    # output value layer for critic
        else:
            self.compute_actor_values = layers.Dense(units=generators.get_num_generators(), activation="softmax", name="actor_output")
            self.padded_output = PaddedOutput()



    def call(self, inputs):
        node_states, edges = inputs[0], self.branches
        # node_states = tf.reshape(node_states, shape=(node_states.shape[1], node_states.shape[2]))
        print("gat:node_state_shape:", node_states.shape, ", edges_shape:", edges.shape)
        x = self.preprocess(node_states)
        print("gat:preprocessed_x_shape:", x.shape)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
            print("gat:attention_layer:", attention_layer, ", x_shape:", x.shape)
        # outputs = self.output_layer(x)
        # print("gat:outputs_shape:", outputs.shape)
        # return outputs

        logits = self.compute_logits(x)                       # Compute logits-actor, value-critic
        print("logits_shape:", logits.shape)
        logits = tf.squeeze(logits, axis=-1)
        # logits = tf.reshape(logits, shape=(1, 24))
        print("squeezed_logits_shape:", logits.shape)
        if self.is_critic:
            output = self.compute_critic_value(logits)
            # print("critic_output_shape:", output.shape)
            # print("critic output:", output)
        else:
            output = self.compute_actor_values(logits)
            print("output:", output)
            output = self.padded_output(output)
            # print("output1:", output)

        return output

