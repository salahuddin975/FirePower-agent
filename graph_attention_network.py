import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


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

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)     # node_states_transformed: (2708, 100); node_states: (2708, 800); kernel: (800, 100)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)  # node_states_expanded: (5429, 2, 100)
        node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))  # node_states_expanded: (5429, 200)
        attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))   # attention_scores: (5429, 1); kernel_attention: (200, 1)
        attention_scores = tf.squeeze(attention_scores, -1)    # attention_scores: (5429,)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))  # attention_scores: (5429,)
        attention_scores_sum = tf.math.unsorted_segment_sum(data=attention_scores, segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,)   # edges[:,0].shape: (5429,)
        attention_scores_sum = tf.repeat(attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32")))
        attention_scores_norm = attention_scores / attention_scores_sum    # attention_scores_norm: (5429,)

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])   # node_states_neighbors: (5429, 100)
        out = tf.math.unsorted_segment_sum(data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0], num_segments=tf.shape(node_states)[0],)     # out: (2708, 100)
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs
        outputs = [attention_layer([atom_features, pair_indices]) for attention_layer in self.attention_layers]          # Obtain outputs from each attention head
        if self.merge_type == "concat":             # Concatenate or average the node states from each head
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        return tf.nn.relu(outputs)        # Activate and return node states


class GNN_gat(keras.Model):
    def __init__(self, generators, is_critic, **kwargs,):
        super().__init__(**kwargs)
        hidden_units = 100
        num_heads = 8
        num_layers = 3
        output_dim = generators.get_num_generators() # len(class_values)

        NUM_EPOCHS = 100
        BATCH_SIZE = 256
        VALIDATION_SPLIT = 0.1
        LEARNING_RATE = 3e-1
        MOMENTUM = 0.9

        branch_path = "configurations/branches.csv"
        branches = pd.read_csv(branch_path, header=None, names=["node_a", "node_b"])
        self.branches = branches[["node_a", "node_b"]].to_numpy().T


        # self.node_states = node_states
        # self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

