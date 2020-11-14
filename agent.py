import copy
import tensorflow as tf
from tensorflow.keras import layers


class Agent:
    def __init__(self, state_spaces, action_spaces):
        self._gamma = 0.9      # discount factor
        self._tau = 0.05       # used to update target network
        actor_lr = 0.001
        critic_lr = 0.002
        self._actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self._critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self._state_spaces =  copy.deepcopy(state_spaces)
        self._action_spaces =  copy.deepcopy(action_spaces)

        self._actor = self._actor_model()
        self._target_actor = self._actor_model()
        self._target_actor.set_weights(self._actor.get_weights())

        self._critic = self._critic_model()
        self._target_critic = self._critic_model()
        self._target_critic.set_weights(self._critic.get_weights())

    def save_weight(self, version, episode_num):
        self._actor.save_weights(f"saved_model/agent_actor{version}_{episode_num}.h5")
        self._critic.save_weights(f"saved_model/agent_critic{version}_{episode_num}.h5")
        self._target_actor.save_weights(f"saved_model/agent_target_actor{version}_{episode_num}.h5")
        self._target_critic.save_weights(f"saved_model/agent_target_critic{version}_{episode_num}.h5")

    def load_weight(self, version, episode_num):
        self._actor.load_weights(f"saved_model/agent_actor{version}_{episode_num}.h5")
        self._target_actor.load_weights(f"saved_model/agent_target_actor{version}_{episode_num}.h5")
        self._critic.load_weights(f"saved_model/agent_critic{version}_{episode_num}.h5")
        self._target_critic.load_weights(f"saved_model/agent_target_critic{version}_{episode_num}.h5")
        print("weights are loaded successfully!")

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        action_batch1 = [action_batch[0], action_batch[1]]
        # update critic network
        with tf.GradientTape() as tape:
            target_actions = self._target_actor(next_state_batch)
            action_batch1.append(target_actions)
            y = reward_batch + self._gamma * self._target_critic([next_state_batch, action_batch1])
            critic_value = self._critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(critic_grad, self._critic.trainable_variables))

        action_batch1.pop()
        # update actor network
        with tf.GradientTape() as tape:
            actions = self._actor(state_batch)
            action_batch1.append(actions)
            critic_value1 = self._critic([state_batch, action_batch1])
            actor_loss = -1 * tf.math.reduce_mean(critic_value1)
        actor_grad = tape.gradient(actor_loss, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(zip(actor_grad, self._actor.trainable_variables))

        self._update_target()
        return critic_loss, tf.math.reduce_mean(reward_batch), tf.math.reduce_mean(critic_value)

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
        for i, actor_weight in enumerate(self._actor.weights):
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


        state = layers.Concatenate() ([bus_input, branch_input, fire_distance_input, gen_inj_input, load_demand_input, theta_input, line_flow_input])
        # state = layers.Concatenate() ([bus_input1, branch_input1, fire_distance_input1, gen_inj_input1, load_demand_input1, theta_input1])
        hidden = layers.Dense(512, activation="tanh") (state)
        hidden = layers.Dense(128, activation="tanh") (hidden)

        # bus -> MultiBinary(24)
        # bus_output = layers.Dense(action_space[0], activation="sigmoid") (hidden)
        #
        # # num_branch -> MultiBinary(34)
        # branch_output = layers.Dense(action_space[1], activation="sigmoid") (hidden)

        # generator_injection (generator output) -> Box(5, )
        gen_inj_output = layers.Dense(self._action_spaces[3], activation="sigmoid") (hidden)

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

        # state = layers.Concatenate() ([st_bus, act_gen_injection])
        state = layers.Concatenate() ([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta, st_line_flow,
                                       act_bus, act_branch, act_gen_injection])
        # state = layers.Concatenate() ([st_bus1, st_branch1, st_fire_distance1, st_gen_output1, st_load_demand1, st_theta1,
        #                                act_bus1, act_branch1, act_gen_injection1])

        hidden = layers.Dense(512, activation="relu") (state)
        hidden = layers.Dense(128, activation="relu") (hidden)
        reward = layers.Dense(1, activation="linear") (hidden)

        model = tf.keras.Model([st_bus, st_branch, st_fire_distance, st_gen_output, st_load_demand, st_theta, st_line_flow,
                                act_bus, act_branch, act_gen_injection], reward)
        return model

