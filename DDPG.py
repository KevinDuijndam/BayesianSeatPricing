import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=512, num_states=10, num_actions=10):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1


class DDPG:
    def __init__(self, gamma: float, batch_size: int, seats_available: int, price_levels: [], nr_flight_types: int):
        self.seed = 42
        # Pretty good values
        # self.tau = 0.01
        # critic_lr = 4e-3
        # actor_lr = 1e-3
        # or critic_lr = 2e-3, actor_lr = 5e-4
        # std_dev = 2

        self.offer_count = 0

        self.tau = 0.01
        critic_lr = 2e-3
        actor_lr = 5e-4
        std_dev = 2

        self.gamma = gamma
        self.seats_available = seats_available
        self.nr_flight_types = nr_flight_types
        self.num_states = seats_available + nr_flight_types
        self.num_actions = 1
        self.lower_bound = min(price_levels)
        self.upper_bound = max(price_levels)
        self.name = "DDPG"

        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=actor_lr)
        self.buffer = Buffer(10000, batch_size, self.num_states, self.num_actions)

    def name(self):
        return "DDPG"

    def initialise_for_flight_type(self, flight_type):
        return

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-3e-1, maxval=3e-1)

        inputs = layers.Input(shape=(self.num_states,))
        #out = layers.Dense(128, activation="relu")(inputs)
        #out = layers.Dense(128, activation="relu")(out)
        # Initial OK actor size 32 x 32
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        # Initial OK critic 16 x 32
        state_out = layers.Dense(32, activation="relu")(state_input)
        state_out = layers.Dense(64, activation="relu")(state_out)

        # Action as input
        # Initial OK critic 32
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(64, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        #out = layers.Dense(256, activation="relu")(concat)
        #out = layers.Dense(256, activation="relu")(out)
        # Initial OK critic 64 x 32
        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(self.num_actions)(out)
        outputs = outputs * self.upper_bound * self.num_states

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_action(self, state: [], flight_type):
        self.offer_count += 1
        if self.offer_count < 100:
            test_price = np.random.randint(self.lower_bound, self.upper_bound)
            price = np.full((self.seats_available), test_price)
            return price, 1

        flight_type_state = np.zeros(self.nr_flight_types)
        flight_type_state[flight_type] = 1
        state_to_use = np.append(state, flight_type_state)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state_to_use), 0)
        sampled_actions = tf.squeeze(self.actor_model(tf_state))
        noise = self.noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        best_price = [np.squeeze(legal_action)][0]
        price = np.full((self.seats_available), best_price)
        return price, 1

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered, flight_type):
        flight_type_state = np.zeros(self.nr_flight_types)
        flight_type_state[flight_type] = 1
        state = np.append(start_state, flight_type_state)
        next_state = np.append(new_state, flight_type_state)
        self.buffer.record((np.copy(state), np.copy(action[0]), np.copy(round_revenue), np.copy(next_state)))
        self.learn()
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def update_during_booking(self, customers_offered, total_customers,
                              action, start_state, prediction, revenue, new_state, flight_type):
        flight_type_state = np.zeros(self.nr_flight_types)
        flight_type_state[flight_type] = 1
        state = np.append(start_state, flight_type_state)
        next_state = np.append(new_state, flight_type_state)
        self.buffer.record((np.copy(state), np.copy(action[0]), np.copy(revenue), np.copy(next_state)))
        self.learn()
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

        updated_price, _ = self.get_action(new_state, flight_type)
        return updated_price, True

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed-up for blocks of code that contain many small TensorFlow operations such as this one.
#    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        #print("Critic loss: ", critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
