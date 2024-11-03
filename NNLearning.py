import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

class NNLearning:
    def __init__(self, gamma: float, epsilon: float, batch_size: int, max_steps: int,
                 seats_available: int, price_levels: [], loss_function, optimizer):
        self.seed = 42
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.action_size = len(price_levels)
        self.price_levels = price_levels
        self.epsilon_interval = self.epsilon_max - self.epsilon_min
        self.batch_size = batch_size
        self.max_steps_per_episode = max_steps
        self.seats_available = seats_available
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = self.create_q_model()
        self.model.compile(optimizer=optimizer,
                      loss=loss_function, run_eagerly=True)
        self.model_target = self.create_q_model()
        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.state_next_history = deque(maxlen=100)
        self.rewards_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.history = deque(maxlen=100)
        self.frame_count = 0
        self.batch_size = 32
        self.update_after_actions = 4
        self.update_target_network = 100

    def create_q_model(self):
        inputs = layers.Input(shape=(self.seats_available))
        # Start with very simple network, one dense layer
        layer1 = layers.Dense(128, activation="tanh")(inputs)
        layer2 = layers.Dense(256, activation="tanh")(layer1)
        layer3 = layers.Dense(128, activation="tanh")(layer2)
        action = layers.Dense(self.action_size, activation="softplus")(layer3)
        return keras.Model(inputs=inputs, outputs=action)

    def take_step(self, current_state: [], training: bool):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(self.action_size)
            return choice, self.price_levels[choice]
        else:
            state_tensor = tf.convert_to_tensor(current_state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probabilities = self.model(state_tensor, training=training)
            best_action = tf.argmax(action_probabilities[0]).numpy()
            return best_action, self.price_levels[best_action]

    def get_action(self, state: []):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(self.action_size)
            prices = np.full(self.seats_available, self.price_levels[choice], dtype=float)
            return prices, choice
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probabilities = self.model(state_tensor, training=False)
            best_action = tf.argmax(action_probabilities[0]).numpy()
            prices = np.full(self.seats_available, self.price_levels[best_action], dtype=float)
            return prices, best_action

    def get_target_prediction(self, state: []):
        return self.model_target.predict(state)

    def get_updated_q_values(self, state: [], rewards_sample):
        future_rewards = self.model_target.predict(state)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
        # updated_q_values = rewards_sample + self.gamma * future_rewards
        return updated_q_values

    def train_model(self, state_sample, state_next_sample, rewards_sample, action_sample, nr_actions):
        updated_q_values = self.get_updated_q_values(state_next_sample, rewards_sample)

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        # update the target network with new weights
        self.model_target.set_weights(self.model.get_weights())

    def name(self):
        return "Deep Q-Learning"

    def process_data(self, action, start_state, prediction, round_revenue, new_state):
        self.action_history.append(prediction)
        self.state_history.append(start_state)
        self.prediction_history.append(prediction)
        self.rewards_history.append(round_revenue)
        self.state_next_history.append(new_state)
        self.frame_count += 1
        if self.frame_count % self.update_after_actions == 0 and len(self.action_history) > self.batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self.action_history)), size=self.batch_size)

            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            prediction_sample = [self.prediction_history[i] for i in indices]

            self.train_model(state_sample=state_sample, state_next_sample=state_next_sample,
                              rewards_sample=rewards_sample, action_sample=action_sample, nr_actions=self.action_size)
        if self.frame_count % self.update_target_network == 0:
            self.update_target_model()

