import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

class DeepPred:
    def __init__(self, epsilon: float, batch_size: int, max_steps: int,
                 seats_available: int, price_levels: []):
        self.seats_available = seats_available
        self.prices_possible = price_levels
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.prices = np.full((self.seats_available), np.random.choice(self.prices_possible))
        self.model = self.create_model()
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                           loss='binary_crossentropy')
        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.state_next_history = deque(maxlen=100)
        self.rewards_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.history = deque(maxlen=100)
        self.frame_count = 0
        self.batch_size = 32
        self.update_after_actions = 5
        self.epochs = 8
        self.create_input_training()
        self.name = "Deep Prediction"

    def create_model(self):
        inputs = layers.Input(shape=(self.seats_available))
        # Start with simple network, two dense layers
        layer1 = layers.Dense(32, activation="relu")(inputs)
        layer2 = layers.Dense(64, activation="relu")(layer1)
        layer3 = layers.Dense(32, activation="relu")(layer2)
        prediction = layers.Dense(self.seats_available, activation="sigmoid")(layer3)
        return keras.Model(inputs=inputs, outputs=prediction)

    def create_input_training(self):
        num_steps_prices = len(self.prices_possible)
        decreasing_probability = np.linspace(1, 0, num_steps_prices)
        input_prices_train = np.array([np.full(self.seats_available, self.prices_possible[i])
                                       for i in range(num_steps_prices)])
        input_results_train = np.array([np.full(self.seats_available, decreasing_probability[i])
                                        for i in range(num_steps_prices)])
        self.model.fit([input_prices_train], [input_results_train],
                       batch_size=15, shuffle=True, epochs=self.epochs, verbose=False)

    def name(self):
        return self.name

    def get_action(self, state: []):
        if np.random.rand() < self.epsilon:
            self.prices = np.full((self.seats_available), np.random.choice(self.prices_possible))
        else:
            for seat_index in range(self.seats_available):
                top_revenue = 0
                top_price = 0
                for price in self.prices_possible:
                    price_offer = np.full((self.seats_available), price)
                    state_tensor = tf.convert_to_tensor(price_offer)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probabilities = self.model(state_tensor, training=False).numpy()
                    probability_to_sell = action_probabilities[0][seat_index]
                    revenue = probability_to_sell * price
                    if revenue > top_revenue:
                        top_revenue = revenue
                        top_price = price
                self.prices[seat_index] = top_price
        state_tensor = tf.convert_to_tensor(self.prices)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probabilities = self.model(state_tensor, training=False).numpy()
        predicted_seats = sum(action_probabilities[0])
        return self.prices, predicted_seats

    def process_data(self, action, start_state, prediction, round_revenue, new_state):
        self.action_history.append(np.copy(action))
        self.state_history.append(np.copy(start_state))
        self.prediction_history.append(np.copy(prediction))
        self.rewards_history.append(np.copy(round_revenue))
        seats_sold = sum(new_state)
        result = new_state.astype(int)
        self.state_next_history.append(result)
        self.frame_count += 1

        if (self.frame_count % self.update_after_actions == 0 and len(self.action_history) > self.batch_size):
            # Get indices of samples for replay buffers
            size = min(self.batch_size, self.frame_count)
            indices = np.random.choice(range(len(self.action_history)), size=size)

            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = np.array([self.action_history[i] for i in indices])
            prediction_sample = [self.prediction_history[i] for i in indices]

            self.model.fit([action_sample], [state_next_sample],
                           batch_size=size, shuffle=True, epochs=self.epochs, verbose=False)
            self.epsilon *= 0.95