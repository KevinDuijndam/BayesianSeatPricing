import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers

from tensorflow.python.framework.ops import disable_eager_execution

from collections import deque

EPOCHS = 10
LOSS_CLIPPING = 0.2
NOISE = 1.0
BUFFER_SIZE = 2048
BATCH_SIZE = 256
NUM_ACTIONS = 4
NUM_STATE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 1
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss

class PPO:
    def __init__(self, seats_available: int):
        disable_eager_execution()
        self.action_size = seats_available
        self.val = False
        self.DUMMY_ACTION, self.DUMMY_VALUE = np.zeros((1, self.action_size)), np.zeros((1, 1))
        self.scaling = 50
        self.critic = self.build_critic()
        self.actor = self.build_actor()
        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.state_next_history = deque(maxlen=100)
        self.rewards_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.history = deque(maxlen=100)
        self.frame_count = 0
        self.batch_size = 32
        self.update_after_actions = 4

    def build_critic(self):
        state_input = layers.Input(shape=(self.action_size,))
        x = layers.Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = layers.Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = layers.Dense(1)(x)

        model = keras.Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='mse')

        return model

    def build_actor(self):
        state_input = layers.Input(shape=(self.action_size,))
        advantage = layers.Input(shape=(1,))
        old_prediction = layers.Input(shape=(self.action_size,))

        x = layers.Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = layers.Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = layers.Dense(self.action_size, name='output', activation='softplus')(x)

        model = keras.Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=keras.optimizers.Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()
        return model

    def get_action(self, state: []):
        p = self.actor.predict([state.reshape(1, self.action_size), self.DUMMY_VALUE, self.DUMMY_ACTION])
        if self.val is False:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        else:
            action = action_matrix = p[0]
        return action * self.scaling, p

    def name(self):
        return "PPO"

    def train_model(self, state_sample, prediction_sample, rewards_sample, action_sample):
        old_prediction = np.reshape(np.array(prediction_sample), (32,20))
        pred_values = self.critic.predict(state_sample)

        advantage = rewards_sample - np.reshape(pred_values, (32,))

        action = np.reshape(action_sample, (32, 20))

        actor_loss = self.actor.fit([state_sample, advantage, old_prediction], [action],
                                    batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        critic_loss = self.critic.fit([state_sample], [rewards_sample], batch_size=BATCH_SIZE,
                                      shuffle=True, epochs=EPOCHS, verbose=False)
        #self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
        #self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

    def process_data(self, action, start_state, prediction, round_revenue, new_state):
        self.action_history.append(action)
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

            self.train_model(state_sample=state_sample, prediction_sample=prediction_sample,
                              rewards_sample=rewards_sample, action_sample=action_sample)

