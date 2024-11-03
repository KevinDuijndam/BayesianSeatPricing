import numpy as np


def get_state(current_state: []):
    return int(''.join(current_state.astype(int).astype(str)), 2)


def get_state_price(price_chosen, price_levels):
    return np.where(price_levels == price_chosen[0])[0][0]


class QLearning:
    def __init__(self, epsilon: float, lr: float, gamma: float, seats_available: int, price_levels: []):
        self.epsilon = epsilon
        self.lr = lr
        self.name = "Q-Learning"
        self.gamma = gamma
        self.seats_available = seats_available
        #self.Q = np.zeros([2**self.seats_available, len(price_levels)])
        self.Q = np.zeros(len(price_levels))
        self.max_price = len(price_levels)
        self.price_levels = price_levels

    def calculate_step(self, current_state: []):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(self.max_price)
            return choice, self.price_levels[choice]
        else:
            #representation = get_state(current_state)
            #choice = np.argmax(self.Q[representation, :])
            choice = np.argmax(self.Q)
            return choice, self.price_levels[choice]

    def get_action(self, state: [], flight_type):
        if np.random.rand() < self.epsilon:
            choice = np.random.choice(self.max_price)
            prices = np.full(self.seats_available, self.price_levels[choice], dtype=float)
            return prices, choice
        else:
            #representation = get_state(state)
            #choice = np.argmax(self.Q[representation, :])
            choice = np.argmax(self.Q)
            prices = np.full(self.seats_available, self.price_levels[choice], dtype=float)
            return prices, choice

    def initialise_for_flight_type(self, flight_type):
        return

    def updateQ(self, state_at_action: [], new_state: [], action: int, reward: float):
        #previous_state_index = get_state(state_at_action)
        #new_state_index = get_state(new_state)
        previous_state_index = get_state_price(action, self.price_levels)
        new_state_index = get_state_price(action, self.price_levels)
        self.Q[previous_state_index, action] = (1 - self.lr) * self.Q[previous_state_index, action] + \
                                            self.lr * (reward + self.gamma * max(self.Q[new_state_index, :]))

    def update_during_booking(self, booking_index, total_customers, action,
                              start_state, prediction, current_revenue, current_state):
        return action, False

    def process_data(self, action, start_state, prediction, round_revenue, new_state,
                     customers_offered, flight_type):
        #previous_state_index = get_state(start_state)
        #new_state_index = get_state(new_state)
        previous_state_index = get_state_price(action, self.price_levels)
        new_state_index = get_state_price(action, self.price_levels)
        self.Q[previous_state_index] = (1 - self.lr) * self.Q[previous_state_index] + \
                                            self.lr * (round_revenue + self.gamma * self.Q[new_state_index])

    def name(self):
        return "Q-Learning"

    def print_matrix(self):
        print(self.Q)

