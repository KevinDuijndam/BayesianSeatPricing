import numpy as np
from collections import deque
import scipy.stats as stats

USE_MULTIPLE_FLIGHT_MODEL = False
USE_HIERARCHICAL_MODEL = False


class ProbaPrediction:
    def __init__(self, type: str, seats_available: int, prices_offered: [], nr_flight_types: int):
        self.TYPE = type
        self.name = "Binomial"
        self.seats_available = seats_available
        self.prices_possible = prices_offered
        self.prices = np.full((self.seats_available), np.random.choice(self.prices_possible))
        if USE_MULTIPLE_FLIGHT_MODEL:
            self.nr_different_flights = nr_flight_types
            self.action_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.state_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.state_next_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.number_offer_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.rewards_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.prediction_history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.history = [deque(maxlen=100) for _ in range(self.nr_different_flights)]
            self.frame_count = np.full(self.nr_different_flights, 0)
            self.name = self.name + " multi-flight"
        else:
            self.action_history = deque(maxlen=100)
            self.state_history = deque(maxlen=100)
            self.state_next_history = deque(maxlen=100)
            self.number_offer_history = deque(maxlen=100)
            self.rewards_history = deque(maxlen=100)
            self.prediction_history = deque(maxlen=100)
            self.history = deque(maxlen=100)
            self.frame_count = 0
        if USE_MULTIPLE_FLIGHT_MODEL:
            self.p = [np.full(len(self.prices_possible), 0.2) for _ in range(self.nr_different_flights)]
        else:
            self.p = np.full(len(self.prices_possible), 0.2)
        if "Bayesian" in type:
            self.bayesian = True
        else:
            self.bayesian = False

        if self.bayesian:
            if USE_MULTIPLE_FLIGHT_MODEL:
                if USE_HIERARCHICAL_MODEL:
                    self.prior_alpha_hierarchical_mu = 4
                    self.posterior_alpha_hierarchical_mu = 4
                    self.prior_beta_hierarchical_mu = 10
                    self.posterior_beta_hierarchical_mu = 10
                    self.prior_hierarchical_sigma = 0.25
                    self.posterior_hierarchical_sigma = 0.25
                    self.alpha = list(
                                          np.array([np.random.normal(self.prior_alpha_hierarchical_mu,
                                                                     self.prior_hierarchical_sigma)
                                                    for _ in range(len(self.prices_possible))])
                                          for _ in range(self.nr_different_flights))
                    self.beta = list(
                                          np.array([np.random.normal(self.prior_beta_hierarchical_mu,
                                                                     self.prior_hierarchical_sigma)
                                                    for _ in range(len(self.prices_possible))])
                                          for _ in range(self.nr_different_flights))
                    for i in range(len(self.alpha)):
                        for j in range(len(self.alpha[i])):
                            if self.alpha[i][j] <= 0:
                                print("Error")
                    self.prior_initialised = np.full(self.nr_different_flights, False)
                else:
                    self.alpha = [np.full(len(self.prices_possible), 4) for _ in range(self.nr_different_flights)]
                    self.beta = [np.full(len(self.prices_possible), 10) for _ in range(self.nr_different_flights)]
            else:
                self.alpha = np.full(len(self.prices_possible), 4)
                self.beta = np.full(len(self.prices_possible), 10)
            self.name = self.name + " Bayesian"
            if USE_HIERARCHICAL_MODEL:
                self.name = self.name + " Hierarchical"
        if "UCB" in type:
            self.UCB = True
            self.name = self.name + " UCB"
        else:
            self.UCB = False

    def name(self):
        return self.name

    def initialise_for_flight_type(self, flight_type):
        if USE_HIERARCHICAL_MODEL:
            if not self.prior_initialised[flight_type]:
                self.alpha[flight_type] = np.array([np.random.normal(self.posterior_alpha_hierarchical_mu,
                                                                     self.posterior_hierarchical_sigma)
                                                    for _ in range(len(self.prices_possible))])
                self.beta[flight_type] = np.array([np.random.normal(self.posterior_beta_hierarchical_mu,
                                                                     self.posterior_hierarchical_sigma)
                                                   for _ in range(len(self.prices_possible))])
                self.prior_initialised[flight_type] = True
                for j in range(len(self.alpha[flight_type])):
                    if self.alpha[flight_type][j] <= 0:
                        self.alpha[flight_type][j] = 0.01

    def get_probability_with_price(self, price, flight_type):
        if self.bayesian:
            price_index = np.where(self.prices_possible == price)
            if self.UCB:
                if USE_MULTIPLE_FLIGHT_MODEL:
                    return stats.beta.ppf(0.9, self.alpha[flight_type][price_index], self.beta[flight_type][price_index])
                else:
                    return stats.beta.ppf(0.9, self.alpha[price_index], self.beta[price_index])
            else:
                if USE_MULTIPLE_FLIGHT_MODEL:
                    return stats.beta.rvs(self.alpha[flight_type][price_index], self.beta[flight_type][price_index],
                                          size=1)[0]
                else:
                    return stats.beta.rvs(self.alpha[price_index], self.beta[price_index], size=1)[0]
        else:
            if USE_MULTIPLE_FLIGHT_MODEL:
                return self.p[flight_type][np.where(self.prices_possible == price)][0]
            else:
                return self.p[np.where(self.prices_possible == price)][0]


    def get_prediction(self, price, flight_type):
        # Return expectation of binomial distribution
        return self.get_probability_with_price(price, flight_type) * self.seats_available

    def get_action(self, state: [], flight_type = 0):
        optimal_price = 0
        best_expectation = 0
        best_expected_seats = 0
        for i in self.prices_possible:
            expected_seats = self.get_prediction(i, flight_type)
            expected_revenue = expected_seats * i
            if expected_revenue > best_expectation:
                optimal_price = i
                best_expectation = expected_revenue
                best_expected_seats = expected_seats
        return np.full((self.seats_available), optimal_price), expected_seats

    def update_during_booking(self, booking_index, total_customers, action,
                              start_state, prediction, current_revenue, current_state):
        #return action
        seats_sold = sum(current_state) - sum(start_state)
        probability_of_seats_lower_equal = stats.binom.cdf(seats_sold,
                                                           booking_index,
                                                           self.get_probability_with_price(action[0]))

        if probability_of_seats_lower_equal < 0.05:
            # Not supposed to sell so few, so decrease price
            used_price = action[0]
            self.action_history.append(np.copy(used_price))
            self.state_next_history.append(np.copy(seats_sold))
            self.number_offer_history.append(np.copy(booking_index))
            action = self.prices_possible[np.where(self.prices_possible == action[0])[0] - 1]
            return np.full(self.seats_available, action), True
        elif probability_of_seats_lower_equal > 0.95 and seats_sold > 1:
            # Not supposed to sell everything, so increase price
            used_price = action[0]
            self.action_history.append(np.copy(used_price))
            self.state_next_history.append(np.copy(seats_sold))
            self.number_offer_history.append(np.copy(booking_index))
            action = self.prices_possible[np.where(self.prices_possible == action[0])[0] + 1]
            return np.full(self.seats_available, action), True
        return np.full(self.seats_available, action), False

    def new_flight(self):
        if self.bayesian:
            for idx in range(len(self.prices_possible)):
                if self.alpha[idx] != 4 and self.beta[idx] != 10:
                    to_divide = self.beta[idx] / 2
                    self.alpha[idx] = self.alpha[idx] / to_divide
                    if self.alpha[idx] == 0:
                        self.alpha[idx] = 1
                    self.beta[idx] = self.beta[idx] / to_divide

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered, flight_type):
        used_price = action[0]
        seats_sold = sum(new_state) - sum(start_state)
        if USE_MULTIPLE_FLIGHT_MODEL:
            self.action_history[flight_type].append(np.copy(used_price))
            self.state_history[flight_type].append(np.copy(start_state))
            self.prediction_history[flight_type].append(prediction)
            self.rewards_history[flight_type].append(round_revenue)
            self.state_next_history[flight_type].append(seats_sold)
            self.number_offer_history[flight_type].append(times_offered)
            self.frame_count[flight_type] += 1
            prev_actions = np.array(self.action_history[flight_type])
        else:
            self.action_history.append(np.copy(used_price))
            self.state_history.append(np.copy(start_state))
            self.prediction_history.append(prediction)
            self.rewards_history.append(round_revenue)
            self.state_next_history.append(seats_sold)
            self.number_offer_history.append(times_offered)
            self.frame_count += 1
            prev_actions = np.array(self.action_history)

        # Update price - probability figures
        for used_price in self.prices_possible:
            if USE_MULTIPLE_FLIGHT_MODEL:
                prev_results = np.array(self.state_next_history[flight_type])[
                    np.where(prev_actions == used_price)]
                prev_number_offered = np.array(self.number_offer_history[flight_type])[
                    np.where(prev_actions == used_price)]
            else:
                prev_results = np.array(self.state_next_history)[np.where(prev_actions == used_price)]
                prev_number_offered = np.array(self.number_offer_history)[np.where(prev_actions == used_price)]

            if len(prev_results) > 0:
                total_sold_prev = prev_results
                if self.bayesian:
                    # Update Bayesian
                    price_to_update = used_price
                    price_index = np.where(self.prices_possible == price_to_update)
                    if USE_MULTIPLE_FLIGHT_MODEL:
                        new_alpha = self.alpha[flight_type][price_index] + sum(prev_results)
                        new_beta = self.beta[flight_type][price_index] + sum(prev_number_offered) - sum(prev_results)
                        self.alpha[flight_type][price_index] = new_alpha
                        self.beta[flight_type][price_index] = new_beta
                    else:
                        new_alpha = self.alpha[price_index] + sum(prev_results)
                        new_beta = self.beta[price_index] + sum(prev_number_offered) - sum(prev_results)
                        self.alpha[price_index] = new_alpha
                        self.beta[price_index] = new_beta
                else:
                    new_probability = np.mean(total_sold_prev / prev_number_offered)
                    if USE_MULTIPLE_FLIGHT_MODEL:
                        self.p[flight_type][np.where(self.prices_possible == used_price)] = new_probability
                    else:
                        self.p[np.where(self.prices_possible == used_price)] = new_probability

        if USE_HIERARCHICAL_MODEL:
            data = self.state_next_history[flight_type]
            N = np.size(data)
            mean_data = np.mean(data)
            SSD = sum((data - mean_data) ** 2) + 1e-5
            self.posterior_hierarchical_sigma = (1 / self.prior_hierarchical_sigma + N / SSD) ** -1
            self.posterior_alpha_hierarchical_mu = (1 / ((1 / self.prior_hierarchical_sigma) +
                                                    (N / self.posterior_hierarchical_sigma))) * \
                                             ((self.prior_alpha_hierarchical_mu / self.prior_hierarchical_sigma) + \
                                              sum(data) / SSD)
            self.posterior_beta_hierarchical_mu = (1 / ((1 / self.prior_hierarchical_sigma) +
                                                         (N / self.posterior_hierarchical_sigma))) * \
                                                   ((self.prior_beta_hierarchical_mu / self.prior_hierarchical_sigma) + \
                                                    sum(data) / SSD)

