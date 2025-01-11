import numpy as np
from collections import deque
from scipy.stats import norm, invgamma


USE_HIERARCHICAL_MODEL = False

class BayesianApproach:
    def __init__(self, type: str, seats_available: int, epsilon: float, prices_offered: [], nr_flight_types: int):
        self.TYPE = type
        self.seats_available = seats_available
        self.prices_possible = prices_offered
        self.epsilon = epsilon
        self.nr_different_flights = nr_flight_types
        self.skip_offer_price = []
        self.rng = np.random.default_rng(seed=42)
        self.prices = np.full(self.seats_available, self.rng.choice(self.prices_possible))
        if type == "logistic":
            self.name = "Bayesian logistic model"
            self.model = []     # Create logistic model per seat
        else:
            self.name = "Bayesian linear model"
            # Just assume something for prior

            prior_mu_value = -1.5
            prior_sigma_value = 0.001
            self.prior_mu = np.full(self.nr_different_flights, prior_mu_value)

            if USE_HIERARCHICAL_MODEL:
                self.prior_hierarchical_mu = prior_mu_value
                self.posterior_hierarchical_mu = prior_mu_value
                self.prior_hierarchical_sigma = 0.2
                self.posterior_hierarchical_sigma = 0.2
                self.prior_mu = np.array([self.rng.normal(self.prior_hierarchical_mu, self.prior_hierarchical_sigma)
                                          for _ in range(self.nr_different_flights)])
                self.mu_initialised = np.full(self.nr_different_flights, False)
                self.name = self.name + " hierarchical"

            self.prior_mu = np.full(self.nr_different_flights, prior_mu_value)
            self.prior_sigma = np.full(self.nr_different_flights, prior_sigma_value)
            self.prior_k = np.full(self.nr_different_flights, 1)
            self.prior_v = np.full(self.nr_different_flights, 50)
            self.posterior_mu = np.full(self.nr_different_flights, -1.5)
            self.posterior_sigma = np.full(self.nr_different_flights, 0.001)
            self.intercept = np.full(self.nr_different_flights, self.seats_available * 2)
        self.action_history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.state_history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.state_next_history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.rewards_history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.prediction_history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.history = [deque(maxlen=100000) for _ in range(self.nr_different_flights)]
        self.frame_count = np.full(self.nr_different_flights, 0)

    def name(self):
        return self.name

    def initialise_for_flight_type(self, flight_type):
        if USE_HIERARCHICAL_MODEL:
            if not self.mu_initialised[flight_type]:
                self.prior_mu[flight_type] = self.rng.normal(self.posterior_hierarchical_mu,
                                                              self.posterior_hierarchical_sigma)
                self.mu_initialised[flight_type] = True

    def get_prediction(self, price_offer, flight_type):
        # Sample parameter and take exp for log-normal
        a = np.exp(self.rng.normal(self.prior_mu[flight_type], self.prior_sigma[flight_type]))
        result = self.intercept[flight_type] - a * price_offer
        if result > self.seats_available:
            result = self.seats_available
        return result

    def get_action(self, state: [], flight_type=0):
        if self.TYPE == "linear":
            if self.frame_count[flight_type] < 5:
                price = self.rng.choice(self.prices_possible)
                self.prices = np.full(self.seats_available, price)
                predicted_seats = self.get_prediction(price, flight_type)
                return self.prices, predicted_seats
            top_revenue = 0
            top_price = 0
            top_seats = 0
            idx = 0
            for price in self.prices_possible:
                # price_offer = np.full((self.seats_available), price)
                seats = self.get_prediction(price, flight_type)
                revenue = seats * price
                if revenue > top_revenue:
                    top_revenue = revenue
                    top_price = price
                    top_seats = seats
                idx += 1
            self.prices = np.full(self.seats_available, top_price)
            predicted_seats = top_seats
            return self.prices, predicted_seats

    def draw_samples(self, data, prior_m, prior_k, prior_s_sq, prior_v, n_samples=10000):
        N = np.size(data)
        mean_data = np.mean(data)
        SSD = sum((data-mean_data)**2)

        posterior_k = float(prior_k + N)
        posterior_m = (prior_k/posterior_k)*prior_m + (N/posterior_k)*mean_data
        posterior_v = prior_v + N
        posterior_v_s_sq = prior_v * prior_s_sq + SSD + (N*prior_k*(prior_m-mean_data)**2)/posterior_k

        alpha = posterior_v / 2
        beta = posterior_v_s_sq / 2

        sig_sq_samples = beta * invgamma.rvs(alpha, size=n_samples, random_state=self.rng)

        mean_norm = posterior_m
        var_norm = np.sqrt(sig_sq_samples) / posterior_k
        mu_samples = norm.rvs(mean_norm, scale=var_norm, size=n_samples)
        return mean_norm, var_norm, mu_samples, sig_sq_samples

    def draw_log_normal_means(self, data, price_offers, prior_m, prior_k, prior_s_sq, prior_v,
                              flight_type, n_samples=10000):
        log_data = np.log((self.intercept[flight_type] - np.array(data)) / np.array(price_offers))
        log_data = log_data[np.isfinite(log_data)]
        mean_norm, var_norm, mu_samples, sig_sq_samples = self.draw_samples(log_data, prior_m, prior_k, prior_s_sq,
                                                                            prior_v, n_samples)
        #log_normal_mean_samples = np.exp(mu_samples + sig_sq_samples/2)
        log_normal_mean_samples = mu_samples + sig_sq_samples/2
        log_normal_sig_sq_samples = np.sqrt((np.exp(sig_sq_samples) - 1) * np.exp(2*mu_samples + sig_sq_samples))

        return log_normal_mean_samples, log_normal_sig_sq_samples, mean_norm, var_norm

    def update_during_booking(self, booking_index, total_customers, action,
                              start_state, prediction, current_revenue, current_state, flight_type):
        if self.frame_count < 5:
            return action, False
        if self.TYPE == "linear":
            seats_sold = sum(current_state)
            sampled_means, sampled_variance, mean_norm, var_norm = self.draw_log_normal_means(
                                                                            self.state_next_history,
                                                                            self.action_history, self.prior_mu,
                                                                            self.prior_k, self.prior_sigma,
                                                                            self.prior_v)
            sampled_seats_sold = self.intercept - sampled_means * action[0]
            current_percentage = float(booking_index) / float(total_customers)
            current_expected_seats_sold = current_percentage * sampled_seats_sold
            if np.mean(current_expected_seats_sold > seats_sold) < 0.05:
                action = action * 1.1
            elif np.mean(current_expected_seats_sold < seats_sold) < 0.05:
                action = action * 0.9
            return action, True
        else:
            return False

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered, flight_type=0):
        self.state_history[flight_type].append(np.copy(start_state))
        self.prediction_history[flight_type].append(prediction)
        self.rewards_history[flight_type].append(round_revenue)
        if self.TYPE == "linear":
            self.action_history[flight_type].append(np.copy(action[0]))
            seats_sold = sum(new_state)
            self.state_next_history[flight_type].append(seats_sold)
            self.frame_count[flight_type] += 1
            if self.frame_count[flight_type] > 5:
                sampled_means, sampled_variance, mean_norm, var_norm = self.draw_log_normal_means(
                                                                                self.state_next_history[flight_type],
                                                                                self.action_history[flight_type],
                                                                                np.copy(self.prior_mu[flight_type]),
                                                                                np.copy(self.prior_k[flight_type]),
                                                                                np.copy(self.prior_sigma[flight_type]),
                                                                                np.copy(self.prior_v[flight_type]),
                                                                                flight_type)

                #mean_update = np.mean(sampled_means)
                #sigma_update = np.mean(sampled_variance)
                #self.posterior_mu[flight_type] = mean_update
                #self.posterior_sigma[flight_type] = sigma_update
                self.posterior_mu[flight_type] = mean_norm
                self.posterior_sigma[flight_type] = np.mean(var_norm)
                #if self.frame_count % 25 == 0:
                #    print("Prior mu: " + str(self.prior_mu) + ", posterior_mu: " + str(self.posterior_mu))
                self.prior_mu[flight_type] = self.posterior_mu[flight_type]
                self.prior_sigma[flight_type] = self.posterior_sigma[flight_type]

                if USE_HIERARCHICAL_MODEL:
                    data = self.state_next_history[flight_type]
                    N = np.size(data)
                    mean_data = np.mean(data)
                    SSD = sum((data - mean_data) ** 2) + 1e-5
                    self.posterior_hierarchical_sigma = (1 / self.prior_hierarchical_sigma + N / SSD)**-1
                    self.posterior_hierarchical_mu = (1 / ((1/self.prior_hierarchical_sigma) +
                                                    (N / self.posterior_hierarchical_sigma))) * \
                                                    ((self.prior_hierarchical_mu / self.prior_hierarchical_sigma) + \
                                                    sum(data) / SSD)

