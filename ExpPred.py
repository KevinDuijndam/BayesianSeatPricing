from collections import deque
import aesara
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

RANDOM_SEED = 42
from scipy import stats
rng = np.random.default_rng(RANDOM_SEED)
USE_HIERARCHICAL_MODEL = True

class ExpPred:
    def __init__(self, type: str, seats_available: int, epsilon: float, prices_offered: [], nr_flight_types: int):
        self.TYPE = type
        self.seats_available = seats_available
        self.prices_possible = prices_offered
        self.nr_different_flights = nr_flight_types
        self.epsilon = epsilon
        self.prices = np.full((self.seats_available), np.random.choice(self.prices_possible))
        if self.TYPE == "exp":
            self.name = "Exponential model"
            prior_intercept_mu_value = 6.0
            prior_intercept_sigma_value = 2.0
            prior_slope_mu_value = -0.09
            prior_slope_sigma_value = 0.02
            prior_sigma_value = 4

            self.prior_intercept_mu = np.full(self.nr_different_flights, prior_intercept_mu_value)
            self.prior_slope_mu = np.full(self.nr_different_flights, prior_slope_mu_value)
            self.prior_sigma = np.full(self.nr_different_flights, prior_sigma_value)

            self.prior_intercept_sigma = np.full(self.nr_different_flights, prior_intercept_sigma_value)
            self.prior_slope_sigma = np.full(self.nr_different_flights, prior_slope_sigma_value)
            self.prior_sigma_sigma = 2

            self.input_mu = np.array([self.prior_intercept_mu, self.prior_intercept_sigma, self.prior_slope_mu, self.prior_slope_sigma])
            self.input_mu = self.input_mu.T
            self.p = self.input_mu

            if USE_HIERARCHICAL_MODEL:
                self.prior_hierarchical_intercept_mu = prior_intercept_mu_value
                self.prior_hierarchical_intercept_sigma = prior_intercept_sigma_value
                self.prior_hierarchical_slope_mu = prior_slope_mu_value
                self.prior_hierarchical_slope_sigma = prior_slope_sigma_value

                self.name = self.name + " hierarchical"

                self.prior_intercept_mu = np.array([np.random.normal(self.prior_hierarchical_intercept_mu,
                                                                     self.prior_hierarchical_intercept_sigma)
                                          for _ in range(self.nr_different_flights)])
                self.prior_intercept_sigma = np.array([np.random.normal(self.prior_hierarchical_intercept_sigma,
                                                                     self.prior_hierarchical_intercept_sigma)
                                                    for _ in range(self.nr_different_flights)])
                self.prior_slope_mu = np.array([np.random.normal(self.prior_hierarchical_slope_mu,
                                                                 self.prior_hierarchical_slope_sigma)
                                            for _ in range(self.nr_different_flights)])
                self.prior_slope_sigma = np.array([np.random.normal(self.prior_hierarchical_slope_sigma,
                                                              self.prior_hierarchical_slope_sigma)
                                               for _ in range(self.nr_different_flights)])

                self.posterior_hierarchical_intercept_mu = self.prior_hierarchical_intercept_mu
                self.posterior_hierarchical_intercept_sigma = self.prior_hierarchical_intercept_sigma
                self.posterior_hierarchical_slope_mu = self.prior_hierarchical_slope_mu
                self.posterior_hierarchical_slope_sigma = self.prior_hierarchical_slope_sigma

                self.input_mu = [np.array([self.prior_intercept_mu[i], self.prior_intercept_sigma[i],
                                           self.prior_slope_mu[i], self.prior_slope_sigma[i]])
                                 for i in range(self.nr_different_flights)]
                self.p = [self.input_mu[i] for i in range(self.nr_different_flights)]
                self.flight_initialised = np.full(self.nr_different_flights, False)

            self.n_samples = 5000
            self.burn_in = 1000
            self.lag = 5

            self.mcmc_results = [[] for _ in range(self.nr_different_flights)]
            self.model_initialised = [False for _ in range(self.nr_different_flights)]
            self.stop_sampling = [False for _ in range(self.nr_different_flights)]

            self.intercepts = [[] for _ in range(self.nr_different_flights)]
            self.intercept_sigmas = [[] for _ in range(self.nr_different_flights)]
            self.slopes = [[] for _ in range(self.nr_different_flights)]
            self.slope_sigmas = [[] for _ in range(self.nr_different_flights)]
            #self.sigmas = [[] for _ in range(self.nr_different_flights)]
        else:
            self.data_used_for_training = ""
            with pm.Model() as self.model:  # model specifications in PyMC are wrapped in a with-statement
                # Define priors
                self.sigma = pm.HalfCauchy("sigma", beta=10)
                self.intercept = pm.Normal("Intercept", self.seats_available / 2, sigma=self.seats_available / 4)
                self.slope = pm.Normal("slope", -0.1, sigma=0.1)
                self.model_initialised = False
                self.likelihood = None
                self.idata = None

        self.action_history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.state_history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.state_next_history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.rewards_history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.prediction_history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.history = [deque(maxlen=10000) for _ in range(self.nr_different_flights)]
        self.frame_count = np.full(self.nr_different_flights, 0)

    def name(self):
        return self.name

    def initialise_for_flight_type(self, flight_type):
        if USE_HIERARCHICAL_MODEL:
            if not self.flight_initialised[flight_type]:
                self.prior_slope_mu[flight_type] = np.random.normal(self.posterior_hierarchical_slope_mu,
                                                              self.posterior_hierarchical_slope_sigma)
                self.prior_slope_sigma[flight_type] = np.random.normal(self.posterior_hierarchical_slope_sigma,
                                                                    self.posterior_hierarchical_slope_sigma)
                self.prior_intercept_mu[flight_type] = np.random.normal(self.posterior_hierarchical_intercept_mu,
                                                                    self.posterior_hierarchical_intercept_sigma)
                self.prior_intercept_sigma[flight_type] = np.random.normal(self.posterior_hierarchical_intercept_sigma,
                                                                    self.posterior_hierarchical_intercept_sigma)
                self.flight_initialised[flight_type] = True

    def get_prediction(self, price_offer, flight_type):
        if self.TYPE == "exp":
            if self.model_initialised[flight_type]:
                mean_intercept = np.mean(self.intercepts[flight_type])
                mean_slope = np.mean(self.slopes[flight_type])
                all_results = np.array([self.seats_available /
                                        (1 + np.exp(-(self.intercepts[flight_type] + self.slopes[flight_type] * i)))
                                        for i in self.prices_possible])

                sample_max = np.percentile(all_results, 90, axis=1)
                sample_min = np.percentile(all_results, 10, axis=1)

                mean_outcome = self.seats_available / (
                            1 + np.exp(-(mean_intercept + mean_slope * self.prices_possible)))

                price_index = np.where(self.prices_possible == price_offer[0])

                return mean_outcome[price_index]
                #return sample_max[price_index]
            else:
                # Use prior
                all_results = np.array([self.seats_available /
                                        (1 + np.exp(-(self.prior_intercept_mu[flight_type] + self.prior_slope_mu[flight_type] * i)))
                                        for i in self.prices_possible])
                price_index = np.where(self.prices_possible == price_offer[0])

                return all_results[price_index]
        else:
            if self.model_initialised:
                with self.model:
                    pm.sample_posterior_predictive(self.idata, extend_inferencedata=True, random_seed=rng)
                post = self.idata.posterior
                mu_pp = self.seats_available / (
                        1 + np.exp(
                    -(post["Intercept"] + post["slope"] * xr.DataArray(self.prices_possible, dims=["obs_id"]))))
                expected_results = np.array(mu_pp.mean(("chain", "draw")))
                return expected_results
            else:
                return 1

    def get_action(self, state: [], flight_type):
        predicted_seats = 0
        if self.TYPE == "exp":
            top_revenue = 0
            top_price = 0
            idx = 0
            for price in self.prices_possible:
                price_offer = np.full((self.seats_available), price)
                seats = self.get_prediction(price_offer, flight_type)
                revenue = seats * price
                if revenue > top_revenue:
                    top_revenue = revenue
                    top_price = price
                idx += 1
            self.prices = np.full((self.seats_available), top_price)
        else:
            with self.model:
                pm.sample_posterior_predictive(self.idata, extend_inferencedata=True, random_seed=rng)
            post = self.idata.posterior
            mu_pp = self.seats_available / (
                    1 + np.exp(
                -(post["Intercept"] + post["slope"] * xr.DataArray(self.prices_possible, dims=["obs_id"]))))
            expected_results = np.array(mu_pp.mean(("chain", "draw")))
            top_revenue = 0
            top_price = 0
            best_idx = 0
            for idx in range(len(self.prices_possible)):
                price = self.prices_possible[idx]
                price_offer = np.full((self.seats_available), price)
                seats = expected_results[idx]
                revenue = seats * price
                if revenue > top_revenue:
                    top_revenue = revenue
                    top_price = price
                    best_idx = idx
                idx += 1
            self.prices = np.full((self.seats_available), top_price)
            predicted_seats = expected_results[best_idx]
        return self.prices, predicted_seats

    def get_prices_for_seat(self, seat_index: int):
        result = []
        for i in range(len(self.action_history)):
            result.append(self.action_history[i][seat_index])
        return result

    def get_sales_for_seat(self, seat_index: int):
        result = []
        for i in range(len(self.state_next_history)):
            result.append(self.state_next_history[i][seat_index])
        return result

    def get_prices_for_middle(self):
        result = []
        mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                for i in range(self.seats_available)]  # Get middle seat idx
        for i in range(len(self.action_history)):
            result.append(self.action_history[i][mask])
        return result

    def get_sales_for_middle(self):
        result = []
        mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                for i in range(self.seats_available)]  # Get middle seat idx
        for i in range(len(self.state_next_history)):
            result.append(sum(self.state_next_history[i][mask]))
        return result

    def get_prices_for_window(self):
        result = []
        mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                for i in range(self.seats_available)]  # Get middle seat idx
        for i in range(len(self.action_history)):
            result.append(self.action_history[i][~np.array(mask)])
        return result

    def get_sales_for_window(self):
        result = []
        mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                for i in range(self.seats_available)]  # Get middle seat idx
        for i in range(len(self.state_next_history)):
            result.append(sum(self.state_next_history[i][~np.array(mask)]))
        return result

    def update_during_booking(self, booking_index, total_customers, action,
                              start_state, prediction, current_revenue, current_state):
        if self.TYPE == "exp":
            return action, False
        return action, False

    def log_likelihood(self, sample_slope, sample_intercept, sample_sigma, x, y):
        predicted_values = self.seats_available / (1 + np.exp(-(sample_intercept + sample_slope * x)))
        log_likelihoods = np.log(stats.norm.pdf(y, predicted_values, sample_sigma))
        return sum(log_likelihoods)

    def normal(self, x, mu, sigma):
        numerator_input = (-(x - mu) ** 2) / (2 * sigma ** 2)
        numerator_input[numerator_input < -1e2] = -1e2
        numerator = np.exp(numerator_input)
        denominator = sigma * np.sqrt(2 * np.pi)
        return numerator / denominator

    def compute_likelihood(self, seats_available, slope_mu, slope_sigma, intercept_mu, intercept_sigma, x, data):
        input_data = np.copy(data).astype(float)
        input_data[input_data == 0] = 1e-1
        input_data[input_data == seats_available] = seats_available - 1e-1
        input_for_normal = -1 * np.log((seats_available / input_data) - 1)
        return self.normal(input_for_normal, slope_mu * x + intercept_mu, intercept_sigma + x ** 2 * slope_sigma)

    def log_likelihood_of_data(self, sample_slope, sample_slope_sigma, sample_intercept, sample_intercept_sigma,
                               input_x, input_y):
        likelihoods = self.compute_likelihood(self.seats_available, sample_slope, sample_slope_sigma,
                                                    sample_intercept, sample_intercept_sigma, input_x, input_y)
        likelihoods[likelihoods <= 0] = 1e-100
        log_likelihoods = np.log(likelihoods)

        return sum(log_likelihoods)

    def prior(self, current_value):
        return current_value + np.random.multivariate_normal(np.zeros(2), np.eye(2) * self.prior_sigma)

    def get_prior_figures(self):
        return [self.prior_slope_mu, self.prior_slope_sigma,
                self.prior_intercept_mu, self.prior_intercept_sigma,
                self.prior_sigma, self.prior_sigma_sigma]

    def sample_mcmc(self, prices_used, observed_data, flight_type):
        if self.stop_sampling[flight_type]:
            if np.random.uniform() < 0.95:
                return

        self.mcmc_results[flight_type] = []
        nr_accepted = 0
        nr_sampled = 0
        for i in range(self.n_samples):
            p_prime_slope_mu = self.prior_slope_mu[flight_type] + np.random.normal(0, 0.02)
            p_prime_slope_sigma = self.prior_slope_sigma[flight_type] + np.random.normal(0, 0.02)
            p_prime_intercept_mu = self.prior_intercept_mu[flight_type] + np.random.normal(0, 0.50)
            p_prime_intercept_sigma = self.prior_intercept_sigma[flight_type] + np.random.normal(0, 0.2)
            nr_sampled += 1

            p_prime = [p_prime_intercept_mu, p_prime_intercept_sigma, p_prime_slope_mu, p_prime_slope_sigma]
            input_ratio = self.log_likelihood_of_data(p_prime[2], p_prime[3], p_prime[0], p_prime[1], prices_used, observed_data) - \
                           self.log_likelihood_of_data(self.p[flight_type][2], self.p[flight_type][3], self.p[flight_type][0], self.p[flight_type][1], prices_used, observed_data)
            if input_ratio < -100:
                input_ratio = -100
            elif input_ratio > 100:
                input_ratio = 100
            ratio = np.exp(input_ratio)

            u = np.random.uniform()
            if ratio > u:
                nr_accepted += 1
                self.p[flight_type] = p_prime

            self.mcmc_results[flight_type].append(self.p[flight_type])

        ratio_accepted = float(nr_accepted) / float(nr_sampled)
        if ratio_accepted < 0.0001:
            self.stop_sampling[flight_type] = False
        #print("Nr sampled: " + str(nr_sampled) + ", nr accepted: " + str(nr_accepted) + ". Ratio: " + str(
        #    float(nr_accepted) / float(nr_sampled)))

    def update_hierarchical_models(self):
        initial_slope_mus = []
        initial_intercept_mus = []
        initial_slope_sigmas = []
        initial_intercept_sigmas = []
        for flight in range(self.nr_different_flights):
            if self.model_initialised[flight]:
                initial_slope_mus.append(self.prior_slope_mu[flight])
                initial_slope_sigmas.append(self.prior_slope_sigma[flight])
                initial_intercept_mus.append(self.prior_intercept_mu[flight])
                initial_intercept_sigmas.append(self.prior_intercept_sigma[flight])

        N = len(initial_slope_mus)
        # Update slope values
        mean_data = np.mean(initial_slope_mus)
        SSD = sum((np.array(initial_slope_mus) - mean_data) ** 2) + 1e-5
        self.posterior_hierarchical_slope_sigma = (1 / self.prior_hierarchical_slope_sigma + N / SSD) ** -1
        self.posterior_hierarchical_slope_mu = (1 / ((1 / self.prior_hierarchical_slope_mu) +
                                               (N / self.posterior_hierarchical_slope_sigma))) * \
                                         ((self.prior_hierarchical_slope_mu / self.prior_hierarchical_slope_sigma) + \
                                          sum(np.array(initial_slope_mus)) / SSD)

        # Update intercept values
        mean_data = np.mean(initial_intercept_mus)
        SSD = sum((np.array(initial_intercept_mus) - mean_data) ** 2) + 1e-5
        self.posterior_hierarchical_intercept_sigma = (1 / self.prior_hierarchical_intercept_sigma + N / SSD) ** -1
        self.posterior_hierarchical_intercept_mu = (1 / ((1 / self.prior_hierarchical_intercept_mu) +
                                                     (N / self.posterior_hierarchical_intercept_sigma))) * \
                                               ((self.prior_hierarchical_intercept_mu / self.prior_hierarchical_intercept_sigma) + \
                                                sum(np.array(initial_intercept_mus)) / SSD)

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered, flight_type):
        self.action_history[flight_type].append(np.copy(action))
        self.state_history[flight_type].append(np.copy(start_state))
        self.prediction_history[flight_type].append(prediction)
        self.rewards_history[flight_type].append(round_revenue)

        self.frame_count += 1
        if self.TYPE == "exp":
            seats_sold = sum(new_state)
            self.state_next_history[flight_type].append(seats_sold)
            prices_used = np.array(self.action_history[flight_type])[:, 0]
            observed_data = np.array(self.state_next_history[flight_type])
            self.sample_mcmc(prices_used, observed_data, flight_type)
            intermediate_results = np.array(self.mcmc_results[flight_type])
            self.intercepts[flight_type] = intermediate_results[:, 0]
            self.intercept_sigmas[flight_type] = intermediate_results[:, 1]
            self.slopes[flight_type] = intermediate_results[:, 2]
            self.slope_sigmas[flight_type] = intermediate_results[:, 3]
            self.model_initialised[flight_type] = True

            self.prior_slope_mu[flight_type] = np.mean(self.slopes[flight_type])
            self.prior_slope_sigma[flight_type] = np.mean(self.slope_sigmas[flight_type])

            self.prior_intercept_mu[flight_type] = np.mean(self.intercepts[flight_type])
            self.prior_intercept_sigma[flight_type] = np.mean(self.intercept_sigmas[flight_type])

            if USE_HIERARCHICAL_MODEL:
                self.update_hierarchical_models()

        else:
            seats_sold = sum(new_state)
            self.state_next_history.append(seats_sold)
            if self.frame_count > 5:
                prices_used = np.array(self.action_history)[:, 0]
                u, c = np.unique(prices_used, return_counts=True)
                duplicates = u[c > 1]
                observed_data = np.array(self.state_next_history)
                # sorted_indexes = prices_used.argsort()
                # sorted_results = observed_data[sorted_indexes[::]]
                for duplicate in duplicates:
                    indexes = np.nonzero(prices_used == duplicate)[0][0:-1]
                    prices_used = np.delete(prices_used, indexes, axis=0)
                    observed_data = np.delete(observed_data, indexes, axis=0)
                # Add NaN for everything not observed
                y_observed = np.full(len(self.prices_possible), np.NaN)
                for idx in range(len(self.prices_possible)):
                    resulting_index = np.where(prices_used == self.prices_possible[idx])[0]
                    if len(resulting_index) == 1:
                        y_observed[idx] = observed_data[resulting_index]

                if self.data_used_for_training != str(prices_used):
                    with pm.Model() as self.model:  # model specifications in PyMC are wrapped in a with-statement
                        # Define priors
                        self.sigma = pm.HalfCauchy("sigma", beta=10)
                        self.intercept = pm.Normal("Intercept", self.seats_available * 2,
                                                   sigma=self.seats_available / 2)
                        self.slope = pm.Normal("slope", -1.5, sigma=0.5)
                        self.model_initialised = True
                        # Define likelihood
                        self.likelihood = pm.Normal("y", mu=self.seats_available /
                                                            (1 + np.exp(
                                                                -(self.intercept + self.slope * self.prices_possible))),
                                                    sigma=self.sigma, observed=y_observed)

                        # Inference!
                        # draw 3000 posterior samples using NUTS sampling
                        self.idata = pm.sample(3000)
                    self.data_used_for_training = str(prices_used)
                self.epsilon *= 0.95
                self.model_initialised = True