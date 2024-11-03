import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import deque
from scipy.stats import multivariate_normal
from scipy.stats import norm as univariate_normal

import bayes_logistic


class BayesPred:
    def __init__(self, type: str, seats_available: int, epsilon: float, prices_offered: []):
        self.TYPE = type
        self.log_normal = True
        self.seats_available = seats_available
        self.prices_possible = prices_offered
        self.epsilon = epsilon
        self.skip_offer_price = []
        self.prices = np.full((self.seats_available), np.random.choice(self.prices_possible))
        if type == "logistic":
            self.name = "Bayes logistic model"
            self.model = []     # Create logistic model per seat
            for i in range(self.seats_available):
                self.model.append(bayes_logistic.EBLogisticRegression())
                X = np.array([self.prices_possible[0], self.prices_possible[len(self.prices_possible) - 1]]).reshape((-1, 1))
                y = np.array([1, 0])
                self.model[i].fit(X, y)  # Random fit just so it's initialised with 50-50 chance
        else:
            self.name = "Bayes linear model"
            # Assume one parameter for now
            self.intercept_mu = 5
            self.intercept_sigma = 10
            self.beta_mu = 5
            self.beta_sigma = 2
            if self.log_normal:
                # other figures
                prior_mean = np.array([-0.005, 0.05])
                prior_cov = 1 / 2 * np.array([[1e-5, 0], [0, 1e-5]])
                noise_var = 1e-5

                self.intercept = seats_available + 1

                self.prior_mean = prior_mean[:, np.newaxis]  # column vector of shape (1, d)
                self.prior_cov = prior_cov  # matrix of shape (d, d)
                # We initialise the prior distribution over the parameters using the given mean and covariance matrix
                # In the formulas above this corresponds to m_0 (prior_mean) and S_0 (prior_cov)
                self.prior = multivariate_normal(prior_mean, prior_cov)

                # We also know the variance of the noise
                self.noise_var = noise_var  # single float value
                self.noise_precision = 1 / noise_var

                # Before performing any inference the parameter posterior equals the parameter prior
                self.param_posterior = self.prior
                # Accordingly, the posterior mean and covariance equal the prior mean and variance
                self.post_mean = self.prior_mean  # corresponds to m_N in formulas
                self.post_cov = self.prior_cov  # corresponds to S_N in formulas
            else:
                # Set some defaults
                prior_mean = np.array([15, -15])
                prior_cov = 1 / 2 * np.array([[0.1, 0], [0, 0.1]])
                noise_var = 1e-4

                self.prior_mean = prior_mean[:, np.newaxis]  # column vector of shape (1, d)
                self.prior_cov = prior_cov  # matrix of shape (d, d)
                # We initialise the prior distribution over the parameters using the given mean and covariance matrix
                # In the formulas above this corresponds to m_0 (prior_mean) and S_0 (prior_cov)
                self.prior = multivariate_normal(prior_mean, prior_cov)

                # We also know the variance of the noise
                self.noise_var = noise_var  # single float value
                self.noise_precision = 1 / noise_var

                # Before performing any inference the parameter posterior equals the parameter prior
                self.param_posterior = self.prior
                # Accordingly, the posterior mean and covariance equal the prior mean and variance
                self.post_mean = self.prior_mean  # corresponds to m_N in formulas
                self.post_cov = self.prior_cov  # corresponds to S_N in formulas

        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.state_next_history = deque(maxlen=100)
        self.rewards_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.history = deque(maxlen=100)
        self.frame_count = 0



    def name(self):
        return self.name

    def update_posterior(self, features: np.ndarray, targets: np.ndarray):
        """
        Update the posterior distribution given new features and targets

        Args:
            features: numpy array of features
            targets: numpy array of targets
        """
        # Reshape targets to allow correct matrix multiplication
        # Input shape is (N,) but we need (N, 1)
        targets = targets[:, np.newaxis]

        # Compute the design matrix, shape (N, 2)
        data = features
        if self.log_normal:
            #targets = self.intercept - targets
            features = np.where(features == 0, 1e-4, features)
            data = np.log(features)
        design_matrix = self.compute_design_matrix(data)

        # Update the covariance matrix, shape (2, 2)
        design_matrix_dot_product = design_matrix.T.dot(design_matrix)
        inv_prior_cov = np.linalg.inv(self.prior_cov)
        self.post_cov = np.linalg.inv(inv_prior_cov + self.noise_precision * design_matrix_dot_product)

        # Update the mean, shape (2, 1)
        self.post_mean = self.post_cov.dot(
            inv_prior_cov.dot(self.prior_mean) +
            self.noise_precision * design_matrix.T.dot(targets))

        # Update the posterior distribution
        self.param_posterior = multivariate_normal(self.post_mean.flatten(), self.post_cov)


    def compute_design_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the design matrix. To keep things simple we use simple linear
        regression and add the value phi_0 = 1 to our input data.

        Args:
            features: numpy array of features
        Returns:
            design_matrix: numpy array of transformed features
        """
        n_samples = len(features)
        phi_0 = np.ones(n_samples)
        design_matrix = np.stack((phi_0, features), axis=1)
        return design_matrix

    def predict(self, features: np.ndarray):
        """
        Compute predictive posterior given new datapoint

        Args:
            features: 1d numpy array of features
        Returns:
            pred_posterior: predictive posterior distribution
        """
        design_matrix = self.compute_design_matrix(features)

        pred_mean = design_matrix.dot(self.post_mean)
        pred_cov = design_matrix.dot(self.post_cov.dot(design_matrix.T)) + self.noise_var

        pred_posterior = univariate_normal(loc=pred_mean.flatten(), scale=pred_cov ** 0.5).rvs()
        # Should have scale around -10 to -5 (for result close to 0), up to 3~4 (for result 20~55).
        # With intercept at 18, means value between close to 18, and -2 to - a lot.
        if self.log_normal:
            pred_posterior = self.intercept - np.exp(pred_posterior)
        return pred_posterior

    def logistic_function(self, intercept, coefficient, x):
        return (1 / (1 + np.exp(-(intercept + coefficient * x)))) * x

    def get_logistic_optimum(self, model):
        intercept = model.intercept_[0]
        coefficient = model.coef_[0][0]
        min_price = min(self.prices_possible)
        max_price = max(self.prices_possible)
        x = np.linspace (min_price, max_price, 1000)
        y = self.logistic_function(intercept, coefficient, x)
        best_index = np.argmax(y)
        return x[best_index]

    def get_prediction(self, price_offer):
        if self.TYPE == "linear":
            prediction = self.predict([price_offer[0]])
            return prediction
        elif self.TYPE == "logistic":
            total_seats = 0
            for i in range(self.seats_available):
                total_seats += self.model[i].predict_proba(price_offer[i].reshape((1, -1)))[0][1]
            return total_seats

    def get_action(self, state: []):
        if self.TYPE == "linear":
            top_revenue = 0
            top_price = 0
            top_seats = 0
            idx = 0
            for price in self.prices_possible:
                price_offer = np.full((self.seats_available), price)
                seats = self.get_prediction(price_offer)
                revenue = seats * price
                if revenue > top_revenue:
                    top_revenue = revenue
                    top_price = price
                    top_seats = seats
                idx += 1
            self.prices = np.full((self.seats_available), top_price)
            predicted_seats = top_seats
            return self.prices, predicted_seats
        elif self.TYPE == "logistic":
            for seat_index in range(self.seats_available):
                best_price = self.get_logistic_optimum(self.model[seat_index])
                self.prices[seat_index] = best_price

        predicted_seats = self.get_prediction(self.prices)
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

    def update_during_booking(self, booking_index, total_customers, action,
                              start_state, prediction, current_revenue, current_state):
        return action, False
        if self.frame_count < 5:
            return action, False
        if self.TYPE == "logistic":
            seats_sold = sum(current_state)
            total_expected_seats_sold = self.get_prediction(action)
            current_percentage = float(booking_index) / float(total_customers)
            current_expected_seats_sold = current_percentage * total_expected_seats_sold
            if seats_sold > 1.2 * current_expected_seats_sold:
                # Too high above expectation, so price too low
                action = action * 1.1
            elif seats_sold < 0.75 * current_expected_seats_sold:
                # Too low under expectation, so price too high
                action = action * 0.9
            return action, False
        else:
            return action, False

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered):

        self.state_history.append(np.copy(start_state))
        self.prediction_history.append(prediction)
        self.rewards_history.append(round_revenue)
        if self.TYPE == "linear":
            self.action_history.append(np.copy(action[0]))
            seats_sold = sum(new_state)
            self.state_next_history.append(seats_sold)

            self.frame_count += 1
            self.update_posterior(np.array(self.action_history), np.array(self.state_next_history))
        else:
            self.action_history.append(np.copy(action))
            self.state_next_history.append(new_state)
            for i in range(len(self.prices)):
                seat_prices = np.array(self.get_prices_for_seat(i)).reshape((-1, 1))
                seat_sales = self.get_sales_for_seat(i)
                unique_prices = np.unique(seat_prices)
                if len(np.unique(seat_sales)) == 1:
                    continue
                    #for price_used in unique_prices:
                    #    idx = np.where(self.prices_possible == price_used)
                    #    self.skip_offer_price[i][idx] = True
                else:
                    #for price_used in unique_prices:
                    #    idx = np.where(self.prices_possible == price_used)
                    #    self.skip_offer_price[i][idx] = False
                    self.model[i].fit(seat_prices, seat_sales)


