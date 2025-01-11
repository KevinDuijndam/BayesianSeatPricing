import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from collections import deque
import ConstrainedLinearRegression
from scipy.optimize import Bounds
from clogistic import LogisticRegression as cLogisticRegression


class LinPred:
    def __init__(self, type: str, seats_available: int, epsilon: float, prices_offered: []):
        self.TYPE = type
        self.seats_available = seats_available
        self.prices_possible = prices_offered
        self.epsilon = epsilon
        self.skip_offer_price = []
        self.rng = np.random.default_rng(seed=42)
        for _ in range(self.seats_available):
            nr_prices = len(prices_offered)
            self.skip_offer_price.append(np.full((nr_prices), False))
        self.prices = np.full((self.seats_available), self.rng.choice(self.prices_possible))
        if self.TYPE == "linear":
            self.name = "Linear model" + "_" + str(epsilon)
            self.model = LinearRegression()
            X = np.array([self.prices]).reshape((1, -1))
            y = np.array(18).reshape((1, -1))
            self.model.fit(X, y)        # Random fit just so it's initialised
        elif self.TYPE == "linear_positive":
            self.name = "Linear (positive) model" + "_" + str(epsilon)
            self.model = ConstrainedLinearRegression.ConstrainedLinearRegression()
            X = np.array([np.full(self.seats_available, self.prices_possible[0]),
                          np.full(self.seats_available, self.prices_possible[len(self.prices_possible)-1])])
            y = np.array([self.seats_available, 0])
            #X = np.array([self.prices]).reshape((1, -1))
            #y = np.array(18).reshape((1, -1))
            self.model.fit(X, y)  # Random fit just so it's initialised
        elif self.TYPE == "double_linear":
            self.name = "Double Linear model" + "_" + str(epsilon)
            self.model = []
            self.model.append(LinearRegression())
            self.model.append(LinearRegression())
            mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                    for i in range(self.seats_available)] # Get middle seat idx
            X_1 = np.full((self.seats_available), self.prices_possible[0])[mask]
            #X = np.array([self.prices[0]]).reshape((1, -1))
            y_1 = np.array(self.seats_available).reshape((1, -1))
            X_2 = np.full((self.seats_available), self.prices_possible[len(self.prices_possible)-1])[mask]
            # X = np.array([self.prices[0]]).reshape((1, -1))
            y_2 = np.array(0).reshape((1, -1))
            self.model[0].fit(np.array([X_1, X_2]), np.array([self.seats_available, 0]))  # Random fit just so it's initialised for middle seats
            #X = np.array([self.prices[~np.array(mask)]]).reshape((1, -1))
            X_1 = np.full((self.seats_available), self.prices_possible[0])[~np.array(mask)]
            X_2 = np.full((self.seats_available), self.prices_possible[len(self.prices_possible)-1])[~np.array(mask)]
            #X = np.array([self.prices[0]]).reshape((1, -1))
            self.model[1].fit(np.array([X_1, X_2]), np.array([self.seats_available, 0]))  # Random fit just so it's initialised for window/aisle seats
        elif self.TYPE == "polynomial":
            self.name = "Polynomial model" + "_" + str(epsilon)
            self.model = make_pipeline(PolynomialFeatures(3),LinearRegression())
            X = np.array([self.prices]).reshape((1, -1))
            y = np.array(18).reshape((1, -1))
            self.model.fit(X, y)  # Random fit just so it's initialised
        elif self.TYPE == "logistic":
            self.name = "Logistic model" + "_" + str(epsilon)
            self.model = []     # Create logistic model per seat
            for i in range(self.seats_available):
                self.model.append(LogisticRegression())
                X = np.array([self.prices_possible[0], self.prices_possible[len(self.prices_possible) - 1]]).reshape((-1, 1))
                y = np.array([1, 0])
                self.model[i].fit(X, y)  # Random fit just so it's initialised with 50-50 chance
        elif self.TYPE == "logistic_positive":
            self.name = "Logistic Positive" + "_" + str(epsilon)
            self.model = []
            for i in range(self.seats_available):
                self.model.append(cLogisticRegression())
                bounds = Bounds([-np.inf, 0], [-np.inf, 0])
                X = np.array([self.prices_possible[0], self.prices_possible[len(self.prices_possible) - 1]]).reshape(
                    (-1, 1))
                y = np.array([1, 0])
                self.model[i].fit(X, y, bounds=bounds)  # Random fit just so it's initialised with 50-50 chance

        self.action_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        self.state_next_history = deque(maxlen=100)
        self.rewards_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.history = deque(maxlen=100)
        self.frame_count = 0

    def name(self):
        return self.name

    def initialise_for_flight_type(self, flight_type):
        return

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
        if self.TYPE == "linear" or self.TYPE == "linear_positive" or self.TYPE == "polynomial":
            return self.model.predict([price_offer])
        elif self.TYPE == "double_linear":
            mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                    for i in range(self.seats_available)]  # Get middle seat idx
            price_window_aisle = price_offer[~np.array(mask)].reshape(1, -1)
            price_middle = price_offer[mask].reshape(1, -1)
            seats_middle = self.model[0].predict(price_middle)
            seats_window_aisle = self.model[1].predict(price_window_aisle)
            return seats_middle+seats_window_aisle
        elif self.TYPE == "logistic" or self.TYPE == "logistic_positive":
            total_seats = 0
            for i in range(self.seats_available):
                total_seats += self.model[i].predict_proba(price_offer[i].reshape((1, -1)))[0][1]
            return total_seats
        else:
            return 0

    def get_seat_prediction(self, price_offer, seat_index):
        return self.model[seat_index].predict_proba(price_offer[seat_index].reshape((1, -1)))[0][1]

    def get_action(self, state: [], flight_type):
        if self.TYPE == "linear_positive":
            if self.frame_count < 2:
                self.prices = np.full((self.seats_available), self.rng.choice(self.prices_possible))
                if self.frame_count > 0:
                    while sum(self.action_history[0] - self.prices) == 0:
                        self.prices = np.full((self.seats_available), self.rng.choice(self.prices_possible))
                predicted_seats = self.get_prediction(self.prices)
                return self.prices, predicted_seats

        if self.rng.random() < self.epsilon:
            self.prices = np.full((self.seats_available), self.rng.choice(self.prices_possible))
        else:
            if self.TYPE == "linear" or self.TYPE == "linear_positive" or self.TYPE == "polynomial":
                top_revenue = 0
                top_price = 0
                idx = 0
                for price in self.prices_possible:
                    price_offer = np.full((self.seats_available), price)
                    seats = self.get_prediction(price_offer)
                    revenue = seats * price
                    if revenue > top_revenue:
                        top_revenue = revenue
                        top_price = price
                    idx += 1
                self.prices = np.full((self.seats_available), top_price)
            elif self.TYPE == "double_linear":
                middle_top_revenue = 0
                window_top_revenue = 0
                middle_top_price = 0
                window_top_price = 0
                mask = [(i - ((i // 6) * 6) == 1) or (i - ((i // 6) * 6) == 4)
                        for i in range(self.seats_available)]  # Get middle seat idx
                found_middle_price = False
                found_window_price = False
                for price in self.prices_possible:
                    price_input = np.full(sum(mask), price).reshape(1, -1)
                    middle_seats = self.model[0].predict(price_input)
                    middle_revenue = middle_seats * price
                    if middle_revenue > middle_top_revenue:
                        middle_top_revenue = middle_revenue
                        middle_top_price = price
                        found_middle_price = True
                    price_input = np.full(sum(~np.array(mask)), price).reshape(1, -1)
                    window_seats = self.model[1].predict(price_input)
                    window_revenue = window_seats * price
                    if window_revenue > window_top_revenue:
                        window_top_revenue = window_revenue
                        window_top_price = price
                        found_window_price = True
                if not found_middle_price:
                    middle_top_price = self.rng.choice(self.prices_possible)
                if not found_window_price:
                    window_top_price = self.rng.choice(self.prices_possible)
                self.prices[mask] = middle_top_price
                self.prices[~np.array(mask)] = window_top_price
            elif self.TYPE == "logistic" or self.TYPE == "logistic_positive":
                for seat_index in range(self.seats_available):
                    best_price = self.get_logistic_optimum(self.model[seat_index])
                    self.prices[seat_index] = best_price
            else:
                self.prices = 1
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
                              start_state, prediction, current_revenue, current_state, flight_type):
        if self.TYPE == "linear" or self.TYPE == "polynomial":
            return action, False
        elif self.TYPE == "linear_positive":
            return action, False
            seats_sold = sum(current_state)
            expected_seats_sold_total = self.get_prediction(action)
            current_percentage = float(booking_index) / float(total_customers)
            current_expected_seats_sold = current_percentage * expected_seats_sold_total
            if seats_sold > 1.1 * current_expected_seats_sold:
                # Too high above expectation, so price too low
                action = action * 1.1
                return action, True
            elif seats_sold < 0.9 * current_expected_seats_sold:
                # Too low under expectation, so price too high
                action = action * 0.9
                return action, True
            return action, False
        return action, False

    def process_data(self, action, start_state, prediction, round_revenue, new_state, times_offered, flight_type):
        self.action_history.append(np.copy(action))
        self.state_history.append(np.copy(start_state))
        self.prediction_history.append(prediction)
        self.rewards_history.append(round_revenue)

        self.frame_count += 1
        if self.TYPE == "linear" or self.TYPE == "polynomial":
            seats_sold = sum(new_state)
            self.state_next_history.append(seats_sold)
            self.model.fit(self.action_history, self.state_next_history)
            self.epsilon *= 0.95
        elif self.TYPE == "linear_positive":
            seats_sold = sum(new_state)
            self.state_next_history.append(seats_sold)
            if self.frame_count > 2:
                min_coef = np.repeat(-np.inf, self.action_history[0].shape)
                max_coef = np.repeat(0, self.action_history[0].shape)
                if len(np.unique(self.action_history)) > 1:
                    self.model.fit(self.action_history, self.state_next_history, max_coef=max_coef, min_coef=min_coef)
                    self.epsilon *= 0.95
        elif self.TYPE == "double_linear":
            self.state_next_history.append(new_state)
            if self.frame_count > 1:
                seats_sold_middle = self.get_sales_for_middle()
                prices_middle = self.get_prices_for_middle()
                seats_sold_window = self.get_sales_for_window()
                prices_window = self.get_prices_for_window()
                self.model[0].fit(np.array(prices_middle), seats_sold_middle)
                self.model[1].fit(np.array(prices_window), seats_sold_window)
                self.epsilon *= 0.95
        elif self.TYPE == "logistic":
            self.state_next_history.append(new_state)
            for i in range(len(self.prices)):
                seat_prices = np.array(self.get_prices_for_seat(i)).reshape((-1, 1))
                seat_sales = self.get_sales_for_seat(i)
                unique_prices = np.unique(seat_prices)
                if len(np.unique(seat_sales)) == 1:
                    for price_used in unique_prices:
                        idx = np.where(self.prices_possible == price_used)
                        self.skip_offer_price[i][idx] = True
                else:
                    for price_used in unique_prices:
                        idx = np.where(self.prices_possible == price_used)
                        self.skip_offer_price[i][idx] = False
                    self.model[i].fit(seat_prices, seat_sales)
            self.epsilon *= 0.95
        elif self.TYPE == "logistic_positive":
            self.state_next_history.append(new_state)
            bounds = Bounds([-np.inf, 0], [-np.inf, 0])
            for i in range(len(self.prices)):
                seat_prices = np.array(self.get_prices_for_seat(i)).reshape((-1, 1))
                seat_sales = self.get_sales_for_seat(i)
                unique_prices = np.unique(seat_prices)
                if len(np.unique(seat_sales)) == 1:
                    for price_used in unique_prices:
                        idx = np.where(self.prices_possible == price_used)
                        self.skip_offer_price[i][idx] = True
                else:
                    for price_used in unique_prices:
                        idx = np.where(self.prices_possible == price_used)
                        self.skip_offer_price[i][idx] = False
                    self.model[i].fit(seat_prices, seat_sales, bounds=bounds)
            self.epsilon *= 0.95


