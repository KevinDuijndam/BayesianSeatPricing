#import BayesPred
import BayesianApproach
#import Customer
#import DeepPred
import ExpPred
import LinPred
import ValueIteration
#import NNLearning
import ProbaPrediction
import SeatSimulation
import QLearning
#import PPO
import DDPG
import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
#from collections import deque
from math import trunc
#import matplotlib.pyplot as plt
#from IPython.display import clear_output
import copy

#import tensorflow as tf
#from tensorflow import keras

import warnings

ENABLE_UPDATE_DURING_SALES = True
ENABLE_MULTIPLE_SALES_WINDOWS = False
OUTPUT_PER_FIVE_FLIGHTS = True


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def run_simulation_multiple_flights(simulation_rounds: int, nr_different_flights: int, nr_flights_per_group: int,
                                    seats_available: int, prices_offered: [], models: []):
    total_revenue = 0
    total_seats_sold = 0
    print_count = []
    model_performance = []
    predictions = []
    results = []
    print("Text; Round; Text; Model name; Text; Revenue; Text; Max Revenue; Text; Share of max revenue")
    for _ in range(len(models)):
        model_performance.append([])
        print_count.append(0)

    flight_group_parameters = list()
    for _ in range(nr_different_flights):
        customer_max = np.random.randint(80, 120)
        customer_probability = np.random.uniform(0.7, 0.95)
        customer_wtp_sigma = np.random.normal(0.35, 0.2)
        customer_wtp_sigma = customer_wtp_sigma if customer_wtp_sigma >= 0 else 0
        customer_wtp_scale = np.random.poisson(40)
        flight_group_parameters.append([customer_max, customer_probability, customer_wtp_sigma, customer_wtp_scale])

    for simulation_count in range(simulation_rounds):
        flight_groups = list()
        for flight_group_nr in range(nr_different_flights):
            flights = list()
            customer_max = flight_group_parameters[flight_group_nr][0]
            customer_probability = flight_group_parameters[flight_group_nr][1]
            customer_wtp_sigma = flight_group_parameters[flight_group_nr][2]
            customer_wtp_scale = flight_group_parameters[flight_group_nr][3]
            for _ in range(nr_flights_per_group):
                total_customers = np.random.binomial(customer_max, customer_probability)
                #total_customers = 100
                flight_simulation = SeatSimulation.SeatSimulationFlight(total_nr_customers=total_customers,
                                                                        wtp_mu=0, wtp_sigma=customer_wtp_sigma,
                                                                        wtp_scale=customer_wtp_scale,
                                                                        seats_available=seats_available,
                                                                        prices_offered=prices_offered)
                flights.append(flight_simulation)
            flight_groups.append(flights)

        for model_idx in range(len(models)):
            model = models[model_idx]
            simulation_seats_sold = 0
            simulation_revenue = 0.0
            simulation_max_seats_sold = 0
            simulation_max_revenue = 0.0
            number_of_flights_offered = 0
            prices_offers_used = []
            #for flight_idx in range(nr_flights_per_group):
            #    for flight_type_idx in range(nr_different_flights):
            for flight_type_idx in range(nr_different_flights):
                for flight_idx in range(nr_flights_per_group):
                    number_of_flights_offered += 1
                    model.initialise_for_flight_type(flight_type_idx)
                    flight = flight_groups[flight_type_idx][flight_idx]
                    flight_max_revenue, flight_max_seats = flight.theoretical_max()
                    simulation_max_revenue += flight_max_revenue
                    simulation_max_seats_sold += flight_max_seats
                    flight_copy = copy.deepcopy(flight)
                    state = np.copy(flight_copy.seats_sold)
                    action, prediction = model.get_action(state=state, flight_type=flight_type_idx)
                    predictions.append(prediction)
                    price_offer = np.copy(action)
                    new_action = np.copy(action)
                    prices_offers_used.append(price_offer[0])
                    flight_copy.update_price_offer(price_offer)
                    flight_revenue = 0
                    customers_offered = 0
                    # Booking window
                    for i in range(flight_copy.nr_bookings):
                        use_data, round_revenue, round_seats_sold = flight_copy.sell_seat2(i)
                        customers_offered += flight_copy.get_nr_customers(i)
                        simulation_seats_sold += round_seats_sold
                        simulation_revenue += round_revenue
                        flight_revenue += round_revenue
                        if ENABLE_UPDATE_DURING_SALES:
                            current_state = np.copy(flight_copy.seats_sold)
                            total_customers = flight_copy.total_nr_customers
                            new_action, price_updated = model.update_during_booking(customers_offered, total_customers,
                                                                                    new_action, state, prediction,
                                                                                    flight_revenue, current_state,
                                                                                    flight_type_idx)
                            if price_updated:
                                price_offer = np.copy(new_action)
                                flight_copy.update_price_offer(price_offer)
                                state = np.copy(flight_copy.seats_sold)
                                customers_offered = 0
                    if ENABLE_MULTIPLE_SALES_WINDOWS:
                        # Reservation window, reset pricing strategy
                        price_offer = np.copy(action)
                        flight_copy.update_price_offer(price_offer)
                        customers_offered = 0
                        for i in range(flight_copy.nr_bookings):
                            chance_check_seats = np.random.uniform(0, 1)
                            if chance_check_seats > 0.8:
                                use_data, round_revenue, round_seats_sold = flight_copy.sell_seat2(i)
                                simulation_seats_sold += round_seats_sold
                                simulation_revenue += round_revenue
                                flight_revenue += round_revenue
                                customers_offered += flight_copy.get_nr_customers(i)
                        # Check-in window
                        for i in range(flight_copy.nr_bookings):
                            use_data, round_revenue, round_seats_sold = flight_copy.sell_seat2(i)
                            simulation_seats_sold += round_seats_sold
                            simulation_revenue += round_revenue
                            flight_revenue += round_revenue
                            customers_offered += flight_copy.get_nr_customers(i)
                    end_state = np.copy(flight_copy.seats_sold)
                    result = sum(flight_copy.seats_sold)
                    results.append(result)
                    model.process_data(action, state, prediction, flight_revenue, end_state,
                                       customers_offered, flight_type_idx)
                    if OUTPUT_PER_FIVE_FLIGHTS:
                        if number_of_flights_offered % 100 == 0:
                            total_revenue += simulation_revenue
                            total_seats_sold += simulation_seats_sold
                            percentage_of_max_revenue = 100 * (simulation_revenue / simulation_max_revenue)
                            model_performance[model_idx].append(simulation_revenue)
                            average_difference = np.mean(np.array(results) - np.array(predictions))
                            average_price = np.mean(prices_offers_used)
                            results = []
                            predictions = []

                            print("Simulation;", print_count[model_idx], ";name;", model.name, ";revenue;",
                                  trunc(simulation_revenue),
                                  ";max revenue;", trunc(simulation_max_revenue), ";share of max;",
                                  trunc(percentage_of_max_revenue),
                                  "; average price;", trunc(average_price))
                            print_count[model_idx] += 1
                            prices_offers_used = []
                            simulation_seats_sold = 0
                            simulation_revenue = 0.0
                            simulation_max_seats_sold = 0
                            simulation_max_revenue = 0.0

            if not OUTPUT_PER_FIVE_FLIGHTS:
                total_revenue += simulation_revenue
                total_seats_sold += simulation_seats_sold
                percentage_of_max_revenue = 100 * (simulation_revenue / simulation_max_revenue)
                model_performance[model_idx].append(simulation_revenue)
                average_difference = np.mean(np.array(results) - np.array(predictions))
                results = []
                predictions = []

                print("Simulation;", simulation_count, ";name;", model.name, ";revenue;", trunc(simulation_revenue),
                      ";max revenue;", trunc(simulation_max_revenue), ";share of max;", trunc(percentage_of_max_revenue))


def run_simulation_flight(simulation_rounds: int, nr_flights: int, seats_available: int,
                          prices_offered: [], models: []):
    total_revenue = 0
    total_seats_sold = 0
    model_performance = []
    predictions = []
    results = []
    for _ in range(len(models)):
        model_performance.append([])

    for simulation_count in range(simulation_rounds):
        flights = list()
        for _ in range(nr_flights):
            total_customers = np.random.binomial(100, 0.85)
            flight_simulation = SeatSimulation.SeatSimulationFlight(total_nr_customers=total_customers,
                                                                    wtp_mu=0, wtp_sigma=0.35, wtp_scale=40,
                                                                    seats_available=seats_available,
                                                                    prices_offered=prices_offered)
            flights.append(flight_simulation)

        for model_idx in range(len(models)):
            model = models[model_idx]
            simulation_seats_sold = 0
            simulation_revenue = 0.0
            simulation_max_seats_sold = 0
            simulation_max_revenue = 0.0
            for flight in flights:
                flight_max_revenue, flight_max_seats = flight.theoretical_max()
                simulation_max_revenue += flight_max_revenue
                simulation_max_seats_sold += flight_max_seats
                flight_copy = copy.deepcopy(flight)
                state = np.copy(flight_copy.seats_sold)
                #if simulation_count % 249 == 0:
                #    print("starting simulation 249")
                action, prediction = model.get_action(state=state)
                predictions.append(prediction)
                price_offer = np.copy(action)
                flight_copy.update_price_offer(price_offer)
                flight_revenue = 0
                for i in range(flight_copy.nr_bookings):
                    use_data, round_revenue, round_seats_sold = flight_copy.sell_seat2(i)
                    simulation_seats_sold += round_seats_sold
                    simulation_revenue += round_revenue
                    flight_revenue += round_revenue
                end_state = np.copy(flight_copy.seats_sold)
                result = sum(flight_copy.seats_sold)
                results.append(result)
                model.process_data(action, state, prediction, flight_revenue, end_state)
            total_revenue += simulation_revenue
            total_seats_sold += simulation_seats_sold
            percentage_of_max_revenue = 100 * (simulation_revenue / simulation_max_revenue)
            model_performance[model_idx].append(simulation_revenue)
            average_difference = np.mean(np.array(results) - np.array(predictions))
            results = []
            predictions = []

            print("Simulation;", simulation_count, ";name;", model.name, ";revenue;", trunc(simulation_revenue),
                  ";max revenue;", trunc(simulation_max_revenue), ";share of max;", trunc(percentage_of_max_revenue))
            #print("Average prediction difference: ", average_difference)


def run_simulation_general(simulation_rounds: int, nr_flights: int, seats_available:int,
                           prices_offered: [], models: []):
    total_revenue = 0
    total_seats_sold = 0
    model_performance = []
    for _ in range(len(models)):
        model_performance.append([])
    #plt.ion()
    for simulation_count in range(simulation_rounds):
        # Prepare input for simulation
        flights = list()
        for _ in range(nr_flights):
            total_customers = np.random.binomial(100, 0.85)
            flight_simulation = SeatSimulation.SeatSimulationFlight(total_nr_customers=total_customers,
                                                                    wtp_mu=0, wtp_sigma=0.5, wtp_scale=20,
                                                                    seats_available=seats_available,
                                                                    prices_offered=prices_offered)
            flights.append(flight_simulation)

        # Run test over every flight within simulation
        for model_idx in range(len(models)):
            model = models[model_idx]
            simulation_seats_sold = 0
            simulation_revenue = 0
            simulation_max_seats_sold = 0
            simulation_max_revenue = 0
            for flight in flights:
                flight_max_revenue, flight_max_seats = flight.theoretical_max()
                simulation_max_revenue += flight_max_revenue
                simulation_max_seats_sold += flight_max_seats
                flight_test = copy.deepcopy(flight)
                for i in range(flight_test.nr_bookings):
                    start_state = np.copy(flight_test.seats_sold)
                    action, prediction = model.get_action(state=flight_test.seats_sold)
                    price_offer = np.copy(action)
                    flight_test.update_price_offer(price_offer)
                    use_data, round_revenue, round_seats_sold = flight_test.sell_seat2(i)
                    simulation_seats_sold += round_seats_sold
                    simulation_revenue += round_revenue
                    new_state = np.copy(flight_test.seats_sold)
                    if use_data:
                        model.process_data(action, start_state, prediction, round_revenue, new_state)
            total_revenue += simulation_revenue
            total_seats_sold += simulation_seats_sold
            percentage_of_max_revenue = 100 * (simulation_revenue / simulation_max_revenue)
            model_performance[model_idx].append(simulation_revenue)
            #for performance in model_performance:
            #    plt.plot(performance)
            #clear_output()
            #plt.show()
            print("Simulation round;", simulation_count, ";name;", model.name, ";revenue;", trunc(simulation_revenue),
                  ";max revenue;", trunc(simulation_max_revenue), ";share of max;", trunc(percentage_of_max_revenue))

@timeit
def start_model():
    prices = np.linspace(1, 250, 10)
    prices_test = [prices]
    np.random.seed(42)
    seats_available = 18
    simulation_rounds = 75
    nr_flights = 6
    nr_different_flights = 50
    #prices_test = [[5, 7, 10, 12, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 47, 50, 53, 55, 58, 60, 63, 65, 70]]

    for price in prices_test:
        models = []
        #qlearning = QLearning.QLearning(epsilon=0.01, lr=0.995, gamma=0.95,
        #                                seats_available=seats_available, price_levels=price)
        #models.append(qlearning)
        #optimiser = keras.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)
        #loss_function = keras.losses.Huber()
        #model_NN = NNLearning.NNLearning(gamma=0.9, epsilon=0.01, batch_size=64,
        #                              max_steps=100, seats_available=seats_available, price_levels=price,
        #                              optimizer=optimiser, loss_function=loss_function)
        #models.append(model_NN)
        #model_PPO = PPO.PPO(seats_available=seats_available)
        #models.append(model_PPO)
        #model_DDPG = DDPG.DDPG(gamma=0.9, batch_size=1024, seats_available=seats_available, price_levels=prices,
        #                       nr_flight_types=nr_different_flights)
        #models.append(model_DDPG)
        #model_lin = LinPred.LinPred(type="linear", seats_available=seats_available, epsilon=0.05,
        #                            prices_offered=price)
        #models.append(model_lin)
        #model_proba = ProbaPrediction.ProbaPrediction(type="Binomial", seats_available=seats_available,
        #                                              prices_offered=price, nr_flight_types=nr_different_flights)
        #models.append(model_proba)
        #model_proba_bayes = ProbaPrediction.ProbaPrediction(type="Binomial Bayesian", seats_available=seats_available,
        #                                              prices_offered=price, nr_flight_types=nr_different_flights)
        #models.append(model_proba_bayes)
        #model_proba_ucb = ProbaPrediction.ProbaPrediction(type="Binomial Bayesian UCB", seats_available=seats_available,
        #                                              prices_offered=price, nr_flight_types=nr_different_flights)
        #models.append(model_proba_ucb)
        #model_lin_pos = LinPred.LinPred(type="linear_positive", seats_available=seats_available, epsilon=0.05,
        #                            prices_offered=price)
        #models.append(model_lin_pos)
        #model_log = LinPred.LinPred(type="logistic", seats_available=seats_available, epsilon=0.05,
        #                            prices_offered=price)
        #models.append(model_log)
        #model_log_pos = LinPred.LinPred(type="logistic_positive", seats_available=seats_available, epsilon=0.05,
        #                            prices_offered=price)
        #models.append(model_log_pos)

        #model_bayes = BayesPred.BayesPred(type="logistic", seats_available=seats_available, epsilon=0.05,
        #                                  prices_offered=price)
        #models.append(model_bayes)
        #model_bays_simple = BayesianApproach.BayesianApproach(type="linear", seats_available=seats_available,
        #                                                      epsilon=0.05, prices_offered=price,
        #                                                      nr_flight_types=nr_different_flights)
        #models.append(model_bays_simple)
        #model_double_lin = LinPred.LinPred(type="double_linear", seats_available=seats_available, epsilon=0.05, prices_offered=price)
        #models.append(model_double_lin)
        #model_poly = LinPred.LinPred(type="polynomial", seats_available=seats_available, epsilon=0.05,
        #                                   prices_offered=price)
        #models.append(model_poly)
        #model_deeppred = DeepPred.DeepPred(epsilon=0.05, batch_size=32, max_steps=10,
        #                                   seats_available=seats_available, price_levels=price)
        #models.append(model_deeppred)
        #model_exp = ExpPred.ExpPred(type="exp", seats_available=seats_available, epsilon=0.05,
        #                            prices_offered=price, nr_flight_types=nr_different_flights)
        #models.append(model_exp)
        model_value_iteration = ValueIteration.ValueIteration(type="Value Iteration", seats_available=seats_available,
                                                              epsilon=0.05, prices_offered=price,
                                                              nr_flight_types=nr_different_flights)
        models.append(model_value_iteration)
        run_simulation_multiple_flights(simulation_rounds, nr_different_flights, nr_flights,
                                        seats_available, price, models)
        #run_simulation_flight(simulation_rounds, nr_flights, seats_available, price, models)
        #run_simulation_general(simulation_rounds, nr_flights, seats_available, price, models)


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    warnings.filterwarnings("ignore", category=UserWarning)
    start_model()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    #print(s.getvalue())
