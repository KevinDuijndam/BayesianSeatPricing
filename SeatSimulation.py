import numpy as np
import Customer
import sys

customerdt = np.dtype([('wtp', np.float), ('nr_in_group', np.int), ('price_bought', np.float), ('bought_seat', np.bool)])

COMPLEX_SEATING = True

class SeatSimulationFlight:
    def __init__(self, total_nr_customers: int, wtp_mu: float, wtp_sigma: float, wtp_scale: float,
                 seats_available: int, prices_offered: np.ndarray):
        self.total_nr_customers = total_nr_customers
        self.wtp_mu = wtp_mu
        self.wtp_sigma = wtp_sigma
        self.wtp_scale = wtp_scale
        self.total_seats = seats_available
        self.seats_available = seats_available
        self.seats_sold = np.full(seats_available, False, dtype=bool)
        self.prices_offered = np.copy(prices_offered)
        self.lowest_price = np.min(self.prices_offered)
        customer_list = self.create_customer_list()
        self.nr_bookings = len(customer_list)
        self.customers = np.empty(len(customer_list), dtype=customerdt)
        i = 0
        for customer in customer_list:
            self.customers[i] = (customer.wtp, customer.nr_in_group, customer.price_bought, customer.bought_seat)
            i += 1

    def create_customer_list(self):
        nr_customers_divided = 0
        customer_list = list()
        #wtp_scale = self.wtp_scale * 0.8
        while nr_customers_divided < self.total_nr_customers:
            booking_size = 1 + np.random.binomial(8, 0.2)
            if booking_size > self.total_nr_customers - nr_customers_divided:
                pass
            wtp = np.random.lognormal(self.wtp_mu, self.wtp_sigma) * self.wtp_scale
            customer_to_add = Customer.Customer(wtp, booking_size, 0.0, False)
            customer_list.append(customer_to_add)
            nr_customers_divided += booking_size
            #wtp_scale = wtp_scale * 1.05
        return customer_list

    def update_price_offer(self, price_offer: np.ndarray):
        self.prices_offered = price_offer
        self.lowest_price = np.min(self.prices_offered)

    def adjacent_seats_available(self, seats_necessary: int):
        available_together = 0
        idx = 0
        for seat_available in self.seats_sold:
            if seat_available:
                available_together += 1
                if available_together >= seats_necessary:
                    return True
            elif idx % 9 == 0:
                available_together = 0
            else:
                available_together = 0
            idx += 1
        return False

    def sell_seat2(self, customer: int):
        resulting_price = 0
        seats_sold = 0
        if self.lowest_price > self.customers[customer]['wtp']:
            return True, resulting_price, seats_sold
        # potential_seats = self.prices_offered[self.prices_offered < self.customers[customer]['wtp']]
        # if len(potential_seats) < 1:
        #     return resulting_price, seats_sold
        if (self.customers[customer]['bought_seat'] == False and
                self.customers[customer]['nr_in_group'] < self.seats_available):
            seats_available_together = 0
            seat_factor = 4 if self.customers[customer]['nr_in_group'] == 1 else 1
            for i in range(self.total_seats):
                if COMPLEX_SEATING:
                    seat_price = self.prices_offered[i]
                    index_in_row = i - ( (i // 6) * 6)
                    if index_in_row == 1 or index_in_row == 4:
                        # Multiply the seat price instead of dividing the customers' WTP, has same effect
                        seat_price *= seat_factor
                    if (seat_price < self.customers[customer]['wtp'] and
                            not self.seats_sold[i]):
                        seats_available_together += 1
                    elif i % 6 == 0:
                        # New row in the aircraft
                        seats_available_together = 0
                    else:
                        # Looped through a seat that is either sold or is too expensive
                        seats_available_together = 0
                else:
                    if (self.prices_offered[i] < self.customers[customer]['wtp'] and
                            not self.seats_sold[i]):
                        seats_available_together += 1
                    else:
                        seats_available_together = 0
                if seats_available_together == self.customers[customer]['nr_in_group']:
                    self.seats_sold[i-(seats_available_together-1) : i+1] = True
                    resulting_price = sum(self.prices_offered[i-(seats_available_together-1) : i+1])
                    price_selected = min(self.prices_offered[i-(seats_available_together-1) : i+1])
                    self.prices_offered[i-(seats_available_together-1):i+1] = 99999
                    if price_selected == self.lowest_price:
                        self.lowest_price = np.min(self.prices_offered)
                    self.customers[customer]['bought_seat'] = True
                    self.customers[customer]['price_bought'] = self.prices_offered[i]
                    seats_sold = self.customers[customer]['nr_in_group']
                    self.seats_available -= self.customers[customer]['nr_in_group']
                    break
            return True, resulting_price, seats_sold
        else:
            return False, resulting_price, seats_sold

    def get_nr_customers(self, customer_index: int):
        return self.customers[customer_index]['nr_in_group']

    def sell_seat(self, price_offered: float):
        resulting_price = 0.0
        seats_sold = 0

        eligible_customers = self.customers[(self.customers['bought_seat'] == False) &
                                            (self.customers['nr_in_group'] < self.seats_available) &
                                            (self.customers['wtp'] > price_offered)]
        if len(eligible_customers) > 0:
            eligible_customers[0]['bought_seat'] = True
            eligible_customers[0]['price_bought'] = price_offered
            resulting_price = price_offered * eligible_customers[0]['nr_in_group']
            seats_sold = eligible_customers[0]['nr_in_group']
            self.seats_available -= eligible_customers[0]['nr_in_group']
        return resulting_price, seats_sold

    def theoretical_max(self):
        ordered_customers = self.customers[np.argsort(-self.customers['wtp'])]
        idx = 0
        max_seats_available = self.seats_available
        total_revenue = 0
        total_seats_sold = 0
        while max_seats_available > 0 and idx < len(ordered_customers):
            customers_in_group = ordered_customers[idx]['nr_in_group']
            if customers_in_group <= max_seats_available:
                max_seats_available -= customers_in_group
                total_revenue += customers_in_group * ordered_customers[idx]['wtp']
                total_seats_sold += customers_in_group
            idx += 1
        return total_revenue, total_seats_sold


