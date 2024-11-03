from dataclasses import dataclass


@dataclass
class Customer:
    """Class for keeping track of specific Customer status, note, a Customer can be with multiple together."""
    wtp: float
    nr_in_group: int
    price_bought: float
    bought_seat: bool

    def total_sold(self) -> float:
        if self.bought_seat:
            return self.nr_in_group * self.price_bought
        return 0.0
