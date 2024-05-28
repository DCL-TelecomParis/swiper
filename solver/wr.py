from fractions import Fraction
from typing import List


class WeightRestriction:

    """Represents an instance of the Weight Restriction problem."""

    def __init__(self, weights: List[int], tw: Fraction, tn: Fraction):
        """
        Create a new instance.

        :param weights: list of weights of the parties
        :param tw: weighted threshold
        :param tn: nominal threshold
        """
        # Number of parties
        self.n = len(weights)
        # List of weights of the parties
        self.weights = weights
        # Total weight of all parties
        self.total_weight = sum(weights)
        self.tw = tw
        self.tn = tn

    def __str__(self):
        return f"WeightRestriction < " \
               f"n={self.n}, weights=[{' '.join(map(str, self.weights))}], tw={self.tw}, tn={self.tn}" \
               f" >"

    def __repr__(self):
        return str(self)
