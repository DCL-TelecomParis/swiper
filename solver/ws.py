from fractions import Fraction
from typing import List


class WeightSeparation:

    """Represents an instance of the Weight Separation problem."""

    def __init__(self, weights: List[int], alpha: Fraction, beta: Fraction):
        """
        Create a new instance.

        :param weights: list of weights of the parties
        :param alpha: the smaller weighted threshold
        :param beta: the larger weighted threshold
        """
        # Number of parties
        self.n = len(weights)
        # List of weights of the parties
        self.weights = weights
        # Total weight of all parties
        self.total_weight = sum(weights)
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return f"WeightSeparation < " \
               f"n={self.n}, weights=[{' '.join(map(str, self.weights))}], alpha={self.alpha}, beta={self.beta}" \
               f" >"

    def __repr__(self):
        return str(self)
