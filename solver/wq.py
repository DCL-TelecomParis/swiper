from fractions import Fraction
from typing import List, Union

from solver.wr import WeightRestriction


class WeightQualification:

    """Represents an instance of the Weight Qualification problem."""

    def __init__(self,
                 weights: List[Union[Fraction, float, int]],
                 tw: Union[Fraction, float],
                 tn: Union[Fraction, float]):
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
        return f"WeightQualification < " \
               f"n={self.n}, weights=[{' '.join(map(str, self.weights))}], tw={self.tw}, tn={self.tn}" \
               f" >"

    def __repr__(self):
        return str(self)

    def to_wr(self):
        return WeightRestriction(self.weights, 1 - self.tw, 1 - self.tn)


