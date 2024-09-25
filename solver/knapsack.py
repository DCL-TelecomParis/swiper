import logging
from fractions import Fraction
from typing import Union, List

import numpy as np
from numba import njit
from numba.core.extending import register_jitable

from solver.util import rev

MAX_INT_64 = np.iinfo(np.int64).max


def knapsack(
        weights: List[int],
        profits: List[int],
        capacity: Union[Fraction, int],
        upper_bound: int,
        no_jit: bool) -> int:
    """
    Solves the given knapsack instance up to the given upper bound on profit.
    Finds the set of items with the highest profit that fits into the knapsack of the given capacity if its profit
    is upper_bound or less.
    Otherwise, finds any set of items that fits into the knapsack and has profit greater than upper_bound.

    If return_set is True, returns the set of items and its profit or upper_bound + 1 if it exceeds the upper bound.
    Otherwise, returns only the profit.

    Running time: O(len(weights) * upper_bound).
    Memory usage: O(len(weights)) if return_set is False, O(len(weights) * upper_bound) otherwise.

    NB: the memory footprint can be reduced from O(len(weights) * upper_bound) to O(len(weights) + upper_bound).
    See: Section 3.3 of "Knapsack problems" by Pisinger, D., & Toth, P. (1998)
    """
    assert len(weights) > 0

    if no_jit:
        return _knapsack_impl(weights, profits, capacity, upper_bound)

    assert isinstance(weights[0], int)

    # Make sure that all integers fit into 64 bits to avoid overflows
    if sum(weights) > MAX_INT_64 or sum(profits) > MAX_INT_64 or capacity > MAX_INT_64:
        logging.warning("Integer overflow while converting fractional weights to integers for JIT compilation. "
                        "Using pure python implementation, which is much slower for large instances.")
        return _knapsack_impl(weights, profits, capacity, upper_bound)

    # Capacity may be a fraction. However, rounding it down does not affect the result.
    capacity = np.int64(int(capacity))

    # Call the JIT-compiled function
    return _knapsack_jit_int(np.array(weights, dtype=np.int64), np.array(profits, dtype=np.int64),
                             capacity, upper_bound)


@register_jitable
def _knapsack_impl(weights, profits, capacity, upper_bound) -> int:

    assert len(weights) == len(profits)

    # If any item fits within the capacity and has profit greater than the upper bound, just return it.
    for i in range(len(weights)):
        if weights[i] <= capacity and profits[i] > upper_bound:
            return profits[i]

    # Ignore items with zero profit
    nonzero_items = [i for i in range(len(weights)) if profits[i] > 0]
    n_nonzero = len(nonzero_items)

    # after i-th iteration of the loop, dp[i] is the minimum weight of a subset of items 0, ..., i
    # with profit at least q.
    dp: List[int] = [0 if q == 0 else MAX_INT_64 for q in range(upper_bound + 2)]

    # Run the dynamic programming algorithm
    for i in range(n_nonzero):
        item = nonzero_items[i]

        # Update the `dp` from right to left to avoid overwriting the values that we still need
        for q in rev(range(upper_bound + 2)):
            if profits[item] >= q:
                dp[q] = min(dp[q], weights[item])
            elif dp[q - profits[item]] != MAX_INT_64:
                dp[q] = min(dp[q], dp[q - profits[item]] + weights[item])

    # Solution is the maximum index of y that does not surpass capacity
    return max([q for q in range(upper_bound + 2) if dp[q] <= capacity])


@njit
def _knapsack_jit_int(
        weights: np.array,
        profits: np.array,
        capacity: np.int64,
        upper_bound: int) -> int:
    return _knapsack_impl(weights, profits, capacity, upper_bound)


def knapsack_bounds(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> (int, int):
    """
    Returns the lower and upper bound for the knapsack solution in quasilinear time.

    NB: these bounds can be computed in linear time using a slightly more complicated algorithm.
    See: Section 3.1 of "Knapsack problems" by Pisinger, D., & Toth, P. (1998)
    """

    n = len(weights)
    assert len(profits) == n

    items = [(profits[i] / weights[i], weights[i], profits[i]) for i in range(n)]
    items.sort(reverse=True, key=lambda x: x[0])

    profit_lower_bound = 0
    profit_upper_bound = 0
    for (_, weight, profit) in items:
        if capacity > weight:
            capacity -= weight
            profit_lower_bound += profit
            profit_upper_bound += profit
        else:
            profit_upper_bound += int(profit * (capacity / weight))
            break

    return profit_lower_bound, profit_upper_bound


def knapsack_upper_bound(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> int:
    return knapsack_bounds(weights, profits, capacity)[1]


def knapsack_lower_bound(
        weights: List[Union[Fraction, float, int]],
        profits: List[int],
        capacity: Union[Fraction, float, int],
) -> int:
    return knapsack_bounds(weights, profits, capacity)[0]
