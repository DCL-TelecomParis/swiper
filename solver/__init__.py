import logging
from fractions import Fraction
from math import floor, ceil
from typing import List, Union

from solver.knapsack import knapsack, knapsack_upper_bound
from solver.wq import WeightQualification
from solver.wr import WeightRestriction


def solve(inst: Union[WeightRestriction, WeightQualification], linear: bool, no_jit: bool, verify: bool) -> List[int]:
    """
    Solve the Weight Restriction or Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    if isinstance(inst, WeightQualification):
        return wq_solve(inst, linear, no_jit, verify)
    elif isinstance(inst, WeightRestriction):
        return wr_solve(inst, linear, no_jit, verify)
    else:
        raise ValueError(f"Unknown instance type {type(inst)}")


def wq_solve(inst: WeightQualification, linear: bool, no_jit: bool, verify: bool) -> List[int]:
    """
    Solve the Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    return wr_solve(inst.to_wr(), linear, no_jit, verify)


def wr_upper_bound_s(inst: WeightRestriction) -> Fraction:
    # s := \frac{\alpha_n (1 - \alpha_w)n}{(\alpha_n - \alpha_w)W}
    assert isinstance(inst.tw, Fraction)
    assert isinstance(inst.tn, Fraction)
    res = inst.tn * (1 - inst.tw) * inst.n / ((inst.tn - inst.tw) * inst.total_weight)
    assert isinstance(res, Fraction)
    return res


def wr_solution_upper_bound(inst: WeightRestriction) -> int:
    # \left\lceil \frac{\alpha_w(1 - \alpha_w)}{\alpha_n - \alpha_w} n \right\rceil
    return ceil(inst.tw * (1 - inst.tw) / (inst.tn - inst.tw) * inst.n)


def allocate(inst: WeightRestriction, s: Fraction, shift: Fraction) -> List[int]:
    return [floor(inst.weights[i] * s + shift) for i in range(inst.n)]


def wr_solve(inst: WeightRestriction, linear: bool, no_jit: bool, verify: bool) -> List[int]:
    assert all(isinstance(inst.weights[i], int) for i in range(inst.n))

    shift = inst.tw
    eps = Fraction(1, max(inst.weights))

    # this is the largest integer smaller than inst.threshold_weight
    threshold_weight_non_strict = ceil(inst.threshold_weight) - 1

    s_high = wr_upper_bound_s(inst)
    s_low = 0

    if verify:
        logging.debug("Verifying the upper bound...")
        assert wr_solution_valid(inst, allocate(inst, s_high, shift), no_jit), "s* upper bound is violated"

    logging.debug("Binary search for s*...")

    # First, use knapsack upper bound instead of actual knapsack to speed up the process
    logging.debug("Using knapsack upper bound to estimate s*...")
    steps = 0
    while s_high - s_low >= eps:
        steps += 1

        s_mid = (s_high + s_low) / 2
        t_mid = allocate(inst, s_mid, shift)

        if knapsack_upper_bound(inst.weights, t_mid, threshold_weight_non_strict) < inst.tn * sum(t_mid):
            s_high = s_mid
        else:
            s_low = s_mid

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("s* <= %s", s_high)

    if linear:
        logging.debug("Skipping further optimization of s* because linear mode is enabled.")
    else:
        # Use actual knapsack to find a local minimum
        # Using a special type of accelerated binary search that is fast with a good initial estimate
        logging.debug("Using knapsack to find s* precisely...")
        speed = eps
        s_low = 0

        steps = 0
        while s_high - s_low >= eps:
            steps += 1

            if 2 * speed < s_high - s_low:
                # Move from s_high with an acceleration
                s_mid = s_high - speed
                speed *= 2
            else:
                # Fall back to regular binary search
                s_mid = (s_high + s_low) / 2

            t_mid = allocate(inst, s_mid, shift)
            sum_t_mid = sum(t_mid)

            knapsack_res = knapsack(inst.weights, t_mid, threshold_weight_non_strict,
                                    upper_bound=floor(sum_t_mid * inst.tn) + 1, no_jit=no_jit)
            if knapsack_res < inst.tn * sum_t_mid:
                s_high = s_mid
            else:
                s_low = s_mid

        logging.debug(f"Finished in {steps} steps.")
        logging.debug("s* = %s", s_high)

    t_low = allocate(inst, s_low, shift)
    t_high = allocate(inst, s_high, shift)

    border_set = [i for i in range(inst.n) if t_low[i] != t_high[i]]
    assert all(t_low[i] == t_high[i] - 1 for i in border_set)

    if verify:
        logging.debug("Verifying the intermediate solution...")
        assert wr_solution_valid(inst, t_high, no_jit), "s* is too low"

    # do binary search to determine how many parties in the border set should be rounded up
    k_low = 0
    k_high = len(border_set)

    logging.debug("Binary search for optimal k*...")

    # Again, first, use knapsack upper bound instead of actual knapsack to speed up the process
    logging.debug("Using knapsack upper bound to estimate k*...")
    steps = 0
    while k_high - k_low > 1:
        steps += 1

        k_mid = (k_high + k_low) // 2
        t_mid = [t_low[i] if i in border_set[k_mid:] else t_high[i] for i in range(inst.n)]

        if knapsack_upper_bound(inst.weights, t_mid, threshold_weight_non_strict) < inst.tn * sum(t_mid):
            k_high = k_mid
        else:
            k_low = k_mid

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("k <= %s/%s", k_high, len(border_set))

    if linear:
        logging.debug("Skipping further optimization of k* because linear mode is enabled.")
    else:
        # Use actual knapsack to find a local minimum
        # Using a special type of accelerated binary search that is fast with a good initial estimate
        logging.debug("Using knapsack to find k* precisely...")

        k_low = 0
        speed = 1

        steps = 0
        while k_high - k_low > 1:
            steps += 1

            if 2 * speed < k_high - k_low:
                # Move from k_high with an acceleration
                k_mid = k_high - speed
                speed *= 2
            else:
                # Fall back to regular binary search
                k_mid = (k_high + k_low) // 2

            t_mid = [t_low[i] if i in border_set[k_mid:] else t_high[i] for i in range(inst.n)]
            sum_t_mid = sum(t_mid)

            knapsack_res = knapsack(inst.weights, t_mid, threshold_weight_non_strict,
                                    upper_bound=floor(sum_t_mid * inst.tn) + 1, no_jit=no_jit)
            if knapsack_res < inst.tn * sum_t_mid:
                k_high = k_mid
            else:
                k_low = k_mid

        logging.debug(f"Finished in {steps} steps.")
        logging.debug("k = %s/%s", k_high, len(border_set))

    t_best = [t_low[i] if i in border_set[k_high:] else t_high[i] for i in range(inst.n)]

    if verify:
        logging.debug("Verifying the final solution...")
        assert wr_solution_valid(inst, t_best, no_jit), "k* is too low"
        assert sum(t_best) <= wr_solution_upper_bound(inst), "Upper bound is violated"

    return t_best


def wr_solution_valid(inst: WeightRestriction, t: List[int], no_jit: bool) -> bool:
    knapsack_res = knapsack(inst.weights, t, ceil(inst.threshold_weight) - 1,
                            upper_bound=floor(sum(t) * inst.tn) + 1, no_jit=no_jit)
    return knapsack_res < inst.tn * sum(t)
