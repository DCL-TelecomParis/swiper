import logging
from fractions import Fraction
from math import floor, ceil
from typing import List, Union, Optional

from solver.knapsack import knapsack, knapsack_upper_bound
from solver.wq import WeightQualification
from solver.wr import WeightRestriction
from solver.ws import WeightSeparation


def solve(
        inst: Union[WeightRestriction, WeightQualification, WeightSeparation],
        linear: bool,
        no_jit: bool,
        verify: bool
) -> List[int]:
    """
    Solve the Weight Restriction or Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    if isinstance(inst, WeightQualification):
        return wq_solve(inst, linear, no_jit, verify)
    elif isinstance(inst, WeightRestriction):
        return wr_solve(inst, linear, no_jit, verify)
    elif isinstance(inst, WeightSeparation):
        return ws_solve(inst, linear, no_jit, verify)
    else:
        raise ValueError(f"Unknown instance type {type(inst)}")


def wq_solve(inst: WeightQualification, linear: bool, no_jit: bool, verify: bool) -> List[int]:
    """
    Solve the Weight Qualification problem.
    Returns the status of the solution, the solution itself and the gas expended.
    """

    return wr_solve(inst.to_wr(), linear, no_jit, verify)


def wr_solution_upper_bound(inst: WeightRestriction) -> int:
    # \left\lceil \frac{\alpha_w(1 - \alpha_w)}{\alpha_n - \alpha_w} n \right\rceil
    return ceil(inst.tw * (1 - inst.tw) / (inst.tn - inst.tw) * inst.n)


def ws_solution_upper_bound(inst: WeightSeparation) -> int:
    # \frac{(\alpha + \beta)(1 - \alpha)}{\beta - \alpha} n
    return (inst.alpha + inst.beta) * (1 - inst.alpha) / (inst.beta - inst.alpha) * inst.n


def allocate(weights: List[Fraction], s: Fraction, shift: Fraction) -> List[int]:
    return [floor(w * s + shift) for w in weights]


def round_down(weights, s: Fraction, shift: Fraction, ts: Optional[List[int]], verify: bool) -> Fraction:
    """Returns the smallest s' <= s such that allocate(weights, s', shift) == allocate(weights, s, shift)."""
    if not ts:
        ts = allocate(weights, s, shift)
    res = max((t - shift) / w for w, t in zip(weights, ts))
    if verify:
        # assert any(t > 0 and t == int(t) for t in (w * res + shift for w in weights)), \
        #     "round_down is incorrect (no integers)"
        assert allocate(weights, res, shift) == ts, "round_down is incorrect (too low)"
        assert allocate(weights, res * 0.99999999, shift) != ts, "round_down is incorrect (too high)"
    return res


def round_up(weights, s: Fraction, shift: Fraction, ts: Optional[List[int]], verify: bool) -> Fraction:
    """Returns the smallest s' > s such that allocate(weights, s', shift) != allocate(weights, s, shift)."""
    if not ts:
        ts = allocate(weights, s, shift)
    res = min((t + 1 - shift) / w for w, t in zip(weights, ts))
    if verify:
        assert allocate(weights, res, shift) != ts, "round_up is incorrect (too low)"
        assert allocate(weights, max(s, res * 0.99999999), shift) == ts, "round_up is incorrect (too high)"
    return res


def wr_solve(inst: WeightRestriction, linear: bool, no_jit: bool, verify: bool) -> List[int]:
    # This is the largest integer smaller than inst.tw * inst.total_weight
    knapsack_weight = ceil(inst.tw * inst.total_weight) - 1

    def check_solution_fast(t: List[int]) -> bool:
        return knapsack_upper_bound(inst.weights, t, knapsack_weight) < inst.tn * sum(t)

    def check_solution_slow(t: List[int]) -> bool:
        knapsack_res = knapsack(inst.weights, t, knapsack_weight,
                                upper_bound=ceil(inst.tn * sum(t)), no_jit=no_jit)
        res = knapsack_res < inst.tn * sum(t)
        return res

    return generic_solver(
        inst=inst,
        shift=inst.tw,
        verify=verify,
        linear=linear,
        solution_upper_bound=wr_solution_upper_bound(inst),
        check_solution_fast=check_solution_fast,
        check_solution_slow=check_solution_slow,
    )


def ws_solve(inst: WeightSeparation, linear: bool, no_jit: bool, verify: bool) -> List[int]:
    # This is the largest integer smaller than inst.alpha * inst.total_weight.
    alpha_knapsack_weight = ceil(inst.alpha * inst.total_weight) - 1

    # This is the largest integer smaller than (1 - inst.beta) * inst.total_weight.
    beta_inv_knapsack_weight = ceil((1 - inst.beta) * inst.total_weight) - 1

    def check_solution_fast(t: List[int]) -> bool:
        alpha_knapsack_upper_bound = knapsack_upper_bound(inst.weights, t, alpha_knapsack_weight)
        beta_inv_knapsack_upper_bound = knapsack_upper_bound(inst.weights, t, beta_inv_knapsack_weight)
        return alpha_knapsack_upper_bound < sum(t) - beta_inv_knapsack_upper_bound

    def check_solution_slow(t: List[int]) -> bool:
        sum_t = sum(t)
        # TODO: the two knapsacks can be solved together in one run
        beta_inv_knapsack = knapsack(inst.weights, t, beta_inv_knapsack_weight,
                                     upper_bound=sum_t, no_jit=no_jit)
        alpha_knapsack = knapsack(inst.weights, t, alpha_knapsack_weight,
                                  upper_bound=sum_t - beta_inv_knapsack, no_jit=no_jit)
        return alpha_knapsack < sum_t - beta_inv_knapsack

    return generic_solver(
        inst=inst,
        shift=(inst.alpha + inst.beta) / 2,
        verify=verify,
        linear=linear,
        solution_upper_bound=ws_solution_upper_bound(inst),
        check_solution_fast=check_solution_fast,
        check_solution_slow=check_solution_slow,
    )


def generic_solver(
        inst,
        shift,
        verify,
        linear,
        solution_upper_bound,
        # check_solution_fast gives a conservative estimate, that is, it must return False if the solution is invalid,
        # but it is allowed to return False even if the solution is valid.
        check_solution_fast,
        check_solution_slow,
):
    n = len(inst.weights)
    assert all(isinstance(inst.weights[i], int) for i in range(n))

    if linear:
        def check_solution_slow_override(_: List[int]) -> bool:
            assert False, "unreachable: quadratic check is called in linear mode"

        check_solution_slow = check_solution_slow_override

    if verify:
        # Override check_solution_slow so that when it returns False, it also
        # verifies that check_solution_fast returns False as it should because
        # it is supposed to give a conservative estimate.
        original_check_solution_slow = check_solution_slow

        def check_solution_slow_override(t: List[int]) -> bool:
            res = original_check_solution_slow(t)
            if not res:
                assert not check_solution_fast(t), "check_solution_fast is incorrect"
            return res

        check_solution_slow = check_solution_slow_override

    verify_solution = None
    if verify:
        if linear:
            verify_solution = check_solution_fast
        else:
            verify_solution = check_solution_slow

    eps = Fraction(1, max(inst.weights))

    logging.debug("Binary search for s*...")

    # First, disregard values of s* such that, if there was a local minimum among them,
    # it would violate the upper bound proof.
    logging.debug("Using the solution upper bound to disregard high values of s*...")

    s_low = 0
    s_high = eps
    steps = 0
    while True:
        steps += 1

        t_high = allocate(inst.weights, s_high, shift)
        if sum(t_high) >= solution_upper_bound:
            break

        s_low = s_high
        s_high *= 2

    while s_high != s_low:
        steps += 1

        s_mid = (s_high + s_low) / 2
        t_mid = allocate(inst.weights, s_mid, shift)

        if sum(t_mid) >= solution_upper_bound:
            s_high = round_down(inst.weights, s_mid, shift, t_mid, verify)
        else:
            s_low = round_up(inst.weights, s_mid, shift, t_mid, verify)

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("s* <= %s (%s)", s_high, float(s_high))

    if verify:
        logging.debug("Verifying the s* upper bound...")
        assert verify_solution(allocate(inst.weights, s_high, shift)), "s* upper bound is violated"

    # Use knapsack bounds instead of actual knapsack solver to speed up the process
    logging.debug("Using the knapsack bounds to estimate s*...")
    steps = 0
    s_low = 0
    while s_high != s_low:
        steps += 1

        s_mid = (s_high + s_low) / 2
        t_mid = allocate(inst.weights, s_mid, shift)

        if check_solution_fast(t_mid):
            s_high = round_down(inst.weights, s_mid, shift, t_mid, verify)
        else:
            s_low = round_up(inst.weights, s_mid, shift, t_mid, verify)

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("s* <= %s (%s)", s_high, float(s_high))

    if linear:
        logging.debug("Skipping further optimization of s* because linear mode is enabled.")
    else:
        # Use actual knapsack to find a local minimum
        # Using a special type of accelerated binary search that is fast with a good initial estimate
        logging.debug("Using knapsack solver to find s* precisely...")
        speed = eps
        s_low = 0

        steps = 0
        while s_high != s_low:
            steps += 1

            if 2 * speed < s_high - s_low:
                # Move from s_high with an acceleration
                s_mid = s_high - speed
                speed *= 2
            else:
                # Fall back to regular binary search
                s_mid = (s_high + s_low) / 2

            t_mid = allocate(inst.weights, s_mid, shift)

            if check_solution_slow(t_mid):
                s_high = round_down(inst.weights, s_mid, shift, t_mid, verify)
            else:
                s_low = round_up(inst.weights, s_mid, shift, t_mid, verify)

        logging.debug(f"Finished in {steps} steps.")
        logging.debug("s* = %s (%s)", s_high, float(s_high))

    t_low = allocate(inst.weights, s_high - eps, shift)
    t_high = allocate(inst.weights, s_high, shift)

    border_set = [i for i in range(inst.n) if t_low[i] != t_high[i]]
    assert all(t_low[i] == t_high[i] - 1 for i in border_set)

    if verify:
        logging.debug("Verifying the intermediate solution...")
        assert verify_solution(t_high), "s* is too low"

    # do binary search to determine how many parties in the border set should be rounded up
    k_low = 0
    k_high = len(border_set) + 1

    logging.debug("Binary search for optimal k*...")

    # Again, first, disregard the values of k* that would violate the upper bound proof.
    logging.debug("Using the solution upper bound to disregard high values of k*...")
    steps = 0
    while k_high - k_low > 1:
        steps += 1

        k_mid = (k_high + k_low) // 2
        t_mid = [t_low[i] if i in border_set[k_mid:] else t_high[i] for i in range(inst.n)]

        if sum(t_mid) > solution_upper_bound:
            k_high = k_mid
        else:
            k_low = k_mid

    # k_low is the largest value of k that satisfies the upper bound
    # hence, k* <= k_low
    assert k_high == k_low + 1
    k_high = k_low

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("k* <= %s/%s", k_high, len(border_set))

    # Use knapsack bounds instead of actual knapsack to speed up the process
    logging.debug("Using knapsack bounds to estimate k*...")
    steps = 0
    k_low = 0
    while k_high - k_low > 1:
        steps += 1

        k_mid = (k_high + k_low) // 2
        t_mid = [t_low[i] if i in border_set[k_mid:] else t_high[i] for i in range(inst.n)]

        if check_solution_fast(t_mid):
            k_high = k_mid
        else:
            k_low = k_mid

    logging.debug(f"Finished in {steps} steps.")
    logging.debug("k* <= %s/%s", k_high, len(border_set))

    if linear:
        logging.debug("Skipping further optimization of k* because linear mode is enabled.")
    else:
        # Use actual knapsack to find a local minimum
        # Using a special type of accelerated binary search that is fast with a good initial estimate
        logging.debug("Using knapsack solver to find k* precisely...")

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

            if check_solution_slow(t_mid):
                k_high = k_mid
            else:
                k_low = k_mid

        logging.debug(f"Finished in {steps} steps.")
        logging.debug("k* = %s/%s", k_high, len(border_set))

    t_best = [t_low[i] if i in border_set[k_high:] else t_high[i] for i in range(inst.n)]

    if verify:
        logging.debug("Verifying the final solution...")
        assert verify_solution(t_best), "k* is too low"

    assert sum(t_best) <= solution_upper_bound, "Upper bound is violated"
    return t_best
