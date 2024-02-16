from numba import jit


# Returns the largest integer k such that k < num/den.
# k = ceil(num/den) - 1
@jit(nopython=True)
def largest_int_smaller_than_frac(num: int, den: int) -> int:
    return (num - 1) // den


# Returns the smallest integer k such that k > num/den.
# k = floor(num/den) + 1
@jit(nopython=True)
def smallest_int_larger_than_frac(num: int, den: int) -> int:
    return num // den + 1


# Returns ceil(num/den).
@jit(nopython=True)
def ceil_frac(num: int, den: int) -> int:
    return (num + den - 1) // den


# Returns the upper bound on the WR solution.
@jit(nopython=True)
def wr_upper_bound(n: int, tw_num: int, tn_num: int, den: int) -> int:
    # \left\lceil \frac{\alpha_w(1 - \alpha_w)}{\alpha_n - \alpha_n} n \right\rceil
    return ceil_frac(n * tw_num * (den - tw_num), (tn_num - tw_num) * den)


@jit(nopython=True)
def two_classes_wr(class_a_weight: int, class_a_count: int,
                   class_b_weight: int, class_b_count: int,
                   tw_num: int, tn_num: int, den: int) -> int:
    """
    Solves Weight Restriction for the special case when there are parties of only two classes:
    class_a_count parties of class_a_weight weight and class_b_count parties of class_b_weight weight.
    The thresholds are represented as integers:
    tw = tw_num / den
    tn = tn_num / den
    """

    assert class_a_count >= 1 and class_b_count >= 1 and class_a_weight > 0 and class_b_weight > 0
    assert tw_num < tn_num

    total_weight = class_a_weight * class_a_count + class_b_weight * class_b_count
    budget = largest_int_smaller_than_frac(tw_num * total_weight, den)

    upper_bound = wr_upper_bound(class_a_count + class_b_count, tw_num, tn_num, den)

    # logging.debug("upper_bound: %s", upper_bound)
    for total_tickets in range(1, upper_bound + 1):
        for class_a_total_tickets in range(total_tickets + 1):
            class_b_total_tickets = total_tickets - class_a_total_tickets

            # This code relies on the fact that there always exists an optimal solution
            # in which parties with the same weight get nearly the same number of keys
            # (the difference is at most 1).
            # a_small_keys is the minimum number of keys a party in class A will get
            a_small_keys = class_a_total_tickets // class_a_count
            # a_small_count is the number of parties in class A that will get a_small_keys+1 keys
            a_big_count = class_a_total_tickets % class_a_count
            # a_small_count is the number of parties in class A that will get a_small_keys keys
            a_small_count = class_a_count - a_big_count
            assert a_small_keys * a_small_count + (a_small_keys+1) * a_big_count == class_a_total_tickets

            # This code is analogous to the one above
            b_small_keys = class_b_total_tickets // class_b_count
            b_big_count = class_b_total_tickets % class_b_count
            b_small_count = class_b_count - b_big_count
            assert b_small_keys * b_small_count + (b_small_keys+1) * b_big_count == class_b_total_tickets

            max_s_tickets = 0
            # The following code goes through all possible ways to compose the set S by going through all
            # possible numbers of class A parties in S. It then greedily uses the rest of the budget
            # in the most efficient way possible.

            # let a_s be the optimal number of parties of class A in S. The following must hold:
            # a_s * class_a_weight + class_b_count * class_b_weight > budget - class_a_weight
            # (otherwise, one more party of class A can be added while staying within the budget)
            # (note that budget is already strictly smaller than tw * W)
            # <=> a_s > (budget - class_a_weight - class_b_count * class_b_weight) / class_a_weight
            min_reasonable_a_s = max(
                0,
                smallest_int_larger_than_frac(budget - class_a_weight - class_b_weight * class_b_count, class_a_weight)
            )

            # a_s * class_a_weight <= budget
            # <=> a_s <= budget / class_a_weight
            # <=> a_s <= budget // class_a_weight
            max_reasonable_a_s = min(class_a_count, budget // class_a_weight)

            for a_s in range(min_reasonable_a_s, max_reasonable_a_s + 1):
                # a_s * class_a_weight + b_s * class_b_weight <= budget
                # <=> b_s <= (budget - a_s * class_a_weight) / class_b_weight
                # <=> b_s <= (budget - a_s * class_a_weight) // class_b_weight
                b_s = min(class_b_count, (budget - a_s * class_a_weight) // class_b_weight)

                s_tickets = min(a_s, a_big_count) * (a_small_keys+1) \
                            + max(0, a_s - a_big_count) * a_small_keys \
                            + min(b_s, b_big_count) * (b_small_keys+1) \
                            + max(0, b_s - b_big_count) * b_small_keys
                max_s_tickets = max(max_s_tickets, s_tickets)

            # logging.debug("max_s_tickets = %s, total_tickets = %s", max_s_tickets, total_tickets)
            if max_s_tickets <= largest_int_smaller_than_frac(tn_num * total_tickets, den):
                return total_tickets

    raise Exception("No solution found within the upper bound")
