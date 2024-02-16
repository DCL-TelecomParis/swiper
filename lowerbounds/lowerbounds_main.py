#!/usr/bin/env python3

import argparse
import logging
import sys
from fractions import Fraction
from math import lcm
from typing import List

from numba import jit, prange
from numba_progress import ProgressBar

from two_classes_wr import two_classes_wr, wr_upper_bound


def two_classes(n: int, max_total_weight: int, tw_num: int, tn_num: int, den: int):
    """
    Search for the worst-case input to the Weight Restriction problem with two classes of parties.
    class_a_count parties of class_a_weight weight and class_b_count parties of class_b_weight weight.
    The thresholds are represented as integers:
    tw = tw_num / den
    tn = tn_num / den
    """
    worst_example_wr = -1
    worst_example = (-1, -1, -1, -1)

    # WLOG, we can assume that class A is the smallest of the two.
    # Hence, it has at most n // 2 parties.
    for class_a_count in prange(1, n // 2 + 1):
        logging.info("trying %s class A nodes", class_a_count)
        class_b_count = n - class_a_count

        class_a_weight = 0
        while True:
            class_a_weight += 1

            class_a_total_weight = class_a_weight * class_a_count
            class_b_weight = (max_total_weight - class_a_total_weight) // class_b_count
            if class_b_weight <= 0:
                break

            wr = two_classes_wr(class_a_weight, class_a_count, class_b_weight, class_b_count, tw_num, tn_num, den)

            if wr > worst_example_wr:
                worst_example_wr = wr
                worst_example = (class_a_weight, class_a_count, class_b_weight, class_b_count)
                # print("new worst example: " + str(class_a_count) + " class A nodes, " + str(class_b_count) +
                #       " class B nodes, class A weight " + str(class_a_weight) + ", class B weight " +
                #       str(class_b_weight) + ", kap " + str(kap))
                logging.info("new worst example: %s class A nodes, %s class B nodes, "
                             "class A weight %s, class B weight %s, wr %s",
                             class_a_count, class_b_count, class_a_weight, class_b_weight, wr)

    return worst_example_wr, worst_example


@jit(nopython=True, parallel=True)
def two_classes_parallel_impl(n: int, max_total_weight: int, tw_num: int, tn_num: int, den: int, progress_proxy):
    worst_example_wr = -1
    worst_example = (-1, -1, -1, -1)

    temp_worst_example_wr = [-1] * (n // 2)
    temp_worst_example = [(-1, -1, -1, -1)] * (n // 2)

    for class_a_count in prange(1, n // 2 + 1):
        class_b_count = n - class_a_count

        class_a_weight = 0
        while True:
            class_a_weight += 1

            class_a_total_weight = class_a_weight * class_a_count
            class_b_weight = (max_total_weight - class_a_total_weight) // class_b_count
            if class_b_weight <= 0:
                break

            wr = two_classes_wr(class_a_weight, class_a_count, class_b_weight, class_b_count, tw_num, tn_num, den)

            if wr > temp_worst_example_wr[class_a_count - 1]:
                temp_worst_example_wr[class_a_count - 1] = wr
                temp_worst_example[class_a_count - 1] = (class_a_weight, class_a_count, class_b_weight, class_b_count)
        if progress_proxy is not None:
            progress_proxy.update(1)

    # Find the overall worst example outside the parallel loop
    for i in range(len(temp_worst_example_wr)):
        if worst_example_wr == -1 or temp_worst_example_wr[i] > worst_example_wr:
            worst_example_wr = temp_worst_example_wr[i]
            worst_example = temp_worst_example[i]

    return worst_example_wr, worst_example


def two_classes_parallel(n: int, max_total_weight: int, tw_num: int, tn_num: int, den: int, progress_bar: bool):
    if progress_bar:
        with ProgressBar(total=n // 2) as progress:
            return two_classes_parallel_impl(n, max_total_weight, tw_num, tn_num, den, progress)
    else:
        return two_classes_parallel_impl(n, max_total_weight, tw_num, tn_num, den, None)


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Tries to find a worst-case example for the Weight Restriction problem "
                    "by trying a large family of examples of special form and solving it in polynomial time "
                    "relying on that special form."
    )
    parser.add_argument("--tw", "--alpha_w", type=Fraction, required=True,
                        help="The weighted threshold. Corresponds to alpha_w in the paper. "
                             "Must be smaller than the nominal threshold alpha_n. "
                             "Can be fractional (e.g., 0.01 or 5/7).")
    parser.add_argument("--tn", "--alpha_n", type=Fraction, required=True,
                        help="The nominal threshold. Corresponds to alpha_n in the paper. "
                             "Must be greater than the weighted threshold alpha_w. "
                             "Can be fractional (e.g., 0.01 or 5/7).")
    parser.add_argument("-n", type=int, required=True,
                        help="Number of parties")
    parser.add_argument("--max-total-weight", "-W", type=int, required=True,
                        help="Maximum total weight of all parties (integer).")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Use parallelization to speed up the process.")
    parser.add_argument("--progress-bar", action="store_true", default=False,
                        help="Show a progress bar (only available in parallel mode).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable more verbose output.")
    args = parser.parse_args(argv)

    den = lcm(args.tw.denominator, args.tn.denominator)
    tw_num = args.tw.numerator * (den // args.tw.denominator)
    tn_num = args.tn.numerator * (den // args.tn.denominator)

    if args.parallel:
        worst_wr, worst_wr_instance = two_classes_parallel(args.n, args.max_total_weight, tw_num, tn_num, den,
                                                           args.progress_bar)
    else:
        worst_wr, worst_wr_instance = two_classes(args.n, args.max_total_weight, tw_num, tn_num, den)

    if args.verbose:
        print(f"{args.n}: {worst_wr}/{wr_upper_bound(args.n, tw_num, tn_num, den)}")
        print(f"instance: "
              f"{worst_wr_instance[1]}*{worst_wr_instance[0]} & {worst_wr_instance[3]}*{worst_wr_instance[2]}")
    else:
        print(worst_wr)


if __name__ == "__main__":
    main(sys.argv[1:])
