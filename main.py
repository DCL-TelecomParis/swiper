#!/usr/bin/env python3

import argparse
import logging
import math
import sys
from fractions import Fraction
from typing import List

from solver import solve, knapsack
from solver.util import lcm, gcd
from solver.wr import WeightRestriction
from solver.wq import WeightQualification

logger = logging.getLogger(__name__)


def parse_input(inp: str) -> List[Fraction]:
    return [Fraction(s) for s in inp.split()]


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="swiper",
        description="Swiper: an experimental solver to the Weight Restriction (WR) "
                    "and Weight Qualification (WQ) problems",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("input_file", type=argparse.FileType("r"), default=sys.stdin, nargs='?',
                               help="The path to the input file. "
                                    "If absent, the standard input will be used. "
                                    "The input consists of weights of parties separated by any whitespaces. "
                                    "Rational numbers and decimals are accepted.")
    common_parser.add_argument("--no-jit", action="store_true",
                               help="Do not use JIT compilation. "
                                    "It may result in a significant performance degradation.")
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False,
                               help="Set this flag to enable verbose logging.")
    common_parser.add_argument("-vv", "--very-verbose", action="store_true", default=False,
                               help="Set this flag to enable very verbose logging.")
    common_parser.add_argument("--debug", action="store_true",
                               help="Verify the validity of the final solution and of some intermediate results.")

    subparsers = parser.add_subparsers(title="problem", required=True, dest="problem")

    wr_parser = subparsers.add_parser("wr", parents=[common_parser],
                                      help="Solve the Weight Restriction problem, i.e., "
                                           "ensure that any group of parties with less than "
                                           "alpha_w fraction of total weight obtains less than "
                                           "alpha_n fraction of total tickets.")
    wr_parser.add_argument("--tw", "--alpha_w", type=Fraction, required=True,
                           help="The weighted threshold. Corresponds to alpha_w in the paper. "
                                "Must be smaller than the nominal threshold alpha_n. "
                                "Can be fractional (e.g., 0.01 or 5/7).")
    wr_parser.add_argument("--tn", "--alpha_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to alpha_n in the paper. "
                                "Must be greater than the weighted threshold alpha_w. "
                                "Can be fractional (e.g., 0.01 or 5/7).")

    wq_parser = subparsers.add_parser("wq", parents=[common_parser],
                                      help="Solve the Weight Qualification problem, i.e., "
                                           "ensure that any group of parties with more than "
                                           "beta_w fraction of total weight obtains more than "
                                           "beta_n fraction of total tickets.")
    wq_parser.add_argument("--tw", "--beta_w", type=Fraction, required=True,
                           help="The weighted threshold. Corresponds to beta_w in the paper. "
                                "Must be greater than the nominal threshold beta_n. "
                                "Can be fractional (e.g., 0.01 or 5/7).")
    wq_parser.add_argument("--tn", "--beta_n", type=Fraction, required=True,
                           help="The nominal threshold. Corresponds to beta_n in the paper. "
                                "Must be smaller than the weighted threshold beta_w. "
                                "Can be fractional (e.g., 0.01 or 5/7).")

    args = parser.parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.very_verbose else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s")

    # Disable numba debug logging
    if args.very_verbose and not args.no_jit:
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.INFO)

    weights = parse_input(args.input_file.read())

    # Help the IDE determine the types
    args.tw = Fraction(args.tw)
    args.tn = Fraction(args.tn)

    # Convert weights to integers
    denominator_lcm = lcm(w.denominator for w in weights + [args.tw, args.tn])
    numerator_gcd = gcd(w.numerator for w in weights)
    weights = [int(w * denominator_lcm // numerator_gcd) for w in weights]

    if args.problem == "wr":
        inst = WeightRestriction(weights, args.tw, args.tn)
    else:
        inst = WeightQualification(weights, args.tw, args.tn)

    logger.info("Problem: %s", inst)
    logger.info("Total weight: %s", inst.total_weight)
    logger.info("Threshold weight: %s", inst.threshold_weight)

    solution = solve(inst, args.no_jit, verify=args.debug)
    logger.info("Solution: %s", sum(solution))

    assert solution is not None
    logger.info(solution)

    print(f"Total tickets allocated: {sum(solution)}.")


if __name__ == '__main__':
    main(sys.argv[1:])
