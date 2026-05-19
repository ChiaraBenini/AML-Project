"""
Baseline test ordering strategies and evaluation utilities.

Implements:
- Original order (as-is in the dataset)
- Greedy coverage ordering (Chiara's approach)
- Detection curve computation
- Early-stop simulation for time savings
"""

import numpy as np
import pandas as pd
from src.data_loader import WaferData


# ---------------------------------------------------------------------------
# Greedy ordering: pick test that catches most uncovered failures
# ---------------------------------------------------------------------------

def greedy_test_order(wafer: WaferData, alpha: float = 0.0) -> list:
    """
    Greedy ordering that maximizes cumulative failure coverage.

    At each step picks the test catching the most not-yet-detected failing dies.
    Optional alpha biases toward tests with high standalone fail rate.

    Args:
        wafer: training wafer data
        alpha: weight on standalone fail-rate bonus (0 = pure greedy)

    Returns:
        Ordered list of test column names.
    """
    df_bin = wafer.binary
    failure_sets = {
        col: set(np.where(df_bin[col].values == 1)[0])
        for col in wafer.test_cols
    }

    remaining = set(wafer.test_cols)
    detected = set()
    ordered = []

    while remaining:
        best_test = None
        best_score = -1

        for test in remaining:
            new_coverage = len(failure_sets[test] - detected)
            if new_coverage == 0:
                continue
            score = new_coverage + alpha * len(failure_sets[test])
            if score > best_score:
                best_score = score
                best_test = test

        if best_test is None:
            # No remaining test adds coverage → append rest arbitrarily
            ordered.extend(sorted(remaining))
            break

        ordered.append(best_test)
        detected |= failure_sets[best_test]
        remaining.remove(best_test)

    return ordered


# ---------------------------------------------------------------------------
# Multi-wafer greedy: train on combined wafers for better generalization
# ---------------------------------------------------------------------------

def multi_wafer_greedy_order(wafers: list, alpha: float = 0.0) -> list:
    """
    Greedy ordering trained on multiple wafers combined.
    Failure sets are the union across all training wafers.

    Args:
        wafers: list of WaferData objects to train on
        alpha: fail-rate bias

    Returns:
        Ordered list of test column names.
    """
    test_cols = wafers[0].test_cols

    # Build combined failure sets (offset indices per wafer)
    combined_failure_sets = {col: set() for col in test_cols}
    offset = 0
    for w in wafers:
        for col in test_cols:
            fails = np.where(w.binary[col].values == 1)[0]
            combined_failure_sets[col] |= set(fails + offset)
        offset += len(w.binary)

    remaining = set(test_cols)
    detected = set()
    ordered = []

    while remaining:
        best_test = None
        best_score = -1

        for test in remaining:
            new_coverage = len(combined_failure_sets[test] - detected)
            if new_coverage == 0:
                continue
            score = new_coverage + alpha * len(combined_failure_sets[test])
            if score > best_score:
                best_score = score
                best_test = test

        if best_test is None:
            ordered.extend(sorted(remaining))
            break

        ordered.append(best_test)
        detected |= combined_failure_sets[best_test]
        remaining.remove(best_test)

    return ordered


# ---------------------------------------------------------------------------
# Detection curve: how many fails caught after each test in sequence
# ---------------------------------------------------------------------------

def detection_curve(wafer: WaferData, order: list) -> np.ndarray:
    """
    Cumulative detection curve: curve[i] = # failing dies caught
    after running the first (i+1) tests in `order`.
    """
    df_bin = wafer.binary
    fail_indices = set(np.where(wafer.y.values == 1)[0])

    detected = set()
    curve = []
    for test in order:
        detected |= set(np.where(df_bin[test].values == 1)[0])
        curve.append(len(detected & fail_indices))

    return np.array(curve)


# ---------------------------------------------------------------------------
# Early-stop simulation
# ---------------------------------------------------------------------------

def simulate_early_stop(wafer: WaferData, order: list) -> dict:
    """
    Simulate sequential testing with stop-at-first-failure.

    Returns dict with:
        tests_per_die: array of how many tests each die runs
        total_tests: total test count in sequence
        n_dies: number of dies
        savings_all: fraction of test executions saved (all dies)
        savings_failing: fraction saved (failing dies only)
        avg_tests_all: avg tests run per die
        avg_tests_failing: avg tests run per failing die
        missed: number of failing dies not caught
    """
    df_bin = wafer.binary
    n_dies = len(df_bin)
    n_tests = len(order)

    # Default: every die runs all tests (no failure found)
    stop_at = np.full(n_dies, n_tests)

    for t_idx, test in enumerate(order):
        failing = np.where(df_bin[test].values == 1)[0]
        # Only stop dies that haven't already stopped
        for die in failing:
            if stop_at[die] == n_tests:
                stop_at[die] = t_idx + 1

    fail_mask = wafer.y.values.astype(bool)
    max_cost = n_dies * n_tests
    actual_cost = stop_at.sum()

    # Check if any failing dies were missed entirely
    detected = set()
    for test in order:
        detected |= set(np.where(df_bin[test].values == 1)[0])
    all_failing = set(np.where(fail_mask)[0])
    missed = len(all_failing - detected)

    fail_stop = stop_at[fail_mask]
    fail_max = fail_mask.sum() * n_tests

    return {
        "tests_per_die": stop_at,
        "total_tests": n_tests,
        "n_dies": n_dies,
        "savings_all": (max_cost - actual_cost) / max_cost,
        "savings_failing": (fail_max - fail_stop.sum()) / fail_max if fail_max > 0 else 0,
        "avg_tests_all": stop_at.mean(),
        "avg_tests_failing": fail_stop.mean() if len(fail_stop) > 0 else 0,
        "missed": missed,
    }


# ---------------------------------------------------------------------------
# Summary table across multiple wafers
# ---------------------------------------------------------------------------

def evaluate_order(order: list, wafers: dict, order_name: str = "order") -> pd.DataFrame:
    """
    Evaluate a test order across all wafers and return a summary DataFrame.
    """
    rows = []
    for wafer_name, wafer in sorted(wafers.items()):
        curve = detection_curve(wafer, order)
        sim = simulate_early_stop(wafer, order)
        total_fails = int(wafer.y.sum())

        rows.append({
            "order": order_name,
            "wafer": wafer_name,
            "total_fails": total_fails,
            "detected_at_10": int(curve[min(9, len(curve)-1)]),
            "detected_at_20": int(curve[min(19, len(curve)-1)]),
            "detected_at_50": int(curve[min(49, len(curve)-1)]),
            "pct_at_10": curve[min(9, len(curve)-1)] / total_fails if total_fails else 0,
            "pct_at_20": curve[min(19, len(curve)-1)] / total_fails if total_fails else 0,
            "pct_at_50": curve[min(49, len(curve)-1)] / total_fails if total_fails else 0,
            "missed": sim["missed"],
            "avg_tests_failing": sim["avg_tests_failing"],
            "savings_failing": sim["savings_failing"],
            "savings_all": sim["savings_all"],
        })

    return pd.DataFrame(rows)
