"""
Full evaluation pipeline: compare all test ordering strategies.

Run this script to produce the comparison table and detection curve data.
Usage:
    python -m src.evaluation
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loader import load_all_wafers
from src.baseline import (
    greedy_test_order,
    multi_wafer_greedy_order,
    evaluate_order,
    detection_curve,
    simulate_early_stop,
)
from src.adaptive import AdaptiveScheduler, simulate_adaptive


def run_full_evaluation():
    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    wafers = load_all_wafers()

    # ---------------------------------------------------------------
    # 2. Build all orderings
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING TEST ORDERINGS")
    print("=" * 70)

    # A. Original order
    original = wafers["801"].test_cols
    print("  A. Original order: dataset column order")

    # B. Greedy on wafer 801 only (Chiara's approach)
    greedy_801 = greedy_test_order(wafers["801"])
    print("  B. Single-wafer greedy (trained on 801)")

    # C. Multi-wafer greedy (trained on 801 + 806)
    multi_greedy = multi_wafer_greedy_order([wafers["801"], wafers["806"]])
    print("  C. Multi-wafer greedy (trained on 801+806)")

    # D. ML-adaptive with varying seed sizes
    seed_sizes = [10, 15, 20]
    schedulers = {}
    adaptive_orders = {}

    for n_seed in seed_sizes:
        seed = multi_greedy[:n_seed]
        print(f"  D-{n_seed}. ML-adaptive (seed={n_seed}, trained on 801+806)...", end="")
        sched = AdaptiveScheduler.train(
            [wafers["801"], wafers["806"]], seed, verbose=False
        )
        schedulers[n_seed] = sched
        adaptive_orders[n_seed] = sched.get_adaptive_order_for_eval(wafers["812"])
        print(" done")

    # ---------------------------------------------------------------
    # 3. Evaluate on HELD-OUT wafer 812
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT WAFER 812")
    print("=" * 70)

    eval_wafers = {"812": wafers["812"]}

    results = [
        evaluate_order(original, eval_wafers, "Original"),
        evaluate_order(greedy_801, eval_wafers, "Greedy (801)"),
        evaluate_order(multi_greedy, eval_wafers, "Multi-greedy (801+806)"),
    ]
    for n_seed in seed_sizes:
        results.append(
            evaluate_order(
                adaptive_orders[n_seed], eval_wafers,
                f"ML-adaptive (seed={n_seed})"
            )
        )

    df_results = pd.concat(results)

    cols = [
        "order", "total_fails",
        "detected_at_10", "pct_at_10",
        "detected_at_20", "pct_at_20",
        "detected_at_50", "pct_at_50",
        "missed", "avg_tests_failing", "savings_failing",
    ]

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n" + df_results[cols].to_string(index=False, float_format="{:.4f}".format))

    # ---------------------------------------------------------------
    # 4. Per-die adaptive simulation (the real advantage)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-DIE ADAPTIVE SIMULATION (personalized order per die)")
    print("=" * 70)

    for n_seed in seed_sizes:
        sim = simulate_adaptive(wafers["812"], schedulers[n_seed])
        total_fails = int(wafers["812"].y.sum())
        print(
            f"  Seed={n_seed:2d}: missed={sim['missed']} | "
            f"avg_tests_failing={sim['avg_tests_failing']:5.1f} | "
            f"savings_failing={sim['savings_failing']*100:5.1f}% | "
            f"savings_all={sim['savings_all']*100:.2f}%"
        )

    # ---------------------------------------------------------------
    # 5. Cross-validation: train on each pair, test on the third
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-WAFER VALIDATION")
    print("=" * 70)

    wafer_keys = ["801", "806", "812"]
    for test_key in wafer_keys:
        train_keys = [k for k in wafer_keys if k != test_key]
        train_list = [wafers[k] for k in train_keys]

        mg = multi_wafer_greedy_order(train_list)
        seed = mg[:15]
        sched = AdaptiveScheduler.train(train_list, seed, verbose=False)

        # Evaluate on held-out
        sim = simulate_adaptive(wafers[test_key], sched)
        total_fails = int(wafers[test_key].y.sum())

        # Also get detection curve for greedy baseline
        curve_mg = detection_curve(wafers[test_key], mg)

        print(
            f"  Train={'+'.join(train_keys)}, Test={test_key}: "
            f"{total_fails} fails | "
            f"greedy@10={curve_mg[9]}/{total_fails} "
            f"({curve_mg[9]/total_fails*100:.0f}%) | "
            f"adaptive: missed={sim['missed']}, "
            f"avg_fail_tests={sim['avg_tests_failing']:.1f}, "
            f"savings_fail={sim['savings_failing']*100:.1f}%"
        )

    # ---------------------------------------------------------------
    # 6. Save detection curves for plotting
    # ---------------------------------------------------------------
    print("\n  Saving detection curves to results/detection_curves.csv...")
    curve_data = {}
    w812 = wafers["812"]
    curve_data["original"] = detection_curve(w812, original)
    curve_data["greedy_801"] = detection_curve(w812, greedy_801)
    curve_data["multi_greedy"] = detection_curve(w812, multi_greedy)
    for n_seed in seed_sizes:
        curve_data[f"adaptive_seed{n_seed}"] = detection_curve(
            w812, adaptive_orders[n_seed]
        )

    curves_df = pd.DataFrame(curve_data)
    curves_df.index.name = "test_index"
    curves_df.to_csv("results/detection_curves.csv")
    print("  Done.")

    # Save summary table
    df_results.to_csv("results/comparison_table.csv", index=False)
    print("  Saved comparison table to results/comparison_table.csv")

    return df_results


if __name__ == "__main__":
    run_full_evaluation()
