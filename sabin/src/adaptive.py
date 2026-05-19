"""
ML-based adaptive test scheduling.

The idea: after running the first K tests on a die, use the continuous
measurement values to predict which remaining test is most likely to
reveal a failure. This lets us "jump" to the most informative next test
instead of following a fixed sequence.

Architecture:
    1. A "failure probability" model: given partial results from the first K
       tests, predict P(die fails any remaining test).
    2. A "next-best-test" selector: among remaining tests, pick the one whose
       predicted failure probability is highest given the current observations.

The model is trained on margin values (continuous, normalized distance from
thresholds) because they carry more signal than binary pass/fail.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.data_loader import WaferData


# ---------------------------------------------------------------------------
# Feature engineering: partial observation vectors
# ---------------------------------------------------------------------------

def _build_partial_features(wafer: WaferData, observed_tests: list) -> np.ndarray:
    """
    Build feature matrix from margin values of observed tests.
    Shape: (n_dies, len(observed_tests))
    NaN margins (shouldn't happen, but safety) are filled with 0.
    """
    return wafer.margin[observed_tests].fillna(0).values


# ---------------------------------------------------------------------------
# Per-test failure predictor
# ---------------------------------------------------------------------------

def train_test_predictor(
    train_wafers: list,
    observed_tests: list,
    target_test: str,
    max_depth: int = 4,
    n_estimators: int = 100,
) -> XGBClassifier:
    """
    Train a classifier: given margin values for `observed_tests`,
    predict whether `target_test` will fail (binary).

    Trains on concatenated data from all `train_wafers`.
    """
    X_parts = []
    y_parts = []
    for w in train_wafers:
        X_parts.append(_build_partial_features(w, observed_tests))
        y_parts.append(w.binary[target_test].values)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # If target never fails in training data, skip training
    if y.sum() == 0:
        return None

    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        scale_pos_weight=max(1, (len(y) - y.sum()) / max(y.sum(), 1)),
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Adaptive scheduling engine
# ---------------------------------------------------------------------------

class AdaptiveScheduler:
    """
    Adaptive test scheduler that uses ML to decide the next test.

    Usage:
        scheduler = AdaptiveScheduler.train(train_wafers, seed_tests, candidate_tests)
        order_per_die = scheduler.schedule(eval_wafer)
    """

    def __init__(self, seed_order: list, models: dict, candidate_tests: list):
        """
        Args:
            seed_order: fixed initial tests to run on every die
            models: {test_name: XGBClassifier} for each candidate test
            candidate_tests: tests that can be scheduled adaptively
        """
        self.seed_order = seed_order
        self.models = models
        self.candidate_tests = candidate_tests

    @classmethod
    def train(
        cls,
        train_wafers: list,
        seed_order: list,
        candidate_tests: list = None,
        max_depth: int = 4,
        n_estimators: int = 100,
        verbose: bool = True,
    ):
        """
        Train predictive models for each candidate test.

        Args:
            train_wafers: list of WaferData for training
            seed_order: fixed initial tests (features for prediction)
            candidate_tests: tests to predict failures for.
                If None, uses all tests not in seed_order.
        """
        all_tests = train_wafers[0].test_cols
        if candidate_tests is None:
            candidate_tests = [t for t in all_tests if t not in seed_order]

        models = {}
        n_trained = 0

        for i, target in enumerate(candidate_tests):
            model = train_test_predictor(
                train_wafers, seed_order, target,
                max_depth=max_depth, n_estimators=n_estimators,
            )
            if model is not None:
                models[target] = model
                n_trained += 1

            if verbose and (i + 1) % 50 == 0:
                print(f"  Trained {i+1}/{len(candidate_tests)} models "
                      f"({n_trained} with failures)")

        if verbose:
            print(f"  Done: {n_trained} models trained out of "
                  f"{len(candidate_tests)} candidates")

        return cls(seed_order, models, candidate_tests)

    def predict_failure_probs(self, wafer: WaferData) -> pd.DataFrame:
        """
        For each die, predict failure probability on each candidate test.

        Returns DataFrame: (n_dies, n_candidates) with probabilities.
        """
        X = _build_partial_features(wafer, self.seed_order)
        probs = {}

        for test, model in self.models.items():
            probs[test] = model.predict_proba(X)[:, 1]

        # Tests without a model (never fail in training) get probability 0
        for test in self.candidate_tests:
            if test not in probs:
                probs[test] = np.zeros(len(wafer.binary))

        return pd.DataFrame(probs, index=wafer.binary.index)

    def schedule(self, wafer: WaferData, max_tests: int = None) -> np.ndarray:
        """
        Adaptive scheduling: for each die, order remaining tests by
        predicted failure probability (highest first).

        Returns:
            stop_at: array of shape (n_dies,) with # tests run per die
                     (stops at first failure, like early-stop simulation)
        """
        if max_tests is None:
            max_tests = len(self.seed_order) + len(self.candidate_tests)

        n_dies = len(wafer.binary)
        df_bin = wafer.binary

        # Step 1: Run seed tests on all dies
        stop_at = np.full(n_dies, max_tests)
        detected = np.zeros(n_dies, dtype=bool)

        for t_idx, test in enumerate(self.seed_order):
            failing = df_bin[test].values.astype(bool)
            newly_caught = failing & ~detected
            stop_at[newly_caught] = t_idx + 1
            detected |= failing

        if detected.all():
            return stop_at

        # Step 2: Get failure probability predictions
        prob_df = self.predict_failure_probs(wafer)

        # Step 3: For each still-active die, schedule remaining tests
        # by descending predicted failure probability
        active_mask = ~detected
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return stop_at

        # For each active die, get its personalized test order
        n_seed = len(self.seed_order)
        remaining_tests = list(prob_df.columns)

        for die_idx in active_indices:
            # Sort remaining tests by predicted failure prob (desc)
            die_probs = prob_df.iloc[die_idx]
            sorted_tests = die_probs.sort_values(ascending=False).index.tolist()

            # Walk through personalized order, stop at first failure
            for t_offset, test in enumerate(sorted_tests):
                if df_bin[test].iloc[die_idx] == 1:
                    stop_at[die_idx] = n_seed + t_offset + 1
                    break

        return stop_at

    def get_adaptive_order_for_eval(self, wafer: WaferData) -> list:
        """
        Build a single global test order by averaging predicted failure
        probabilities across all dies. This is a simpler evaluation mode
        that produces one fixed order (useful for comparison with baselines).
        """
        prob_df = self.predict_failure_probs(wafer)
        # Average predicted failure prob across all dies
        avg_probs = prob_df.mean(axis=0).sort_values(ascending=False)
        return self.seed_order + avg_probs.index.tolist()


# ---------------------------------------------------------------------------
# Full adaptive simulation (for evaluation)
# ---------------------------------------------------------------------------

def simulate_adaptive(wafer: WaferData, scheduler: AdaptiveScheduler) -> dict:
    """
    Run full adaptive simulation and return metrics comparable to
    baseline.simulate_early_stop().
    """
    stop_at = scheduler.schedule(wafer)
    n_tests = len(scheduler.seed_order) + len(scheduler.candidate_tests)
    n_dies = len(wafer.binary)

    fail_mask = wafer.y.values.astype(bool)
    max_cost = n_dies * n_tests
    actual_cost = stop_at.sum()

    # Check missed failures
    df_bin = wafer.binary
    detected = set()
    for test in scheduler.seed_order:
        detected |= set(np.where(df_bin[test].values == 1)[0])
    for test in scheduler.candidate_tests:
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
