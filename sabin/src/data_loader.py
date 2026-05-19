"""
Data loading and preprocessing pipeline for NCA9555 wafer test data.

Reads the raw Excel dataset, separates metadata from test measurements,
extracts threshold rows, creates binary pass/fail labels, and splits
data by wafer. All downstream notebooks import from this module.

Usage:
    from src.data_loader import load_all_wafers, WaferData
    wafers = load_all_wafers()  # returns dict[str, WaferData]
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_XLSX = PROJECT_ROOT / "NCA9555_ROA_dataset2_cleaned.xlsx"

METADATA_COLUMNS = {
    "Source Lot", "Lot", "Wafer", "rework_flag", "Program", "temperature",
    "subid", "site", "die_x", "die_y", "device_nr", "rom_code",
    "hardbin", "lib_info", "BinName", "BinState",
}

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class WaferData:
    """Container for one wafer's processed data."""
    name: str
    meta: pd.DataFrame          # metadata columns
    cont: pd.DataFrame          # continuous test measurements
    binary: pd.DataFrame        # binary fail mask (1 = fail)
    margin: pd.DataFrame        # normalized margin (negative = fail)
    y: pd.Series                # overall die-level fail label
    thresholds: dict             # {test_col: (lower, upper)}
    test_cols: list              # ordered list of test column names


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def _read_raw_excel(path: Path = RAW_XLSX) -> pd.DataFrame:
    """Read the full xlsx (slow – ~390 MB). Caches to parquet on first run."""
    parquet_cache = DATA_DIR / "raw_cache.parquet"

    if parquet_cache.exists():
        print(f"  Loading from parquet cache: {parquet_cache}")
        return pd.read_parquet(parquet_cache)

    print(f"  Reading Excel file (this takes a while): {path}")
    df = pd.read_excel(path, header=0, engine="openpyxl")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(parquet_cache, index=False)
    print(f"  Cached to {parquet_cache}")
    return df


def _separate_columns(df: pd.DataFrame):
    """Identify metadata vs test measurement columns."""
    all_cols = df.columns.tolist()
    meta_cols = [c for c in all_cols if c in METADATA_COLUMNS]
    test_cols = [c for c in all_cols if c not in METADATA_COLUMNS]
    return meta_cols, test_cols


def _extract_thresholds(df: pd.DataFrame, test_cols: list) -> dict:
    """
    First two data rows contain lower/upper thresholds.
    Returns {col: (lower, upper)} for every test column with valid bounds.
    """
    thresholds = {}
    for col in test_cols:
        try:
            lo = float(df[col].iloc[0])
            hi = float(df[col].iloc[1])
            if lo > hi:
                lo, hi = hi, lo
            thresholds[col] = (lo, hi)
        except (ValueError, TypeError):
            pass
    return thresholds


def _build_binary(df_cont: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Create binary fail mask: 1 if measurement is outside thresholds."""
    fail_frames = {}
    for col in df_cont.columns:
        if col not in thresholds:
            continue
        lo, hi = thresholds[col]
        vals = df_cont[col]
        if lo == hi:
            fail = (vals != lo)
        else:
            fail = (vals < lo) | (vals > hi)
        # NaN → fail (conservative)
        fail = fail | vals.isna()
        fail_frames[col] = fail.astype(np.int8)

    return pd.DataFrame(fail_frames, index=df_cont.index)


def _build_margin(df_cont: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Normalized margin: how far inside the pass window.
    Positive = passing, negative = failing. Scaled by (upper - lower).
    """
    margin_data = {}
    for col in df_cont.columns:
        if col not in thresholds:
            continue
        lo, hi = thresholds[col]
        vals = df_cont[col]
        dist_lo = vals - lo
        dist_hi = hi - vals
        margin = np.minimum(dist_lo, dist_hi)
        width = hi - lo
        if width > 0:
            margin = margin / width
        margin_data[col] = margin
    return pd.DataFrame(margin_data, index=df_cont.index)


def _split_wafers(df: pd.DataFrame) -> dict:
    """
    Split dataframe by wafer number (e.g. 801, 806, 812).
    Each wafer has multiple sub-wafers (C1-C10); we merge them.
    Pattern: DMKY801-C_C2 → wafer "801".
    """
    import re
    if "Wafer" not in df.columns:
        raise ValueError("No 'Wafer' column found in dataset")

    # Extract 3-digit wafer number from names like "DMKY801-C_C2"
    df = df.copy()
    df["_wafer_num"] = df["Wafer"].astype(str).str.extract(r"DMKY(\d{3})")

    # Drop rows where we couldn't parse a wafer number (e.g. NaN wafer)
    df = df.dropna(subset=["_wafer_num"])

    wafer_groups = {}
    for wafer_num, group in df.groupby("_wafer_num"):
        wafer_groups[wafer_num] = group.drop(columns=["_wafer_num"]).reset_index(drop=True)

    return wafer_groups


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_all_wafers(xlsx_path: Path = None) -> dict:
    """
    Full pipeline: xlsx → dict of WaferData objects.

    Returns:
        {"801": WaferData, "806": WaferData, "812": WaferData}
    """
    if xlsx_path is None:
        xlsx_path = RAW_XLSX

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    # 1. Read raw data
    df_raw = _read_raw_excel(xlsx_path)
    print(f"  Raw shape: {df_raw.shape}")

    # 2. Separate columns
    meta_cols, test_cols = _separate_columns(df_raw)
    print(f"  Metadata columns: {len(meta_cols)}")
    print(f"  Test columns: {len(test_cols)}")

    # 3. Extract thresholds from first two rows
    thresholds = _extract_thresholds(df_raw, test_cols)
    print(f"  Tests with valid thresholds: {len(thresholds)}")

    # 4. Remove threshold rows, keep only die data
    df_data = df_raw.iloc[2:].reset_index(drop=True)

    # 5. Filter test columns to only those with thresholds
    valid_test_cols = [c for c in test_cols if c in thresholds]

    # 6. Split by wafer
    wafer_groups = _split_wafers(df_data)
    print(f"  Wafers found: {sorted(wafer_groups.keys())}")

    # 7. Build WaferData for each wafer
    result = {}
    for wafer_name, wafer_df in sorted(wafer_groups.items()):
        print(f"\n  Processing wafer {wafer_name} ({len(wafer_df)} dies)...")

        meta = wafer_df[[c for c in meta_cols if c in wafer_df.columns]].copy()
        cont = wafer_df[valid_test_cols].apply(pd.to_numeric, errors="coerce")
        binary = _build_binary(cont, thresholds)
        margin = _build_margin(cont, thresholds)
        y = binary.any(axis=1).astype(np.int8)

        n_fail = int(y.sum())
        print(f"    Tests: {len(valid_test_cols)} | "
              f"Fails: {n_fail} ({n_fail/len(y)*100:.3f}%)")

        result[wafer_name] = WaferData(
            name=wafer_name,
            meta=meta,
            cont=cont,
            binary=binary,
            margin=margin,
            y=y,
            thresholds=thresholds,
            test_cols=valid_test_cols,
        )

    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Convenience: per-test statistics
# ---------------------------------------------------------------------------

def compute_test_stats(wafer: WaferData) -> pd.DataFrame:
    """Compute per-test failure rate, margin stats, and unique fail counts."""
    stats = pd.DataFrame(index=wafer.test_cols)
    stats["fail_count"] = wafer.binary.sum()
    stats["fail_rate"] = wafer.binary.mean()
    stats["margin_mean"] = wafer.margin.mean()
    stats["margin_std"] = wafer.margin.std()
    stats["margin_min"] = wafer.margin.min()

    # First-fail contribution (order-dependent)
    first_fail_counts = []
    failed_before = np.zeros(len(wafer.binary), dtype=bool)
    for col in wafer.test_cols:
        current_fail = wafer.binary[col].values.astype(bool)
        first_fail = current_fail & (~failed_before)
        first_fail_counts.append(first_fail.sum())
        failed_before = failed_before | current_fail
    stats["first_fail"] = first_fail_counts

    return stats.sort_values("fail_rate", ascending=False)


# ---------------------------------------------------------------------------
# CLI: run directly to build the cache
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wafers = load_all_wafers()
    for name, wd in wafers.items():
        print(f"\nWafer {name}:")
        stats = compute_test_stats(wd)
        print(f"  Top 5 tests by fail rate:")
        for test, row in stats.head(5).iterrows():
            print(f"    {test[:55]}: {row['fail_rate']*100:.4f}% "
                  f"(first-fail: {int(row['first_fail'])})")
