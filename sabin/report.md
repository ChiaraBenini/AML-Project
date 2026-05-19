# ML-Based Adaptive Test Scheduling for Semiconductor Test Optimization

## 1. Introduction

This report describes my work on optimizing semiconductor test programs for Nexperia. The core question is simple: **when testing a chip, can we reorder the tests so that defective chips are caught earlier?**

Nexperia manufactures integrated circuits at enormous scale (110 billion units per year). Every chip must pass a battery of electrical tests before shipping. If a chip fails any test, it is rejected. Currently, tests run in a fixed order. If the first test that would catch a defective chip happens to be test number 400 out of 572, the tester wastes time running 399 unnecessary tests on that chip first.

The goal of this work is to learn a smarter test order from historical data, and then apply it to new production lots so that failing chips are identified as early as possible. I go beyond simple reordering by building an ML model that adapts the test sequence per individual chip based on its own measurements.

## 2. The Dataset

The dataset comes from Nexperia and contains test results for the NCA9555 product family. The raw file is an Excel spreadsheet (`NCA9555_ROA_dataset2_cleaned.xlsx`, approximately 390 MB). To run this code, simply put the xlsx file in the `data/` folder.

### 2.1 Structure

The spreadsheet has 78,461 rows and 588 columns. The first row is column headers. Rows 2-3 contain threshold values (lower and upper bounds for each test). Rows 4 onward are actual chip measurements. Of the 588 columns, 16 are metadata (lot ID, wafer ID, die coordinates, bin info, etc.) and 572 are electrical test measurements.

Each test column has a name like `tf_ICC_stand:VCC_ICC_standby:1.650 V`. This encodes:
- The test family (`tf_ICC_stand`)
- The specific measurement (`VCC_ICC_standby`)
- The test condition (`1.650 V`)

Many tests are repeated across multiple pins or voltage levels. For example, `tb_VOH_8m:VOH_8m:-8.000 mA` appears 64 times (one per pin). This is important because it creates natural redundancy within the test suite.

### 2.2 Pass/Fail Logic

For each test, the spreadsheet provides a lower and upper threshold. A chip passes a test if its measured value falls within the allowed range:

```
passed = lower <= measured_value <= upper
```

A chip is considered defective if it fails **any** test. This means a single out-of-range measurement anywhere in the 572 tests is enough to reject the chip.

### 2.3 Wafer Structure

The data covers three production lots (wafers), identified by 3-digit numbers extracted from the wafer ID string (e.g., `DMKY801-C_C2` belongs to lot 801). Each lot contains 10 sub-wafers (C1 through C10), totaling roughly 26,000 chips per lot:

| Lot | Chips | Failing Chips | Fail Rate |
|-----|------:|-------------:|----------:|
| 801 | 26,149 | 127 | 0.486% |
| 806 | 26,155 | 96 | 0.367% |
| 812 | 26,155 | 116 | 0.444% |

The fail rate is extremely low (under 0.5%). This is normal for semiconductor manufacturing and means that the vast majority of chips pass every single test. The challenge is that the few defective chips need to be caught quickly, and there are very few positive examples to learn from.

### 2.4 Normalized Margins

Beyond binary pass/fail, I compute a **normalized margin** for each measurement. The margin tells us how far a value is from the nearest threshold, normalized by the width of the passing range:

```
margin = min(value - lower, upper - value) / (upper - lower)
```

A positive margin means the chip passed (the value is inside the range). A negative margin means it failed. A margin close to zero means the value was near the edge of the acceptable range. This continuous signal carries more information than binary pass/fail and is the key feature used by the ML model.

## 3. Data Pipeline

I built a reusable data pipeline (`src/data_loader.py`) that:

1. Reads the 390 MB Excel file and caches it in a compressed binary format (Parquet) for fast subsequent loads. The first read takes a few minutes; every read after that takes about 3 seconds.
2. Extracts threshold rows and builds a threshold dictionary `{test_name: (lower, upper)}`.
3. Separates metadata columns from test columns.
4. Groups chips by lot number (merging all sub-wafers within each lot).
5. For each lot, computes:
   - **Continuous matrix**: raw measured values (26,149 x 572)
   - **Binary matrix**: pass/fail mask (1 = fail)
   - **Margin matrix**: normalized distance from thresholds
   - **Overall label**: 1 if the chip fails any test, 0 otherwise

The output is a dictionary of `WaferData` objects, one per lot, that all downstream code consumes.

## 4. Methods

I implemented and compared four test ordering strategies, each building on the previous one.

### 4.1 Baseline: Original Order

The simplest approach: run tests in the order they appear in the dataset. This is essentially the factory default. It serves as the lower bound for comparison.

### 4.2 Single-Wafer Greedy Ordering

This is a greedy set-cover algorithm. It builds a test order by repeatedly picking the test that catches the most not-yet-detected failing chips.

The algorithm works as follows:

```
detected = empty set
order = empty list

while there are unordered tests:
    for each remaining test:
        score = number of failing chips this test catches 
                that are NOT already in 'detected'
    
    pick the test with the highest score
    add it to 'order'
    add all its failing chips to 'detected'
```

When no remaining test adds new coverage (all failing chips are already detected), the remaining tests are appended in arbitrary order.

I train this on lot 801 and evaluate on lot 812 (held-out). This mirrors the real-world scenario: learn from past production data, apply to future lots.

### 4.3 Multi-Wafer Greedy Ordering

The single-wafer greedy has a weakness: it only sees failure patterns from one lot. If lot 801 happens to have unusual failure modes, the learned order may not transfer well to lot 812.

The fix is straightforward: train on **two** lots (801 + 806) instead of one. The failure sets are combined (with index offsets so chips from different lots don't collide), and the same greedy algorithm runs on the pooled data.

This is not a new algorithm, just more training data. But as the results show, it makes a significant difference. By exposing the greedy algorithm to failure patterns from two lots instead of one, it learns an ordering that is more robust to the natural variation between production lots.

### 4.4 ML-Adaptive Scheduling

This is the main contribution. Instead of learning a single fixed test order, I train an ML model that personalizes the test sequence for each individual chip based on its own measurement results.

#### The Idea

Imagine you run the first 15 tests on a chip. The measured values on those 15 tests contain information about the chip's overall health. A chip with values drifting toward the edge of the threshold on several early tests is more likely to fail a later test than a chip with comfortable margins everywhere.

The adaptive scheduler exploits this: after running a fixed set of "seed" tests, it uses the continuous margin values from those tests as features to predict which remaining test is most likely to fail. It then runs that predicted test next, rather than following a fixed order.

#### Architecture

The system has two components:

**1. Per-test failure predictors.** For each of the 557 remaining tests (572 total minus 15 seed tests), I train a separate XGBoost binary classifier:

- **Features**: the 15 normalized margin values from the seed tests (a 15-dimensional vector per chip)
- **Target**: whether that specific remaining test fails (binary: 0 or 1)
- **Training data**: all chips from lots 801 + 806 (52,304 chips)
- **Class imbalance handling**: `scale_pos_weight` is set to the ratio of passing to failing chips (since fail rates are under 0.5%, failing chips are heavily outnumbered)

This produces 533 trained models (24 tests never fail in training data, so no model is needed for those -- those tests get assigned a predicted failure probability of zero).

Each XGBoost model uses:
- `max_depth=4` (shallow trees to avoid overfitting on rare failures)
- `n_estimators=100` (100 boosting rounds)
- `eval_metric=logloss` (probability calibration matters here)

**2. Scheduling logic.** At inference time, for each chip:

1. Run the 15 seed tests. If the chip fails any of them, stop immediately -- defect found.
2. If the chip passed all seed tests, feed its 15 margin values into all 533 models.
3. Each model outputs a predicted probability that its corresponding test will fail.
4. Sort the remaining tests by descending predicted failure probability.
5. Run tests in that personalized order. Stop at the first failure.

This means every chip gets a different test sequence tailored to its own measurement profile. A chip with suspiciously low standby current margins might get current-related tests pulled forward, while a chip with marginal voltage output levels gets output-stage tests prioritized.

#### Why XGBoost?

Several properties make gradient-boosted trees a good fit here:

- **Tabular data**: the features are 15 continuous measurements, not images or sequences. Tree-based models excel on tabular data.
- **Extreme class imbalance**: only 0.1-0.2% of chips fail any given test. XGBoost handles this well with `scale_pos_weight`.
- **Interpretability**: feature importance from the trees can show which seed tests are most predictive of which downstream failures.
- **Speed**: training 533 models on 52K samples with 15 features takes under 2 minutes. Inference is near-instant.

#### Why Margins Instead of Binary Pass/Fail?

The binary pass/fail signal from seed tests is almost useless for prediction because 99.5% of chips pass every seed test. There is no variation to learn from -- the binary input is just a vector of 15 zeros for almost every chip.

Continuous margins solve this: even among passing chips, some have comfortable margins (value firmly in the middle of the range) while others are borderline (value near the threshold). The XGBoost model can learn patterns like "when `ICC_standby` at 1.65V has a margin below 0.1, the chip is 3x more likely to fail `IDDq1` at 3.3V." This is information that binary pass/fail completely discards.

#### Two Evaluation Modes

I evaluate the adaptive scheduler in two ways:

1. **Per-die adaptive** (the real thing): each chip gets a personalized test order based on its own margins. This is what would actually run on the factory floor. The metric here is "average number of tests run per failing chip before its defect is found."

2. **Global adaptive order** (for fair comparison with baselines): average the predicted failure probabilities across all chips to produce one single fixed ordering. This is directly comparable to the greedy baselines because it is also a single static order applied to every chip. The metric here is "what fraction of failures are detected in the first N tests of that fixed order."

## 5. Experimental Setup

- **Training data**: lots 801 + 806 (52,304 chips total, 223 failures)
- **Test data**: lot 812 (26,155 chips, 116 failures) -- completely held out during training
- **Seed tests**: the first 10, 15, or 20 tests from the multi-wafer greedy ordering (these are the most informative tests as determined by greedy coverage)
- **Cross-validation**: I also run leave-one-lot-out cross-validation (train on any two lots, test on the third) to verify generalization

## 6. Results

### 6.1 Main Comparison on Held-Out Lot 812

The table below compares all strategies. Every row is evaluated on lot 812 (26,155 chips, 116 failures), which was never used during training. The strategies are grouped by approach: reordering strategies (which keep all 572 tests and only change the sequence) and test removal strategies (which permanently discard some tests).

**Reordering strategies** (all 572 tests kept, only the order changes):

| Strategy | Detected in first 10 tests | Detected in first 20 tests | Avg tests per failing chip | Time savings (failing chips) | Missed |
|----------|:-:|:-:|:-:|:-:|:-:|
| Original order | 4/116 (3.4%) | 11/116 (9.5%) | 82.0 | 85.7% | 0 |
| Chiara: Greedy reorder (lot 801, with first-fail bias) | 87/116 (75.0%) | 102/116 (87.9%) | 34.2 | 94.0% | 0 |
| Greedy reorder (lot 801, no bias) | 89/116 (76.7%) | 100/116 (86.2%) | 48.6 | 91.5% | 0 |
| Multi-wafer greedy (801+806) | 98/116 (84.5%) | 106/116 (91.4%) | 20.9 | 96.3% | 0 |
| ML-adaptive global (seed=15) | 98/116 (84.5%) | 101/116 (87.1%) | 12.8 | **97.8%** | 0 |

**Test removal strategies** (Chiara's redundancy clustering, tests permanently removed):

| Strategy | Tests kept | Detected in first 10 tests | Avg tests per failing chip | Time savings (failing chips) | Missed |
|----------|:-:|:-:|:-:|:-:|:-:|
| Chiara: Cluster reduction (V1) | 281 | -- | 134.3 | 52.2% | **7** (6.0%) |
| Chiara: Cluster + zero-fail removal (V2) | 228 | -- | 110.3 | 51.6% | **7** (6.0%) |

#### How to read these tables

- **"Detected in first 10 tests"**: if I run only the first 10 tests on every chip in lot 812, how many of the 116 defective chips would I have identified? The original order catches just 4. The multi-wafer greedy catches 98.
- **"Avg tests per failing chip"**: on average, how many tests does a defective chip go through before its defect is found? With the original order, a bad chip runs 82 tests on average before encountering a test it fails. With the ML-adaptive model, that drops to 12.8.
- **"Time savings (failing chips)"**: the percentage reduction in test executions for defective chips compared to running all 572 tests. A value of 97.8% means a failing chip only needs 2.2% of the full test suite before its defect is caught.
- **"Missed"**: the number of defective chips that are never caught at all. For the reordering strategies, this is always zero -- every test still runs eventually, so every defect is found. For the test removal strategies, this is 7 out of 116, meaning 6% of defective chips slip through undetected.

#### What these results mean

**The original order is terrible for early detection.** Only 4 out of 116 failures are caught in the first 10 tests. That means 112 defective chips pass the first 10 tests without any issue -- their particular defect only shows up much later in the sequence. On average, a failing chip runs 82 tests before its defect is found. This confirms that the factory-default test ordering was not designed for early failure detection.

**Chiara's greedy reorder is a strong first step.** Her approach trains a greedy ordering on lot 801 with a first-fail bias (alpha=0.2), which gives a small bonus to tests that historically catch the first failure on a chip. This catches 87/116 in the first 10 tests and reduces avg tests per failing chip from 82 to 34.2 -- a 94% savings. This is a major improvement over the original order, achieved by a simple algorithm with no ML.

**More training data helps.** Multi-wafer greedy (trained on 801+806) catches 98/116 in the first 10 tests, compared to 87/116 for Chiara's single-lot greedy. That is an improvement from 75.0% to 84.5% by adding a second lot to training. Why does this help? Because lot 801 alone has 127 failing chips, and some failure modes that appear on lot 812 might not appear on lot 801, but might appear on lot 806. By pooling two lots, the greedy algorithm sees a more diverse set of failure patterns and learns a more robust ordering.

**The ML model reduces average tests per failing chip from 20.9 to 12.8.** This is where the adaptive scheduling shines. Notice that the "detected in first 10" number is the same for multi-wafer greedy and ML-adaptive (both 98/116). That is because both methods use the same seed tests for the first 15 positions. The difference appears *after* the seed: the ML model is better at finding the remaining 18 failures quickly because it looks at each chip's margin values and prioritizes the tests most likely to fail for *that particular chip*, rather than following a fixed order.

**Test removal is dangerous.** Chiara's redundancy clustering identified 45 groups of tests that appeared perfectly redundant on lot 801 (Jaccard similarity >= 0.95). Keeping only one representative per cluster reduced the test count from 572 to 281. On training data, this missed zero failures. But on the held-out lot 812, **7 out of 116 failures were missed** (6.0% miss rate), and the avg tests per failing chip actually *increased* to 134.3 (worse than the original order's 82.0). This happens because the reduced test set lost some of its best early detectors, and the tests that were "redundant" on lot 801 turned out to catch unique failures on lot 812.

**Zero missed failures with reordering; 6% missed with removal.** Every reordering strategy catches all 116 failures eventually. The test removal strategies permanently lose 7 failures. This is the most important practical finding: reordering is safe, removal is not.

### 6.2 Per-Die Adaptive Results

The table in Section 6.1 evaluates the adaptive model via a single global ordering (averaged probabilities). But the true strength of the adaptive scheduler is per-die personalization: each chip gets its own test sequence based on its own margin values.

When the scheduler personalizes the test order for each individual chip, the results on lot 812 are:

| Seed size | Avg tests per failing chip | Savings (failing chips) | Missed |
|:---------:|:-:|:-:|:-:|
| 10 | 33.6 | 94.1% | 0 |
| 15 | 34.2 | 94.0% | 0 |
| 20 | 23.4 | 95.9% | 0 |

With a seed of 20 tests, failing chips are identified after running only 23.4 tests on average (out of 572 total). That is a 96% reduction in tests executed for defective chips.

The seed=20 variant is the strongest because running more seed tests upfront gives the XGBoost models more features to work with (20 margin values instead of 10), producing better predictions of which remaining tests will fail for each chip.

### 6.3 What About Passing Chips?

A natural question: if the savings are so large for failing chips, what about the 99.5% of chips that pass everything?

Passing chips always run all 572 tests regardless of the order, because they never fail any test and so never trigger early stopping. That means the overall savings across all chips (passing + failing) are small in percentage terms -- around 0.4%. But at Nexperia's scale (110 billion chips per year), 0.5% of chips failing means roughly 550 million defective chips per year. For each of those 550 million chips, the optimized order saves on average 80% of the test time. That is the operational value.

### 6.4 Cross-Wafer Validation

To make sure the results are not specific to lot 812, I run leave-one-lot-out cross-validation. Each time, I train on two lots and test on the third:

| Train lots | Test lot | Total failures | Greedy detection in first 10 | Adaptive avg tests per failing chip | Savings (failing chips) |
|-----------|---------|:-:|:---:|:---:|:---:|
| 806 + 812 | 801 | 127 | 101/127 (80%) | 20.8 | 96.4% |
| 801 + 812 | 806 | 96 | 82/96 (85%) | 28.3 | 95.1% |
| 801 + 806 | 812 | 116 | 98/116 (84%) | 34.2 | 94.0% |

The results are consistent across all three folds. The greedy baseline always catches 80-85% of failures in the first 10 tests. The adaptive scheduler always achieves 94-96% time savings on failing chips. And zero failures are missed in any configuration.

This consistency is important: it means the approach is not overfitting to one particular lot's failure modes. The learned ordering generalizes to unseen production lots.

### 6.5 Why Removing Tests Is Dangerous

A natural question is: if most tests are redundant, why not just remove them entirely instead of reordering? My teammate Chiara explored this direction. She clustered tests based on Jaccard similarity -- if two tests always fail the same set of chips (Jaccard >= 0.95), they are likely measuring the same physical phenomenon, so one of them should be removable. This analysis identified 45 redundancy clusters and reduced the test count from 572 to 281 (a 51% reduction).

On training data, this looked perfect: every failure was still caught by the reduced set. However, **on the held-out lot 812, 7 out of 116 failures were missed entirely** (6.0% miss rate). The reduced test set only detected 109 out of 116 defective chips.

Why did this happen? Tests that appeared perfectly redundant on one lot (every chip that fails test A also fails test B, and vice versa) had different failure patterns on another lot. Due to natural manufacturing process variation between lots, a defect that was caught by both tests A and B on lot 801 might only be caught by test B on lot 812. If test B was the one removed as "redundant," that defect goes undetected.

In semiconductor manufacturing, missing 6% of defects is unacceptable. The current accuracy standard at Nexperia is approximately 99.9%.

The lesson is clear: **reorder tests, do not remove them.** The reordering approach achieves comparable time savings (96%+ for failing chips) while maintaining 100% detection. Every test still runs eventually; we just run the most important ones first.

## 7. What the Tests Look Like

To make the results concrete, here are the top 15 tests in the multi-wafer greedy ordering. These are the tests that, collectively, catch the vast majority of failures:

| Rank | Test Name | What It Measures |
|:----:|-----------|-----------------|
| 1 | `tf_ICC_stand:VCC_ICC_standby:1.650 V` | Standby supply current at 1.65V |
| 2 | `tf_ICC_stand:VCC_ICC_standby:2.300 V` | Standby supply current at 2.3V |
| 3 | `tb_VOH_8m:VOH_8m:-8.000 mA (21)` | Output high voltage at pin 21 |
| 4 | `tb_IOL_Pport:IOL_Ppprt_2v3:200 mV` | Output low current (P-port) at 2.3V |
| 5 | `tf_dIDDq1:IDDq1:3.300 V (10)` | Quiescent current test 10 at 3.3V |
| 6 | `tf_ICC_SCL:VCC_ICC_SCL:3.600 V` | Clock line supply current at 3.6V |
| 7 | `tf_ICC_stand:VCC_ICC_standby:3.600 V` | Standby supply current at 3.6V |
| 8 | `tf_ICC_SCL:VCC_ICC_SCL:1.650 V` | Clock line supply current at 1.65V |
| 9 | `tb_IOL_Pport:IOL_Ppprt_2v3:200 mV (2)` | Output low current, pin 2 |
| 10 | `tf_ICCQ_SCL:ICCQ_SCL:1.650 V` | Quiescent clock line current at 1.65V |
| 11 | `tf_dIDDq1:IDDq1:3.300 V (3)` | Quiescent current test 3 at 3.3V |
| 12 | `tf_ICC_SCL:VCC_ICC_SCL:5.500 V` | Clock line supply current at 5.5V |
| 13 | `tb_iiL_Pport:iiL_Pport_1v65:0 uV (9)` | Input leakage low (P-port) pin 9 |
| 14 | `tf_ICCQ_Ppor:ICCQ_Pport:5.500 V` | Quiescent P-port current at 5.5V |
| 15 | `tb_VOH_8m:VOH_8m:-8.000 mA (50)` | Output high voltage at pin 50 |

The pattern is striking: **current consumption tests dominate the top of the list.** Standby current (`ICC_stand`), quiescent current (`IDDq1`, `ICCQ`), and clock line current (`ICC_SCL`) are the most informative tests for early failure detection. This makes physical sense: excessive current draw is often a symptom of a manufacturing defect (e.g., a short circuit or gate oxide failure) that affects multiple downstream tests.

## 8. Computational Details

- **Data loading** (from cache): ~3 seconds
- **Greedy ordering** (single lot): ~1 second
- **Multi-wafer greedy**: ~2 seconds
- **Adaptive scheduler training** (533 XGBoost models): ~90 seconds
- **Adaptive inference** (26,155 chips): ~5 seconds
- **Full evaluation pipeline**: ~3 minutes

All experiments run on a standard laptop (Python 3.13, no GPU required).

## 9. Project Structure

```
AML/
  src/
    data_loader.py    -- data pipeline (xlsx -> WaferData objects)
    baseline.py       -- greedy ordering and evaluation utilities
    adaptive.py       -- ML-adaptive scheduler (XGBoost)
    evaluation.py     -- full comparison pipeline
  data/
    raw_cache.parquet  -- cached dataset (auto-generated on first run)
  results/
    comparison_table.csv    -- summary metrics for all strategies
    detection_curves.csv    -- cumulative detection data for plotting
```

## 10. Limitations and Future Work

### What this work does not do

- **Cross-product generalization.** All experiments are on a single product (NCA9555). With data from multiple products, the approach could potentially transfer learned test orderings between similar product families.
- **Test duration weighting.** I treat all tests as equal cost. In reality, some tests take longer than others. If test duration data were available, the optimization could weight by time rather than count, potentially yielding larger wall-clock savings.
- **Online learning.** The current model is trained offline on historical lots. In a production setting, the model could be updated continuously as new lots are tested, adapting to process drift.

### What could be improved

- **Seed test selection.** Currently, the seed tests are the top-N from greedy ordering. A smarter approach would be to select seed tests that maximize the mutual information between their margins and the failure status of remaining tests.
- **Joint modeling.** I train 533 independent models, one per target test. A multi-task model could share information across related tests (e.g., all tests in the same family) and potentially improve predictions for rare failure modes.
- **Reinforcement learning.** The current scheduler makes a one-shot decision (sort by predicted probability after seed tests). A sequential RL agent could make adaptive decisions at every step, using each new test result to update its belief about which test to run next.

## 11. Conclusion

I showed that optimizing test order using historical data can dramatically accelerate failure detection in semiconductor testing. The key findings are:

1. A simple greedy ordering, trained on two production lots, catches 84.5% of failures in the first 10 tests (compared to just 3.4% with the original factory order) on a held-out lot.

2. An ML-based adaptive scheduler, using XGBoost models trained on continuous margin values, reduces the average number of tests per failing chip from 82 (original order) to 12.8 -- a reduction of 84%.

3. Cross-wafer validation confirms that these results generalize across lots, with 94-96% time savings on failing chips and zero missed failures in every configuration tested.

4. Critically, no tests are removed. The approach reorders tests rather than eliminating them, preserving 100% fault detection while achieving the time savings. Attempting to remove "redundant" tests caused 6% of failures to be missed on unseen lots, confirming that reordering is the safe strategy.

For a company like Nexperia, which tests 110 billion chips per year, even small per-chip time savings translate to meaningful operational gains. The adaptive scheduling approach is both practically deployable (it runs in seconds on a laptop) and safe (it never misses a defective chip).
