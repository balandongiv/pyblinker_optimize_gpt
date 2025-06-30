
---

### 🧠 Instruction: Extend and Enhance the Blink Signal Refinement Evaluation Pipeline

You are tasked with improving the existing blink signal refinement evaluation codebase, focusing on three key goals:

lets name the python file as batch_refinement_pipeline.py

you must use
if __name__ == "__main__":

do not use argparse, i will harcoded  the folders directory

add detail docstring
---

### ✅ **1. Add Visualization + Reporting Per Technique**

**Objective**: Enable visual inspection of blink correction performance *epoch-by-epoch* for each refinement method.

**Requirements**:

* Implement a new function to:

    * Load the cached `.pkl` results (must include DataFrame with: `["leftZero", "rightZero", "blinkmin", "startBlinks", "endBlinks"]`)
    * For **each ****`.fif`**** file** and associated **refinement technique**, generate a **MNE report**:

        * Each figure shows **top**: original signal, and **bottom**: original + corrected signals overlapped.
        * Add **10-frame padding** before `startBlinks` and after `endBlinks` when plotting.
        * Store **all figures** into a single HTML file per `.fif` file × technique.
        * Use `mne.Report` to manage report creation and saving.
        * If there are more 40 images, each of the mne report should contain only maximum 40 images per html, create batches of mne report
* Include a `logger` to trace steps, such as caching, plotting, and report generation.

---


### ✅ **2. Batch Processing: Auto-Match `.fif` and `.zip` Files Across Nested Folders**

**Objective**: Automatically scan and match `.fif` files with all compatible `.zip` annotation files across nested participant folders, then run the full blink signal refinement evaluation pipeline.

---

### 🗂️ Folder Structure

* **FIF directory example**:

  ```
  /data/fif_files/
    S01_20170519_043933.fif
    TEST_20170601_042544.fif
  ```

* **ZIP directory example**:

  ```
  /data/zip_files/
    S1/from_cvat/S01_20170519_043933.zip
    S1/from_cvat/S01_20170519_043933_2.zip
    S1/from_cvat/S01_20170519_043933_3.zip
    S2/from_cvat/TEST_20170601_042544.zip
    S2/from_cvat/TEST_20170601_042544_2.zip
  ```

---

### 🔁 Required Function: `process_all_fif_zip_pairs(fif_dir: str, zip_dir: str)`

**Steps**:

1. **Scan FIF directory** for `.fif` files. Extract base names (e.g. `S01_20170519_043933`).
2. **Recursively scan ZIP directory** for `.zip` files.
3. For each `.fif` file:

    * Match all `.zip` files whose name starts with the base `.fif` name (e.g. match `S01_20170519_043933*.zip`).
    * For each `.zip` match:

        * Call:

          ```python
          raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)
          sfreq = raw.info["sfreq"]
          blink_df = extract_blink_durations(annotation_df, frame_offset=5, sfreq=sfreq, video_fps=30)
          signal = raw.get_data(picks=1)[0]
          params = {"sfreq": sfreq}
          ```
        * Call `get_refiners()` and `get_grids(blink_df, sfreq)`
        * Call `run_experiments()` to evaluate all techniques
        * (Optional) Generate visual reports here (Point #1)

---

### 🛠 Function Signature Example

```python
from pathlib import Path
from direct_blink_properties.util import load_fif_and_annotations, extract_blink_durations

def process_all_fif_zip_pairs(fif_dir: str, zip_dir: str):
    fif_dir = Path(fif_dir)
    zip_dir = Path(zip_dir)

    fif_files = list(fif_dir.glob("*.fif"))
    zip_files = list(zip_dir.rglob("*.zip"))

    for fif_path in fif_files:
        base = fif_path.stem  # e.g., S01_20170519_043933
        matched_zips = [z for z in zip_files if z.stem.startswith(base)]

        if not matched_zips:
            print(f"⚠️ No ZIPs found for {base}")
            continue

        for zip_path in matched_zips:
            print(f"🔄 Processing: {fif_path.name} + {zip_path.name}")
            try:
                raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)
                sfreq = raw.info["sfreq"]
                blink_df = extract_blink_durations(annotation_df, frame_offset=5, sfreq=sfreq, video_fps=30)
                signal = raw.get_data(picks=1)[0]
                params = {"sfreq": sfreq}

                refiners = get_refiners()
                grids = get_grids(blink_df, sfreq)
                leaderboard = run_experiments(signal, blink_df, params, refiners, grids)

                print(leaderboard.head(3))  # Preview top results
                # Call report generator here if needed

            except Exception as e:
                print(f"❌ Error processing {fif_path.name} + {zip_path.name}: {e}")
```

---


### ✅ **3. Generate Summary Statistics Across Subjects**

**Objective**: After evaluating all `.fif + .zip` pairs, produce an aggregated summary of method performance across the entire dataset (all subjects/sessions).

---

### 📊 Required Output

A single `pandas.DataFrame` containing, for each method:
- Mean and std of:
    - `kept_ratio`
    - `positive_peak_fraction`
    - `median_peak_amp`
- Count of how often the method ranked **best** across all files (based on sorting logic).

---


---

### 🧩 Integration

Modify your batch loop (`process_all_fif_zip_pairs`) to store the leaderboards:

```python
all_results = []

for zip_path in matched_zips:
    ...
    leaderboard = run_experiments(...)
    all_results.append(leaderboard)
```

Then, call:

```python
summary_df = summarize_method_performance(all_results)
print(summary_df.to_string(index=False, float_format="%.3f"))
```

---
---

### 🛠 Notes and Constraints

* Use `matplotlib` for plotting.
* Use `mne.Report` to consolidate figures per file/method.
* Use structured logging throughout for reproducibility.
* Reuse/refactor utilities from `viz.py` and `blink_refiners.py`.
* Padding in plots: add ±10 frames around each blink interval.
* Reuse cache logic in `.cache/*.pkl` to avoid redundant computation.

---


this is the code for ear_blink_transformer.py

"""
blink_refiners.py
=================

This module defines a suite of signal refinement functions tailored to blink detection
in eye-aspect-ratio (EAR) time-series. Each function transforms the raw EAR signal
to enhance blink epoch detection by emphasizing positive peaks and reducing baseline noise.

====================
Capabilities
------------

**Signal Refinement**
All refiners produce a transformed signal of the same length as the input, with the
goal of making each blink interval produce a distinct **positive peak**. Some methods
are global, others operate on local epochs.

**Caching**
Results are automatically cached in `.cache/*.pkl` using a stable MD5 hash of the method
name and parameter dictionary. When re-running, results are silently loaded unless
inputs or configurations have changed.

**Metrics & Ranking**
Each refinement is evaluated using ground-truth blink epochs and scored on:

- `kept_ratio`: Proportion of blinks retained after filtering invalid zero-crossings.
- `positive_peak_fraction`: Fraction of blink intervals where the refined signal has a max > 0.
- `median_peak_amp`: Median amplitude of the blink peak across valid epochs.

These metrics are computed in `_evaluate()` and stored in `Metrics`. A leaderboard is
returned as a ranked `DataFrame` (higher = better), sorted by:
kept_ratio → positive_peak_fraction → median_peak_amp

**Structured Logging**
Pipeline progress and evaluations are tracked using Python’s `logging` module.
Verbose results can be enabled by increasing the logging level (e.g., DEBUG).

=================
Signal‑refinement strategies Overview
---------------
Each method below corresponds to a registered refiner. Most support parameter grids
for exhaustive benchmarking.

Basic Methods:
--------------
- `global_mean`: Subtracts the global mean and flips sign to enforce positive peaks.
- `local_mean`: Subtracts the mean within each blink epoch (requires `epochs`).
- `detrend_linear`: Removes a best-fit linear trend from the entire signal.
- `detrend_poly`: Polynomial trend removal using least-squares (orders 2–4).
- `savgol`: Savitzky-Golay filter for local polynomial baseline smoothing.
- `robust_scale`: Applies robust scaling using median-centered normalization.

Filters:
--------
- `highpass`: Butterworth high-pass filter (e.g. cutoff 0.2 Hz).
- `highpass_iir`: IIR Chebyshev-II high-pass filter (zero-phase).
- `bandpass`: Butterworth band-pass filter (e.g. 0.1–10 Hz).
- `firhp`: Kaiser-window FIR high-pass filter.

Smoothing / Trend Removal:
--------------------------
- `als`: Asymmetric Least Squares baseline correction (configurable `lam` and `p`).
- `airpls`: Adaptive iteratively reweighted penalized least squares (PLS).
- `whittaker`: Smoother using Whittaker penalized least squares.
- `medfilt`: Rolling median filter.
- `quantile`: Rolling quantile baseline subtraction (e.g. 5th percentile).
- `loess`: Locally weighted scatterplot smoothing (LOWESS).
- `zscore`: Rolling z-score normalization.
- `tophat`: Morphological top-hat transform using opening.
- `morph`: Morphological opening followed by closing.

Decomposition-Based:
--------------------
- `wavelet`: Multi-resolution wavelet transform (removes approximation coefficients).
- `emd`: Empirical Mode Decomposition – sum of early intrinsic mode functions (IMFs).
- `ceemdan`: CEEMDAN (noise-assisted EMD) – sum of selected IMFs.
- `vmd`: Variational Mode Decomposition – sum of dominant modes.

Epoch-Aware:
------------
- `epoch_poly`: Polynomial detrending applied per epoch with boundary padding.
- `quantile_hp`: Cascade of rolling quantile removal + high-pass filter.
- `quantile_hp_flip`: Same as `quantile_hp`, but with epoch-wise peak polarity correction.

Meta-Techniques:
- `_ensure_peaks_positive`: Ensures max > 0 in global signal.
- `apply_epoch_flip`: Ensures each blink epoch has a positive peak.


Each method ensures output signal length matches the input and guarantees positive blink peaks.
The `run_experiments()` function benchmarks these over multiple configurations with metrics:
`kept_ratio`, `positive_peak_fraction`, and `median_peak_amp`.

See Also:
---------
- `_ensure_peaks_positive`: Utility to flip signal globally if needed.
- `apply_epoch_flip`: Epoch-level polarity enforcement.
- `left_right_zero_crossing`: Core detection logic (unchanged from base code).
- `_evaluate`: Core metric computation.

The core blink‑segmentation routine `left_right_zero_crossing` is *copied verbatim*
from your code and must remain untouched.

Usage example
-------------
# >>> from blink_signal_refinement import run_experiments
# >>> leaderboard = run_experiments(raw_signal, blink_df, params)
# >>> print(leaderboard.head())

You can also run the module directly (`python blink_signal_refinement.py`) – it will
create a synthetic demo signal if no FIF/zip path is supplied.

==============================
Choosing the Best Method

The leaderboard DataFrame ranks each refiner configuration using the formula:
sort by (kept_ratio DESC, positive_peak_fraction DESC, median_peak_amp DESC)

This means that the method which retains the most blink intervals with valid zero-crossings and
ensures strong positive peaks will appear at the top.

The top-ranked row is typically the most robust method, but you may also inspect
the trade-off between kept_ratio and median_peak_amp if your application prefers
signal clarity over detection breadth.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable, Dict

from tqdm import tqdm

from direct_blink_properties.signal_refinement import *
from direct_blink_properties.util import _hash, Metrics, get_grids, get_refiners, ExperimentResult
from pyblinkers.zero_crossing import left_right_zero_crossing

logger = logging.getLogger("blink")
logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG) # Instead of logging.INFO
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_handler)

_CACHE_DIR = Path(".cache")
_CACHE_DIR.mkdir(exist_ok=True)



# -----------------------------------------------------------------------------
# Utility – force positive blink peaks (epoch max > 0)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Dataclass containers
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
kept_ratio: float
positive_peak_fraction: float
median_peak_amp: float
df: pd.DataFrame = field(repr=False, compare=False)

    def to_dict(self):
        # only include the numeric fields
        return {
            "kept_ratio": self.kept_ratio,
            "positive_peak_fraction": self.positive_peak_fraction,
            "median_peak_amp": self.median_peak_amp,
        }

@dataclass
class ExperimentResult:
method: str
params: Dict
metrics: Metrics

    def to_row(self):
        d = self.metrics.to_dict()
        d.update(method=self.method)
        return d


def _cache_load(method: str, params: Dict) -> Metrics | None:
key = f"{method}_{_hash(params)}"
path = _CACHE_DIR / f"{key}.pkl"
if path.exists():
metrics: Metrics = pickle.loads(path.read_bytes())
logger.info(
"Loaded cached result for %s → kept=%.3f | positive=%.3f | median_amp=%.3f",
method, metrics.kept_ratio, metrics.positive_peak_fraction, metrics.median_peak_amp
)
return metrics
return None


def _cache_save(method: str, params: Dict, metrics: Metrics):
key = f"{method}_{_hash(params)}"
path = _CACHE_DIR / f"{key}.pkl"
path.write_bytes(pickle.dumps(metrics, protocol=4))
logger.debug("Saved cache → %s", path)
# -----------------------------------------------------------------------------
# Core: run one refinement + blink evaluation (with caching)
# -----------------------------------------------------------------------------

def _evaluate(refined: np.ndarray, df: pd.DataFrame,params:dict) -> Metrics:
"""Attach L/R zero‑crossings, drop invalid rows and compute metrics."""
original_len = len(df)

    tmp = df.copy()
    tmp[["leftZero", "rightZero"]] = tmp.apply(
        lambda row: left_right_zero_crossing(
            refined, row["blinkmin"], row["startBlinks"], row["endBlinks"]
        ),
        axis=1,
        result_type="expand",
    )
    tmp = tmp.dropna()
    min_gap = max(3, int(params["sfreq"] * 0.01))  # e.g. 0.01 s = 10 ms
    keep = (tmp["rightZero"] - tmp["leftZero"]) >= min_gap
    tmp = tmp[keep]
    # tmp = tmp[(tmp["rightZero"] - tmp["leftZero"]) >= 3]
    kept_len = len(tmp)

    # Metrics
    peaks = refined[tmp["blinkmin"].astype(int)]
    positive_peak_fraction = np.mean(peaks > 0) if peaks.size else 0.0
    median_peak_amp = np.median(peaks) if peaks.size else 0.0

    metrics = Metrics(
        kept_ratio=kept_len / original_len if original_len else 0.0,
        positive_peak_fraction=float(positive_peak_fraction),
        median_peak_amp=float(median_peak_amp),
        df=tmp
    )

    logger.info("Evaluation result: kept_ratio=%.3f | positive_peak_fraction=%.3f | median_peak_amp=%.3f",
                metrics.kept_ratio, metrics.positive_peak_fraction, metrics.median_peak_amp)

    return metrics




# -----------------------------------------------------------------------------
# Public API – run experiments & return leaderboard (pd.DataFrame)
# -----------------------------------------------------------------------------

def run_experiments(candidate_signal: np.ndarray,
df: pd.DataFrame,
params: Dict,
REFINERS: Dict[str, Callable],
grids: Dict[str, List[Dict]]) -> pd.DataFrame:
"""Run every registered refiner (and param grid, if any) → leaderboard."""
logger.info("Starting experiments on %d samples – %d blink intervals",
candidate_signal.size, len(df))

    joblist = [
            (method, fn, p)
            for method, fn in REFINERS.items()
            for p in grids[method]
        ]

    results: List[ExperimentResult] = []

    # create a standalone progress bar
    with tqdm(total=len(joblist), desc="Running refiners", unit="method") as pbar:
        for method, fn, p in joblist:
            try:
                # 1) load or compute
                cached = _cache_load(method, p)
                if cached is None:
                    logger.info("%s – running…", method)
                    refined = fn(candidate_signal.copy(), **p)
                    metrics = _evaluate(refined, df, params)
                    _cache_save(method, p, metrics)
                else:
                    metrics = cached

                # 2) store result
                results.append(ExperimentResult(method, p, metrics))

                # 3) update bar and postfix
                pbar.set_postfix(method=method, kept=f"{metrics.kept_ratio:.3f}")

            except Exception as e:
                logger.warning("Skipping %s due to error: %s", method, str(e))

            finally:
                pbar.update(1)


    leaderboard = pd.DataFrame([r.to_row() for r in results])
    leaderboard.sort_values(["kept_ratio", "positive_peak_fraction", "median_peak_amp"],
                            ascending=False, inplace=True)
    logger.info("Experiments finished – best method: %s",
                leaderboard.iloc[0]["method"])
    return leaderboard.reset_index(drop=True)



# -----------------------------------------------------------------------------
# CLI helper for quick manual tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
fif_path = Path(r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif")
zip_path = Path(r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip")

    # Optional parameters
    use_data_files = fif_path.exists() and zip_path.exists()

    if use_data_files:
        from direct_blink_properties.util import load_fif_and_annotations, extract_blink_durations

        raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)
        sfreq = raw.info["sfreq"]
        video_fps = 30
        frame_offset = 5
        blink_df = extract_blink_durations(annotation_df, frame_offset, sfreq, video_fps)
        signal = raw.get_data(picks=1)[0]
        params_demo = {"sfreq": sfreq}

    params_demo = {"sfreq": sfreq}
    refiners = get_refiners()
    grids = get_grids(blink_df, sfreq)
    board = run_experiments(signal, blink_df, params_demo, refiners, grids)
    print("\nLeaderboard (higher = better):")
    print(board.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
