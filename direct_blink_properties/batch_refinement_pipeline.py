"""
Batch Blink‑Signal Refinement Evaluation & Reporting Pipeline
============================================================

This script extends the original *blink_refiners* evaluation set‑up with three
major capabilities that are usually executed together:

1. **Per‑Technique Visual QA**
   * For every `.fif` × *refinement technique* run during benchmarking a full
     *epoch‑by‑epoch* quality‑assurance report is created.
   * Each epoch figure contains **top:** original EAR trace and **bottom:** an
     overlay of the original (gray) + corrected (blue) signals.
   * A ±10‑frame (≈ ⅓ s @ 30 fps) context window is added before/after each blink.
   * Reports are stored as HTML files via :class:`mne.Report`.  If more than
     40 figures are required, the output is split into numbered batches to keep
     file sizes reasonable.

2. **Recursive Batch Processing**
   * :pyfunc:`process_all_fif_zip_pairs` automatically matches **all** `.fif`
     recordings inside *fif_dir* with their corresponding annotation archives
     located somewhere under *zip_dir* (potentially deeply nested).
   * For every matched pair the full blink‑refinement grid search is executed
     (re‑using the existing caching logic so that already‑computed results are
     loaded instantly).

3. **Cross‑Subject Summary Statistics**
   * Once all pairs have been processed a compact :class:`pandas.DataFrame`
     summarises the mean ± sd of the three core performance metrics for each
     method and also counts how often a method ranked **1st**.

The script purposefully avoids *argparse* to keep usage minimal.  Simply edit
``FIF_DIR`` and ``ZIP_DIR`` near the bottom and run the file directly.

Example
-------
    $ python batch_refinement_pipeline.py

Dependencies
------------
* direct_blink_properties (local package with util & refiners)
* pyblinkers               (zero‑crossing; upstream dependency)
* MNE‑Python ≥1.6          (for :class:`mne.Report`)
* matplotlib, numpy, pandas, tqdm

"""
from __future__ import annotations

import logging
import math
import pickle
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Third‑party / local imports – keep guarded so the file can be imported even
# when the heavy deps are missing while still enabling static analysis tools.
# -----------------------------------------------------------------------------

from direct_blink_properties.util import (
        load_fif_and_annotations,
        extract_blink_durations)
from direct_blink_properties.ear_blink_transformers import (
        get_refiners,
        get_grids,
        run_experiments,
        _cache_load)


# -----------------------------------------------------------------------------
# Configuration & logging helpers
# -----------------------------------------------------------------------------

def _configure_logger(level: int = logging.INFO) -> logging.Logger:
    """Return the pipeline‑wide *root* logger with a sensible formatter."""
    logger = logging.getLogger("blink.batch")
    logger.setLevel(level)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        logger.addHandler(_h)
    return logger


logger = _configure_logger()

# -----------------------------------------------------------------------------
# 1) Visualisation helpers – one figure per blink epoch
# -----------------------------------------------------------------------------

def _epoch_figure(
        original: np.ndarray,
        refined: np.ndarray,
        start: int,
        end: int,
        sfreq: float,
        idx: int,
        method: str,
) -> plt.Figure:
    """Return a 2‑row figure for the blink epoch *idx* (sample indices start:end)."""
    t = np.arange(start, end) / sfreq
    orig_seg = original[start:end]
    ref_seg = refined[start:end]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.suptitle(f"{method} – Blink {idx} ({start}‑{end} samples)")

    ax0.plot(t, orig_seg)
    ax0.set_ylabel("EAR")
    ax0.set_title("Original signal")

    ax1.plot(t, orig_seg, alpha=0.4, label="Original")
    ax1.plot(t, ref_seg, label="Refined")
    ax1.set_ylabel("EAR")
    ax1.set_title("Overlay")
    ax1.legend(loc="best")

    ax1.set_xlabel("Time [s]")
    plt.tight_layout()
    return fig


def _create_epoch_reports(
        raw_signal: np.ndarray,
        refined_signal: np.ndarray,
        df_metrics: pd.DataFrame,
        sfreq: float,
        method: str,
        fif_stem: str,
        out_root: Path,
        pad_frames: int = 10,
        batch_size: int = 40,
):
    """Generate one or multiple MNE reports for *method* showing each blink epoch.

    Parameters
    ----------
    raw_signal
        Original EAR trace.
    refined_signal
        Signal after applying the refinement method.
    df_metrics
        Per‑epoch metrics table carrying *startBlinks* & *endBlinks* columns.
    sfreq
        Sampling frequency of *raw_signal*.
    method
        Name of the refinement technique.
    fif_stem
        Base name of the original recording (used in filenames).
    out_root
        Directory to place the generated reports (created if necessary).
    pad_frames
        Extra samples to prepend/append around each blink interval.
    batch_size
        Maximum number of figures per individual HTML file.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    epochs = df_metrics.sort_values("startBlinks")
    total_epochs = len(epochs)

    logger.info(
        "Generating %d figures for %s | %s (batch=%d)…",
        total_epochs,
        fif_stem,
        method,
        batch_size,
    )

    for batch_idx, start_row in enumerate(range(0, total_epochs, batch_size)):
        rep = mne.Report(title=f"{fif_stem} – {method} (batch {batch_idx + 1})")

        subset = epochs.iloc[start_row : start_row + batch_size]
        for i, row in subset.iterrows():
            beg = max(0, int(row["startBlinks"]) - pad_frames)
            end = int(row["endBlinks"]) + pad_frames
            fig = _epoch_figure(
                raw_signal,
                refined_signal,
                beg,
                end,
                sfreq,
                idx=i,
                method=method,
            )
            rep.add_figure(fig, title=f"blink_{i:03d}")
            plt.close(fig)

        fname = (
                out_root
                / f"{fif_stem}_{method}_b{batch_idx + 1:02d}.html"
        )
        rep.save(fname, overwrite=True, open_browser=False)
        logger.info("Saved report → %s", fname)


# -----------------------------------------------------------------------------
# 2) Batch processing of FIF × ZIP pairs
# -----------------------------------------------------------------------------

def process_all_fif_zip_pairs(fif_dir: str | Path, zip_dir: str | Path):
    """Run the full blink‑refinement experimentation on every matched file pair.

    Parameters
    ----------
    fif_dir
        Directory containing *only* `.fif` files at its root level.
    zip_dir
        Root directory that may hold annotation `.zip`s in arbitrary sub‑folders.

    Returns
    -------
    List[pd.DataFrame]
        A list of per‑recording leaderboards (one DataFrame per processed pair).
    """
    # fif_dir_p = Path(fif_dir)
    # zip_dir_p = Path(zip_dir)

    # fif_files = list(fif_dir_p.glob("*.fif"))
    # zip_files = list(zip_dir_p.rglob("*.zip"))
    fif_files=[r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1\S01_20170519_043933.fif',
               r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1\S01_20170519_043933_2.fif',
               # r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1'
                ]
    zip_files=[r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final\S1\from_cvat\S01_20170519_043933.zip',
               r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final\S1\from_cvat\S01_20170519_043933_2.zip'
    ]
    logger.info("Found %d FIF files and %d ZIP files", len(fif_files), len(zip_files))

    all_leaderboards: List[pd.DataFrame] = []

    refiners = get_refiners()
    # Will build each grid lazily once we know *sfreq*.

    for fif_path in fif_files:
        base = fif_path.stem
        matched_zips = [z for z in zip_files if z.stem.startswith(base)]

        if not matched_zips:
            logger.warning("No ZIPs found for %s", base)
            continue

        for zip_path in matched_zips:
            logger.info("Processing %s + %s", fif_path.name, zip_path.name)
            try:
                raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)
                sfreq = raw.info["sfreq"]
                blink_df = extract_blink_durations(annotation_df, frame_offset=5, sfreq=sfreq, video_fps=30)
                signal = raw.get_data(picks=1)[0]  # EAR trace
                params = {"sfreq": sfreq}

                grids = get_grids(blink_df, sfreq)

                leaderboard = run_experiments(signal, blink_df, params, refiners, grids)
                all_leaderboards.append(leaderboard.assign(recording=base))

                logger.debug("Top‑3 methods:\n%s", leaderboard.head(3).to_string(index=False))

                # ------------------------------------------------------------------
                # Optional visual QA – disabled by default for speed.  Toggle using
                # the constant below or set an env‑var / config as you like.
                # ------------------------------------------------------------------
                if _ENABLE_REPORTS:
                    out_rep_dir = fif_path.parent / "reports"
                    for method in leaderboard["method"].unique():
                        # Locate *some* parameter set for the method (pull first
                        # param dictionary from the refiner grid).
                        param_dict = grids[method][0]
                        refined = refiners[method](signal.copy(), **param_dict)

                        # Use cached metrics belonging to this (method, param) pair;
                        # if missing, skip reporting for that combo.
                        metrics = _cache_load(method, param_dict)
                        if metrics is None:
                            logger.debug("No cached result for %s %s – skipping report", method, param_dict)
                            continue

                        _create_epoch_reports(
                            raw_signal=signal,
                            refined_signal=refined,
                            df_metrics=metrics.df,
                            sfreq=sfreq,
                            method=method,
                            fif_stem=base,
                            out_root=out_rep_dir,
                        )

            except Exception as e:
                logger.exception("Error processing %s + %s: %s", fif_path.name, zip_path.name, e)

    return all_leaderboards


# -----------------------------------------------------------------------------
# 3) Aggregated summary statistics
# -----------------------------------------------------------------------------

def summarize_method_performance(leaderboards: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate metrics across all leaderboards returned by *run_experiments*.

    The function respects the original leaderboard sorting hierarchy when
    determining the *best‑ranked* method per recording.
    """
    if not leaderboards:
        raise ValueError("No leaderboards supplied — nothing to summarise.")

    concat = pd.concat(leaderboards, ignore_index=True)

    # Compute descriptive stats per method
    stats = (
        concat.groupby("method")[
            [
                "kept_ratio",
                "positive_peak_fraction",
                "median_peak_amp",
            ]
        ]
        .agg(["mean", "std"])
        .droplevel(0, axis=1)
        .reset_index()
    )

    # Count how many times a method ranked #1 (first row per individual board)
    best_counts = (
        pd.concat(
            [lb.iloc[[0]] for lb in leaderboards], ignore_index=True
        )["method"].value_counts()
    )
    stats["best_count"] = stats["method"].map(best_counts).fillna(0).astype(int)

    # Re‑order columns for readability
    stats = stats[
        [
            "method",
            "kept_ratio_mean",
            "kept_ratio_std",
            "positive_peak_fraction_mean",
            "positive_peak_fraction_std",
            "median_peak_amp_mean",
            "median_peak_amp_std",
            "best_count",
        ]
    ]

    return stats.sort_values("best_count", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# -------------------------------  Main entry  ---------------------------------
# -----------------------------------------------------------------------------
_ENABLE_REPORTS = True  # set False to skip heavy HTML generation

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Hard‑coded data locations – change these to your actual directories.
    # ------------------------------------------------------------------
    FIF_DIR = Path(r"C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final\S1\from_cvat")
    ZIP_DIR = Path(r"C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1")

    logger.info("Starting batch refinement pipeline…")

    all_lb = process_all_fif_zip_pairs(FIF_DIR, ZIP_DIR)

    if not all_lb:
        logger.error("No leaderboards produced — aborting summary step.")
    else:
        summary = summarize_method_performance(all_lb)
        pad = " " * 4
        print("\n" + pad + "Aggregated method performance across all recordings:\n")
        print(textwrap.indent(summary.to_string(index=False, float_format="%.3f"), pad))

        # Optionally write to CSV for easy downstream analysis.
        summary.to_csv("method_performance_summary.csv", index=False)
        logger.info("Saved summary → method_performance_summary.csv")
