# End-to-End Blink Refinement and Validation
#
# This notebook demonstrates the complete workflow of loading a raw EOG signal, refining blink annotations, and validating the results against a ground truth. We will use the `prepare_refined_segments` function, which encapsulates the entire process.

from pathlib import Path
import pandas as pd
from pyblinkers.utils import prepare_refined_segments
from pyblinkers.pipeline import extract_features

# Get the project root directory
# Note: This path might need adjustment depending on your project structure.
# Assuming the script is run from a directory where the parent is the project root.
PROJECT_ROOT = Path().resolve().parent

# Define file paths
RAW_FILE = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"
GROUND_TRUTH_FILE = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_blink_count_epoch.csv"

# Load ground truth data
try:
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
except FileNotFoundError:
    print(f"Error: Ground truth file not found at {GROUND_TRUTH_FILE}")
    print("Please ensure the file path is correct and the necessary data is available.")
    # Exit or handle the error as appropriate
    exit()


# ## The `prepare_refined_segments` function
#
# The `prepare_refined_segments` function is a high-level utility that handles the entire blink refinement process. Here’s a breakdown of what it does behind the scenes:
#
# 1.  **Loads the Raw Data:** If you provide a file path, it loads the `mne.io.Raw` object.
# 2.  **Slices into Epochs:** It divides the continuous recording into 30-second segments (or epochs).
# 3.  **Refines Blink Timings:** For each segment, it identifies the precise start, peak, and end of every blink using the `refine_blinks_from_epochs` function. This is the core of the refinement process, where it analyzes the signal to find the exact moments of eye closure and opening.
# 4.  **Updates Annotations:** Finally, it replaces the original, rough annotations in each segment with the new, precise ones.
#
# The function returns two things: a list of the processed `mne.io.Raw` segments and a list of dictionaries containing the detailed refined blink information.

print("Refining blink segments...")
try:
    segments, refined_blinks = prepare_refined_segments(
        RAW_FILE,
        channel="EOG-EEG-eog_vert_left",
        keep_epoch_signal=True,
    )
except FileNotFoundError:
    print(f"Error: Raw data file not found at {RAW_FILE}")
    print("Please ensure the file path is correct and the necessary data is available.")
    exit()


# ## Validating the Results
#
# Now that we have the refined segments, we can count the blinks in each one and compare the counts to our ground truth data. This allows us to verify that the refinement process correctly identified all blinks.

# Count blinks in each segment
refined_counts = [len(segment.annotations) for segment in segments]

# Create a DataFrame for comparison
validation_df = pd.DataFrame({
    'Epoch': ground_truth['epoch_id'],
    'Ground Truth Blinks': ground_truth['blink_count'],
    'Refined Blinks': refined_counts[:len(ground_truth)]
})

# Check if the counts match
validation_df['Match'] = validation_df['Ground Truth Blinks'] == validation_df['Refined Blinks']

print("\nValidation Results:")
# In a script, use print() instead of display()
print(validation_df)


# ## Known Limitation: Boundary-Spanning Blinks
#
# As you can see in the validation table, the blink counts for epochs 31 and 55 do not match the ground truth. This is a known limitation in the current version of the processing pipeline.
#
# The discrepancy occurs because a single blink annotation can sometimes span across the boundary of two consecutive 30-second segments. For example, a blink might start at 29.9 seconds in epoch 31 and end at 30.1 seconds in epoch 32. The current refinement logic does not yet handle this specific edge case correctly, leading to an inaccurate count in the affected epochs.
#
# **TODO:** Future work will address this by implementing a mechanism to merge or correctly attribute blinks that cross epoch boundaries.


# ### Visualizing the Discrepancy
#
# To better understand the issue, let's plot the EOG signal for the problematic epochs (31 and 55) and their subsequent epochs (32 and 56). We will overlay the refined blink annotations to see exactly where the refinement process is placing the blink markers.
#
# (Note: Plotting code would be added here to visualize the EOG signal and annotations)


# ### Extracting Features for Each Segment
#
# With the refined blink annotations we can compute blink metrics for each 30-second segment. Calling `extract_features` without specifying a `features` list calculates every available feature group. We also pass `raw_segments=segments` so interval-based metrics can be generated.

print("\nExtracting features from refined blinks...")
sfreq = segments[0].info['sfreq']
epoch_len = 30.0
n_epochs = len(segments)

df = extract_features(refined_blinks, sfreq, epoch_len, n_epochs, raw_segments=segments)
df_with_gt = df.copy()
df_with_gt.insert(1, "Ground Truth Blinks", ground_truth["blink_count"])
df_with_gt.insert(2, "Match", df_with_gt["blink_count"] == ground_truth["blink_count"])

print("\nCombined Features DataFrame (first 5 rows):")
print(df_with_gt.head())


# The DataFrame shows one row per epoch with columns such as `blink_count` and various morphological or frequency metrics. Summing `blink_count` should match the total number of blinks in the ground truth CSV, providing a quick validation of the extracted features.

print(f"\nTotal blinks from ground truth: {ground_truth['blink_count'].sum()}")
print(f"Total blinks from extracted features: {df['blink_count'].sum()}")