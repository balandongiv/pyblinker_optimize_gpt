import pickle
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, savgol_filter
from scipy.ndimage import grey_opening

# --- Mocking util functions if not available ---
# Replace these with your actual imports if they are in a different structure
try:
    # Assuming your util functions are structured as you showed
    from direct_blink_properties.util import load_fif_and_annotations, extract_blink_durations
except ImportError:
    print("Warning: 'direct_blink_properties.util' not found. Using mock functions for demonstration.")
    def load_fif_and_annotations(fif_path, zip_path):
        # Mock implementation
        srate = 250.0 # Example sampling rate
        n_channels = 2
        duration_seconds = 7200 # 1.8M points / 250 Hz = 7200s
        n_samples = int(srate * duration_seconds)
        if n_samples != 1800000: # Adjust if mock doesn't match 1.8M points
            n_samples = 1800000
            duration_seconds = n_samples / srate

        data = np.random.randn(n_channels, n_samples)
        # Create a few EOG-like blinks (negative deflections)
        for i in range(50, n_samples - int(srate*2), int(n_samples / 70)): # approx 70 blinks
            blink_duration_samples = int(srate * 0.3) # 300ms blink
            data[1, i : i + blink_duration_samples // 2] -= np.linspace(0, 2, blink_duration_samples // 2)**2
            data[1, i + blink_duration_samples // 2 : i + blink_duration_samples] -= np.linspace(2, 0, blink_duration_samples // 2)**2


        raw = mne.io.RawArray(data,
                              mne.create_info(ch_names=['EEG001', 'EOG'], sfreq=srate, ch_types=['eeg', 'eog']))

        # Create mock annotations based on the injected blinks
        onsets_seconds = []
        durations_seconds = []
        descriptions = []
        for i in range(50, n_samples - int(srate*2), int(n_samples / 70)):
            onsets_seconds.append(i / srate)
            durations_seconds.append(0.3) # 300ms blink duration
            descriptions.append('blink')

        annotation_df = pd.DataFrame({
            'onset': np.array(onsets_seconds),
            'duration': np.array(durations_seconds),
            'description': descriptions
        })
        return raw, annotation_df

    def extract_blink_durations(annotation_df, frame_offset, sfreq, video_fps=None):
        # Mock implementation - This needs to generate 'startBlinks', 'endBlinks', 'blink_min' in *samples*
        blinks_data = []
        if annotation_df.empty: return pd.DataFrame()

        for _, row in annotation_df.iterrows():
            if 'blink' in row['description'].lower():
                # Convert onset/duration from seconds (MNE standard) to samples
                start_sample_abs = int(row['onset'] * sfreq)
                duration_samples_abs = int(row['duration'] * sfreq)
                end_sample_abs = start_sample_abs + duration_samples_abs

                # For mock, blink_min is middle of the annotated duration
                blink_min_sample = start_sample_abs + duration_samples_abs // 2

                # Apply frame_offset (in samples if sfreq is primary time base)
                # Assuming frame_offset is in samples for this mock
                start_blink_offset = start_sample_abs - frame_offset
                end_blink_offset = end_sample_abs + frame_offset

                blinks_data.append({
                    'startBlinks': start_blink_offset,
                    'endBlinks': end_blink_offset,
                    'blink_min': blink_min_sample
                })
        return pd.DataFrame(blinks_data)

# --- Signal Processing Strategies ---

def strategy_original_transform(ear_signal_original, **kwargs): # Add **kwargs to accept unused params
    """Applies the user's original transformation."""
    signal = -ear_signal_original.copy()
    signal = signal - np.mean(signal)
    return signal

def strategy_baseline_rolling_median(ear_signal_original, window_size_frames=150, center_final=True, **kwargs):
    """Baseline correction using rolling median."""
    if not isinstance(ear_signal_original, pd.Series):
        ear_series = pd.Series(ear_signal_original)
    else:
        ear_series = ear_signal_original.copy()

    baseline = ear_series.rolling(window=int(window_size_frames), center=True, min_periods=1).median()
    baseline = baseline.fillna(method='bfill').fillna(method='ffill').values

    ear_corrected = ear_signal_original - baseline
    processed_signal = -ear_corrected # Invert: blinks become positive
    if center_final:
        processed_signal = processed_signal - np.mean(processed_signal)
    return processed_signal

def strategy_baseline_lowpass_butterworth(ear_signal_original, sfreq, cutoff_hz=0.5, order=3, center_final=True, **kwargs):
    """Baseline correction using Butterworth low-pass filter."""
    nyq = 0.5 * sfreq
    normal_cutoff = cutoff_hz / nyq
    if normal_cutoff >= 1.0:
        print(f"Warning: Butterworth normal_cutoff ({normal_cutoff:.2f}) >= 1.0. Using raw signal as baseline (potential issue).")
        baseline = ear_signal_original.copy()
    elif normal_cutoff <= 0:
        print(f"Warning: Butterworth normal_cutoff ({normal_cutoff:.2f}) <= 0. Using raw signal as baseline (potential issue).")
        baseline = ear_signal_original.copy()
    else:
        sos = butter(N=order, Wn=normal_cutoff, btype='low', analog=False, output='sos')
        baseline = sosfiltfilt(sos, ear_signal_original)

    ear_corrected = ear_signal_original - baseline
    processed_signal = -ear_corrected # Invert: blinks become positive
    if center_final:
        processed_signal = processed_signal - np.mean(processed_signal)
    return processed_signal

def strategy_baseline_savgol(ear_signal_original, window_length_frames=201, polyorder=3, center_final=True, **kwargs):
    """Baseline correction using Savitzky-Golay filter."""
    window_length_frames = int(window_length_frames)
    polyorder = int(polyorder)
    if window_length_frames % 2 == 0:
        window_length_frames +=1
    if polyorder >= window_length_frames:
        polyorder = max(1, window_length_frames - 2) # Ensure polyorder is valid and < window_length
        print(f"Warning: SavGol polyorder adjusted to {polyorder} for window_length {window_length_frames}")

    baseline = savgol_filter(ear_signal_original, window_length=window_length_frames, polyorder=polyorder)
    ear_corrected = ear_signal_original - baseline
    processed_signal = -ear_corrected # Invert: blinks become positive
    if center_final:
        processed_signal = processed_signal - np.mean(processed_signal)
    return processed_signal

def strategy_baseline_morphological_opening(ear_signal_original, structure_size_frames=100, center_final=True, **kwargs):
    """Baseline correction using morphological opening."""
    baseline = grey_opening(ear_signal_original, size=int(structure_size_frames))
    ear_corrected = ear_signal_original - baseline
    processed_signal = -ear_corrected # Invert: blinks become positive
    if center_final:
        processed_signal = processed_signal - np.mean(processed_signal)
    return processed_signal


# --- Refined Zero-Crossing Detection ---
def _to_int_safe(val, default_val=0):
    if pd.isna(val): return default_val
    try: return int(val)
    except (ValueError, TypeError): return default_val

def find_blink_zero_crossings_refined(processed_signal, peak_frame_idx,
                                      search_start_idx, search_end_idx,
                                      max_search_extension_frames=50):
    peak_frame_idx = _to_int_safe(peak_frame_idx)
    search_start_idx = _to_int_safe(search_start_idx, default_val=0)
    # search_end_idx is exclusive for slicing, so can be len(signal)
    search_end_idx = _to_int_safe(search_end_idx, default_val=len(processed_signal))


    n_signal = len(processed_signal)
    left_zero_idx, right_zero_idx = np.nan, np.nan

    # Ensure indices are valid and within bounds for access
    peak_frame_idx = max(0, min(peak_frame_idx, n_signal - 1))
    search_start_idx = max(0, min(search_start_idx, n_signal -1)) # for access
    search_end_idx = max(0, min(search_end_idx, n_signal)) # for slicing

    # --- Left Zero Crossing ---
    # Search window: [actual_search_start, peak_frame_idx)
    # Ensure actual_left_search_start does not exceed peak_frame_idx
    actual_left_search_start = min(search_start_idx, peak_frame_idx)
    left_search_end = peak_frame_idx # exclusive

    if left_search_end > actual_left_search_start: # If there is a segment to search
        left_segment = processed_signal[actual_left_search_start:left_search_end]
        neg_indices_in_left = np.flatnonzero(left_segment < 0)
        if neg_indices_in_left.size > 0:
            left_zero_idx = actual_left_search_start + neg_indices_in_left[-1]
        elif max_search_extension_frames > 0:
            extended_left_start = max(0, actual_left_search_start - max_search_extension_frames)
            if extended_left_start < actual_left_search_start: # If there's new area to search
                extended_left_segment = processed_signal[extended_left_start:actual_left_search_start]
                neg_indices_in_extended_left = np.flatnonzero(extended_left_segment < 0)
                if neg_indices_in_extended_left.size > 0:
                    left_zero_idx = extended_left_start + neg_indices_in_extended_left[-1]

    # --- Right Zero Crossing ---
    # Search window: (peak_frame_idx, actual_search_end] -> python slice [peak_frame_idx + 1, actual_search_end)
    right_search_start = peak_frame_idx + 1
    # Ensure actual_right_search_end is not less than right_search_start
    actual_right_search_end = max(search_end_idx, right_search_start)


    if actual_right_search_end > right_search_start and right_search_start < n_signal: # If there is a segment to search
        right_segment = processed_signal[right_search_start:actual_right_search_end]
        neg_indices_in_right = np.flatnonzero(right_segment < 0)
        if neg_indices_in_right.size > 0:
            right_zero_idx = right_search_start + neg_indices_in_right[0]
        elif max_search_extension_frames > 0:
            extended_right_end = min(n_signal, actual_right_search_end + max_search_extension_frames)
            if extended_right_end > actual_right_search_end: # If there's new area to search
                extended_right_segment = processed_signal[actual_right_search_end:extended_right_end]
                neg_indices_in_extended_right = np.flatnonzero(extended_right_segment < 0)
                if neg_indices_in_extended_right.size > 0:
                    right_zero_idx = actual_right_search_end + neg_indices_in_extended_right[0]

    return left_zero_idx, right_zero_idx


# --- Main Processing and Evaluation Function ---
def evaluate_blink_processing_strategy(
        original_ear_signal,
        ground_truth_df,
        strategy_fn,
        strategy_params=None,
        zc_extension_frames=50,
        sfreq=None
):
    if strategy_params is None: strategy_params = {}

    # Pass sfreq to strategy_params if not already there and function expects it
    if 'sfreq' not in strategy_params and sfreq is not None:
        strategy_params_with_sfreq = {**strategy_params, 'sfreq': sfreq}
    else:
        strategy_params_with_sfreq = strategy_params

    processed_signal = strategy_fn(original_ear_signal, **strategy_params_with_sfreq)

    output_df = ground_truth_df.copy()
    output_df['processed_peak_value'] = np.nan
    output_df['left_zc_idx'] = np.nan
    output_df['right_zc_idx'] = np.nan
    output_df['is_peak_positive'] = False
    output_df['zc_interval_valid'] = False


    positive_peak_count = 0
    zc_found_and_valid_duration_count = 0
    valid_gt_blinks_count = 0

    for idx, blink_gt in ground_truth_df.iterrows():
        blink_min_gt = _to_int_safe(blink_gt['blink_min'])
        start_blink_gt = _to_int_safe(blink_gt['startBlinks'])
        end_blink_gt = _to_int_safe(blink_gt['endBlinks'])

        # Basic validation of GT indices against processed signal length
        if not (0 <= blink_min_gt < len(processed_signal) and
                0 <= start_blink_gt < len(processed_signal) and
                0 < end_blink_gt <= len(processed_signal) and # end_blink_gt is exclusive for slicing
                start_blink_gt < end_blink_gt and blink_min_gt >= start_blink_gt and blink_min_gt < end_blink_gt):
            # print(f"Warning: Blink GT index {idx} out of bounds or invalid range, skipping. "
            #       f"Min: {blink_min_gt}, Start: {start_blink_gt}, End: {end_blink_gt}, Signal len: {len(processed_signal)}")
            continue
        valid_gt_blinks_count +=1

        peak_value = processed_signal[blink_min_gt]
        output_df.loc[idx, 'processed_peak_value'] = peak_value
        is_positive = peak_value > 0
        output_df.loc[idx, 'is_peak_positive'] = is_positive
        if is_positive: positive_peak_count += 1

        left_zc, right_zc = find_blink_zero_crossings_refined(
            processed_signal,
            peak_frame_idx=blink_min_gt,
            search_start_idx=start_blink_gt,
            search_end_idx=end_blink_gt,
            max_search_extension_frames=zc_extension_frames
        )
        output_df.loc[idx, 'left_zc_idx'] = left_zc
        output_df.loc[idx, 'right_zc_idx'] = right_zc

        if not pd.isna(left_zc) and not pd.isna(right_zc) and (right_zc - left_zc >= 3):
            output_df.loc[idx, 'zc_interval_valid'] = True
            zc_found_and_valid_duration_count += 1

    stats = {
        "strategy_name": strategy_fn.__name__,
        "total_gt_blinks_input": len(ground_truth_df),
        "valid_gt_blinks_processed": valid_gt_blinks_count,
        "blinks_with_positive_peak": positive_peak_count,
        "percentage_positive_peaks": (positive_peak_count / valid_gt_blinks_count * 100) if valid_gt_blinks_count > 0 else 0,
        "blinks_with_valid_zc_interval": zc_found_and_valid_duration_count,
        "percentage_valid_zc_interval": (zc_found_and_valid_duration_count / valid_gt_blinks_count * 100) if valid_gt_blinks_count > 0 else 0, # This line should have 'else 0'
    }
    if strategy_params: stats["strategy_params"] = strategy_params
    return output_df, stats

# --- Main Execution ---
if __name__ == "__main__":
    fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
    zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"
    pickle_path = "fitblinks_debug.pkl" # Ensure this path is correct

    # Load data - PRIORITIZE PICKLE if it contains the necessary fields
    original_ear_signal = None
    blink_df_gt = None
    sfreq = None # Will be determined

    try:
        with open(pickle_path, "rb") as f:
            debug_data = pickle.load(f)

        # IMPORTANT: Ensure your pickle contains these exact keys with the correct data
        # 'candidate_signal_original' should be the raw EAR signal (1.8M points)
        # 'gt_df' should be your ground truth DataFrame with 'startBlinks', 'endBlinks', 'blink_min'
        # 'params' should be a dict, hopefully containing 'sfreq'
        if 'candidate_signal_original' in debug_data and 'gt_df' in debug_data and 'params' in debug_data:
            original_ear_signal = debug_data["candidate_signal_original"]
            blink_df_gt = debug_data["gt_df"]
            sfreq = debug_data["params"].get('sfreq')
            print(f"Successfully loaded data from '{pickle_path}'.")
            if not isinstance(original_ear_signal, np.ndarray) or original_ear_signal.ndim != 1:
                raise ValueError("Pickled 'candidate_signal_original' is not a 1D numpy array.")
            if not isinstance(blink_df_gt, pd.DataFrame):
                raise ValueError("Pickled 'gt_df' is not a pandas DataFrame.")
            required_cols = ['startBlinks', 'endBlinks', 'blink_min']
            if not all(col in blink_df_gt.columns for col in required_cols):
                raise ValueError(f"Pickled 'gt_df' is missing one of required columns: {required_cols}")
            if sfreq is None:
                print(f"Warning: 'sfreq' not found in pickled 'params'. Attempting to load from FIF.")


        else:
            print(f"Pickle '{pickle_path}' found, but missing required keys. Attempting to load from FIF.")
            # Fall through to FIF loading if pickle is incomplete
    except FileNotFoundError:
        print(f"Pickle file '{pickle_path}' not found. Attempting to load from FIF.")
    except Exception as e:
        print(f"Error loading or validating data from pickle '{pickle_path}': {e}. Attempting to load from FIF.")

    if original_ear_signal is None or blink_df_gt is None or sfreq is None:
        print("Loading data from FIF and annotations...")
        raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)

        # --- THIS IS A CRITICAL STEP: Identify your actual EAR signal ---
        # If your EAR signal is one of the MNE raw channels (e.g., 'EOG'):
        try:
            original_ear_signal = raw.get_data(picks=['EOG'])[0] # Or the correct channel name/index
            print(f"Using 'EOG' channel from FIF as original_ear_signal.")
        except ValueError: # If EOG channel doesn't exist
            original_ear_signal = raw.get_data(picks=[0])[0] # Fallback to first channel, **ADJUST THIS**
            print(f"Warning: 'EOG' channel not found in FIF. Using first channel as original_ear_signal. PLEASE VERIFY.")

        sfreq = raw.info['sfreq']

        # --- THIS IS A CRITICAL STEP: Ensure blink_df_gt is correctly formed ---
        # extract_blink_durations needs to convert MNE annotations (time-based)
        # to sample-based 'startBlinks', 'endBlinks', 'blink_min' for original_ear_signal
        frame_offset = debug_data["params"].get('frame_offset_samples', int(0.02 * sfreq)) # Example: 20ms offset in samples
        blink_df_gt = extract_blink_durations(annotation_df, frame_offset, sfreq)
        print(f"Extracted ground truth blinks from annotations.")


    # Final checks
    if original_ear_signal is None:
        raise ValueError("Could not load or determine original_ear_signal.")
    if blink_df_gt is None or blink_df_gt.empty:
        raise ValueError("Ground truth blink DataFrame (blink_df_gt) is empty or could not be loaded.")
    if sfreq is None:
        raise ValueError("Sampling frequency (sfreq) could not be determined.")

    print(f"\nUsing sfreq: {sfreq:.2f} Hz")
    print(f"Original EAR signal length: {len(original_ear_signal)} points")
    print(f"Number of ground truth blinks: {len(blink_df_gt)}")
    if not blink_df_gt.empty:
        print("Ground truth DataFrame head:\n", blink_df_gt.head())


    all_results_dfs = {}
    all_stats = []

    # Define strategies to test (adjust parameters based on sfreq and data characteristics)
    # Window sizes/structure sizes are in frames/samples
    strategies_to_test = [
        {
            "name": "Original Transform", "fn": strategy_original_transform, "params": {}
        },
        {
            "name": "RollMedian (W:0.2s, Center)", "fn": strategy_baseline_rolling_median,
            "params": {"window_size_frames": int(0.2 * sfreq), "center_final": True}
        },
        {
            "name": "RollMedian (W:0.5s, Center)", "fn": strategy_baseline_rolling_median,
            "params": {"window_size_frames": int(0.5 * sfreq), "center_final": True}
        },
        {
            "name": "RollMedian (W:1.0s, No Center)", "fn": strategy_baseline_rolling_median,
            "params": {"window_size_frames": int(1.0 * sfreq), "center_final": False}
        },
        {
            "name": "Butterworth (Cut:0.5Hz, Ord:3, Center)", "fn": strategy_baseline_lowpass_butterworth,
            "params": {"cutoff_hz": 0.5, "order": 3, "center_final": True}
        },
        {
            "name": "Butterworth (Cut:1.0Hz, Ord:3, NoCenter)", "fn": strategy_baseline_lowpass_butterworth,
            "params": {"cutoff_hz": 1.0, "order": 3, "center_final": False}
        },
        {
            "name": "SavGol (W:0.5s, P:3, Center)", "fn": strategy_baseline_savgol,
            "params": {"window_length_frames": int(0.5 * sfreq), "polyorder": 3, "center_final": True}
        },
        {
            "name": "SavGol (W:1s, P:2, Center)", "fn": strategy_baseline_savgol, # Polyorder P:2 is common for smoothing
            "params": {"window_length_frames": int(1.0 * sfreq), "polyorder": 2, "center_final": True}
        },
        {
            "name": "MorphOpen (S:0.5s, Center)", "fn": strategy_baseline_morphological_opening,
            "params": {"structure_size_frames": int(0.5 * sfreq), "center_final": True}
        },
        {
            "name": "MorphOpen (S:1s, Center)", "fn": strategy_baseline_morphological_opening,
            "params": {"structure_size_frames": int(1.0 * sfreq), "center_final": True}
        }
    ]

    zc_ext_frames = int(0.05 * sfreq) # e.g., 50ms zero-crossing search extension

    for config in strategies_to_test:
        print(f"\n--- Testing Strategy: {config['name']} ---")
        try:
            processed_df, stats = evaluate_blink_processing_strategy(
                original_ear_signal.copy(),
                blink_df_gt.copy(),
                strategy_fn=config["fn"],
                strategy_params=config["params"],
                zc_extension_frames=zc_ext_frames,
                sfreq=sfreq
            )
            all_results_dfs[config['name']] = processed_df
            all_stats.append(stats)
            print(f"  Strategy: {stats['strategy_name']}")
            print(f"  Parameters: {stats.get('strategy_params', 'N/A')}")
            print(f"  Valid GT Blinks Processed: {stats['valid_gt_blinks_processed']}/{stats['total_gt_blinks_input']}")
            print(f"  Blinks w/ Positive Peak: {stats['blinks_with_positive_peak']} ({stats['percentage_positive_peaks']:.2f}%)")
            print(f"  Blinks w/ Valid ZC Interval: {stats['blinks_with_valid_zc_interval']} ({stats['percentage_valid_zc_interval']:.2f}%)")

        except Exception as e:
            print(f"ERROR processing strategy {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n--- Summary of Strategies (Ranked) ---")
    if all_stats:
        ranked_stats = sorted(all_stats,
                              key=lambda x: (x['percentage_positive_peaks'], x['percentage_valid_zc_interval']),
                              reverse=True)
        for rank, stats in enumerate(ranked_stats):
            print(f"{rank+1}. Strategy: {stats['strategy_name']}")
            print(f"   Params: {stats.get('strategy_params', 'N/A')}")
            print(f"   % Positive Peaks: {stats['percentage_positive_peaks']:.2f}% ({stats['blinks_with_positive_peak']}/{stats['valid_gt_blinks_processed']})")
            print(f"   % Valid ZC Intervals: {stats['percentage_valid_zc_interval']:.2f}% ({stats['blinks_with_valid_zc_interval']}/{stats['valid_gt_blinks_processed']})")

        best_strategy_name = ranked_stats[0]['strategy_name']
        print(f"\nRECOMMENDED STRATEGY based on this run: {best_strategy_name}")
        # Example: Accessing the DataFrame for the best strategy
        # best_df = all_results_dfs[best_strategy_name]
        # print("\nHead of DataFrame from best strategy:")
        # print(best_df.head())
    else:
        print("No strategies were successfully evaluated.")