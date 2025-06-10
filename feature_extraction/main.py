import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
import numpy as np

logging.basicConfig(level=logging.INFO)

def load_and_pick_channels(fif_path, channel_names):
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    return raw.copy().pick(channel_names)

def extract_time_window(raw, start_time, end_time):
    sfreq = raw.info['sfreq']
    s0 = int(start_time * sfreq)
    s1 = int(end_time   * sfreq)
    data, times = raw[:, s0:s1]
    return data, times

def scale_data(data, channel_names,
               method='none',
               custom_factors=None,
               target_amplitude=1.0):
    """
    data: array (n_ch, n_s)
    method: 'none' | 'custom' | 'zscore' | 'minmax'
    - 'none'    → leave as is
    - 'custom'  → multiply each channel i by custom_factors[i]
    - 'zscore'  → subtract mean & divide by std (per channel)
    - 'minmax'  → map each channel to [-target_amplitude, +target_amplitude]
    """
    n_ch = data.shape[0]
    scaled = data.copy()

    if method == 'custom':
        if custom_factors is None or len(custom_factors) != n_ch:
            raise ValueError("Provide one factor per channel")
        for i in range(n_ch):
            scaled[i] *= custom_factors[i]

    elif method == 'zscore':
        for i in range(n_ch):
            mean = np.mean(scaled[i])
            std  = np.std(scaled[i])
            if std < 1e-12:
                raise ValueError(f"Channel {channel_names[i]} is constant")
            scaled[i] = (scaled[i] - mean) / std

    elif method == 'minmax':
        for i in range(n_ch):
            mn = scaled[i].min()
            mx = scaled[i].max()
            if mx == mn:
                raise ValueError(f"Channel {channel_names[i]} is constant")
            # map [mn, mx] → [-target, +target]
            scaled[i] = ((scaled[i] - mn) / (mx - mn) * 2 - 1) * target_amplitude

    # method == 'none' → do nothing
    return scaled

def plot_signals(times, data, channel_names,
                 start_time, end_time,
                 separate_plots=False):
    n_ch = data.shape[0]
    if not separate_plots:
        plt.figure(figsize=(10, 4))
        for i, name in enumerate(channel_names):
            plt.plot(times, data[i], label=name)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (scaled)')
        plt.title(f"Signals: {', '.join(channel_names)}\n({start_time}s–{end_time}s)")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 3*n_ch), sharex=True)
        if n_ch == 1: axes = [axes]
        for ax, i, name in zip(axes, range(n_ch), channel_names):
            ax.plot(times, data[i])
            ax.set_ylabel('Amp (scaled)')
            ax.set_title(name)
            ax.grid(True)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

# ---------------- MAIN EXECUTION ----------------

if __name__ == '__main__':
    fif_path = r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\temp_for_combine\combined_s1_aligned_raw.fif'

    # pick whichever channels you like:
    channels = [
        'EAR-avg_ear',
        'EEG-eog_vert_right',
        'EEG-eog_vert_left',
    ]

    raw = load_and_pick_channels(fif_path, channels)
    t0, t1 = 10, 14
    data, times = extract_time_window(raw, t0, t1)

    # --- scale your data here! options:
    # 1) custom: e.g. boost EEG-eog channels by 10×
    custom_factors = [1.0, 500.0, 500.0]
    data_scaled = scale_data(data, channels,
                             method='custom',
                             custom_factors=custom_factors)

    # 2) OR: z-score normalization
    # data_scaled = scale_data(data, channels, method='zscore')

    # 3) OR: min–max to ±1
    # data_scaled = scale_data(data, channels, method='minmax', target_amplitude=1.0)

    # check zero-mean AFTER scaling (if you want)
    if np.allclose(np.mean(data_scaled, axis=1), 0, atol=1e-6):
        print("✅ All channels zero-mean after scaling.")
    else:
        print("⚠️ Some channels not zero-mean after scaling.")

    # plot
    plot_signals(times, data_scaled, channels, t0, t1, separate_plots=False)
