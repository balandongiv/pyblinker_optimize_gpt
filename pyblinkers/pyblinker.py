from pyblinkers.utils._logging import logger

import pandas as pd
import logging
from tqdm import tqdm

from pyblinkers import default_setting
from pyblinkers.extractBlinkProperties import BlinkProperties, get_good_blink_mask
from pyblinkers.extractBlinkProperties import get_blink_statistic
from pyblinkers.fit_blink import FitBlinks
from pyblinkers.getBlinkPositions import get_blink_position
from pyblinkers.getRepresentativeChannel import channel_selection
from pyblinkers.misc import create_annotation
from pyblinkers.viz_pd import viz_complete_blink_prop


class BlinkDetector:
    def __init__(self,
                 raw_data,
                 visualize=False,
                 annot_label=None,
                 filter_bad=False,
                 filter_low=0.5,
                 filter_high=20.5,
                 resample_rate=30,
                 n_jobs=1,
                 use_multiprocessing=False,
                 pick_types_options=None):
        """
        Initialize the BlinkDetector.

        Parameters:
            raw_data: The raw EEG data.
            visualize (bool): Whether to generate visualization data.
            annot_label (str): Annotation label for blink events.
            filter_bad (bool): Whether to filter out bad blinks.
            filter_low (float): Low frequency filter cutoff.
            filter_high (float): High frequency filter cutoff.
            resample_rate (int): New sampling rate for the data.
            n_jobs (int): Number of jobs to use for processing.
            use_multiprocessing (bool): Whether to use multiprocessing.
            pick_types_options (dict): Dictionary of channel type options to pass
                                       to raw_data.pick_types, e.g. {'eeg': True, 'eog': True}.
        """
        self.filter_bad = filter_bad
        self.raw_data = raw_data
        self.viz_data = visualize
        self.annot_label = annot_label
        self.sfreq = self.raw_data.info['sfreq']
        self.params = default_setting.params
        self.channel_list = self.raw_data.ch_names
        self.all_data_info = []  # To store processed blink data per channel
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample_rate = resample_rate
        self.n_jobs = n_jobs
        self.use_multiprocessing = use_multiprocessing
        self.all_data = []
        # Default to picking only EEG if none is provided
        self.pick_types_options = pick_types_options if pick_types_options is not None else {'eeg': True}

    def prepare_raw_signal(self):
        """
        Preprocess raw signal:
          - pick channel types
          - filter
          - resample
        """
        logger.info("Preparing raw signal: picking channels, filtering, and resampling.")
        self.raw_data.pick_types(**self.pick_types_options)
        self.raw_data.filter(self.filter_low, self.filter_high, fir_design='firwin', n_jobs=self.n_jobs)
        self.raw_data.resample(self.resample_rate, n_jobs=self.n_jobs)
        logger.info(f"Signal prepared with resample rate: {self.resample_rate} Hz")
        return self.raw_data

    def process_channel_data(self, channel):
        """Process data for a single channel."""
        logger.info(f"Processing channel: {channel}")
        
        # STEP 1: Get blink positions
        df = get_blink_position(self.params,
                                blink_component=self.raw_data.get_data(picks=channel)[0],
                                ch=channel)
        
        if df.empty:
            logger.warning(f"No blinks detected in channel: {channel}")

        # STEP 2: Fit blinks
        fitblinks = FitBlinks(
            candidate_signal=self.raw_data.get_data(picks=channel)[0],
            df=df,
            params=self.params
        )
        fitblinks.dprocess()
        df = fitblinks.frame_blinks

        # STEP 3: Extract blink statistics
        blink_stats = get_blink_statistic(
            df, self.params['z_thresholds'],
            signal=self.raw_data.get_data(picks=channel)[0]
        )
        blink_stats['ch'] = channel

        # STEP 4: Get good blink mask
        good_blink_mask, df = get_good_blink_mask(
            df,
            blink_stats['bestMedian'],
            blink_stats['bestRobustStd'],
            self.params['z_thresholds']
        )

        # STEP 5: Compute blink properties
        df = BlinkProperties(
            self.raw_data.get_data(picks=channel)[0],
            df,
            self.params['sfreq'],
            self.params
        ).df

        # STEP 6: Apply pAVR restriction
        condition_1 = df['posAmpVelRatioZero'] < self.params['pAVRThreshold']
        condition_2 = df['maxValue'] < (blink_stats['bestMedian'] - blink_stats['bestRobustStd'])
        df = df[~(condition_1 & condition_2)]

        # Store results
        self.all_data_info.append(dict(df=df, ch=channel))
        self.all_data.append(blink_stats)

    @staticmethod
    def filter_point(ch, all_data_info):
        """Helper to extract data information for a specific channel."""
        return list(filter(lambda data: data['ch'] == ch, all_data_info))[0]

    def filter_bad_blink(self, df):
        """Optionally filter out bad blinks."""
        if self.filter_bad:
            df = df[df['blink_quality'] == 'Good']
        return df

    def generate_viz(self, data, df):
        """Generate visualization for each blink if visualization is enabled."""
        fig_data = [
            viz_complete_blink_prop(data, row, self.sfreq)
            for _, row in df.iterrows()
        ]
        return fig_data

    def process_all_channels(self):
        """Process all channels available in the raw data."""
        logger.info(f"Processing {len(self.channel_list)} channels.")
        for channel in tqdm(self.channel_list, desc="Processing Channels", unit="channel"):
            self.process_channel_data(channel)
        logger.info("Finished processing all channels.")

    def select_representative_channel(self):
        """Select the best representative channel based on blink statistics."""
        ch_blink_stat = pd.DataFrame(self.all_data)
        ch_selected = channel_selection(ch_blink_stat, self.params)
        ch_selected.reset_index(drop=True, inplace=True)
        return ch_selected

    def get_representative_blink_data(self, ch_selected):
        """Retrieve blink data from the selected representative channel."""
        ch = ch_selected.loc[0, 'ch']
        data = self.raw_data.get_data(picks=ch)[0]
        rep_blink_channel = self.filter_point(ch, self.all_data_info)
        df = rep_blink_channel['df']
        df = self.filter_bad_blink(df)
        return ch, data, df

    def create_annotations(self, df):
        """Create annotations based on the blink data."""
        annot_description = self.annot_label if self.annot_label else 'eye_blink'
        return create_annotation(df, self.sfreq, annot_description)

    def get_blink(self):
        """
        Run the complete blink detection pipeline:
            - Prepare raw signal
            - Process all channels
            - Select representative channel
            - Create annotations
            - Generate visualizations (optional)
        """
        logger.info("Starting blink detection pipeline.")
        
        self.prepare_raw_signal()
        self.process_all_channels()
        
        ch_selected = self.select_representative_channel()
        logger.info(f"Selected representative channel: {ch_selected.loc[0, 'ch']}")
        
        ch, data, df = self.get_representative_blink_data(ch_selected)
        annot = self.create_annotations(df)
        
        fig_data = self.generate_viz(data, df) if self.viz_data else []
        n_good_blinks = ch_selected.loc[0, 'numberGoodBlinks']
        
        logger.info(f"Blink detection completed. {n_good_blinks} good blinks detected.")
        
        return annot, ch, n_good_blinks, df, fig_data, ch_selected


def run_blink_detection_pipeline(raw_data, config=None):
    """
    Execute the blink detection pipeline end-to-end.

    Parameters:
        raw_data: The raw EEG data.
        config (dict, optional): Configuration parameters for BlinkDetector.
            Example defaults:
                {
                    'visualize': False,
                    'annot_label': None,
                    'filter_low': 0.5,
                    'filter_high': 20.5,
                    'resample_rate': 100,
                    'n_jobs': 2,
                    'use_multiprocessing': True,
                    'pick_types_options': {'eeg': True}
                }

    Returns:
        dict: A dictionary containing the pipeline results:
              {
                'annotations': ...,
                'channel': ...,
                'number_good_blinks': ...,
                'blink_dataframe': ...,
                'figures': ...,
                'selected_channel_data': ...
              }
    """
    logger.info("Initializing blink detection pipeline.")
    
    # Default configuration parameters
    default_config = {
        'visualize': False,
        'annot_label': None,
        'filter_low': 0.5,
        'filter_high': 20.5,
        'resample_rate': 100,
        'n_jobs': 2,
        'use_multiprocessing': True,
        'pick_types_options': {'eeg': True}
    }

    if config is not None:
        default_config.update(config)

    # Instantiate and run the detector
    detector = BlinkDetector(raw_data, **default_config)
    annot, ch, number_good_blinks, df, fig_data, ch_selected = detector.get_blink()

    logger.info(f"Pipeline completed. Selected channel: {ch}, Good blinks: {number_good_blinks}")

    return {
        'annotations': annot,
        'channel': ch,
        'number_good_blinks': number_good_blinks,
        'blink_dataframe': df,
        'figures': fig_data,
        'selected_channel_data': ch_selected,
    }


def main():
    """
    Example main function demonstrating how the pipeline might be called.
    Replace placeholder code (comments) with actual data loading or
    configuration logic as needed.
    """
    logger.info("Starting blink detection pipeline...")

    # TODO: Load your raw EEG data here. For example:
    # raw = mne.io.read_raw_fif('path_to_your_file.fif', preload=True)

    # Example config
    config = {
        'visualize': False,
        'annot_label': 'my_blink_label',
        'filter_low': 0.5,
        'filter_high': 20.5,
        'resample_rate': 100,
        'n_jobs': 2,
        'use_multiprocessing': True,
        'pick_types_options': {'eeg': True, 'eog': True}
    }

    # results = run_blink_detection_pipeline(raw, config=config)
    # print("Blink Detection Results:")
    # print(results)

    print("No real data loaded. Replace placeholder code in main() with actual logic.")


if __name__ == "__main__":
    main()
