from pyblinkers.utils._logging import logger

from pyblinkers import default_setting
from pyblinkers.misc import create_annotation
from pyblinkers.viz_pd import viz_complete_blink_prop
from .pipeline_steps import (
    process_channel_data as core_process_channel_data,
    process_all_channels as core_process_all_channels,
    select_representative_channel as core_select_representative_channel,
    get_representative_blink_data as core_get_representative_blink_data,
    get_blink as core_get_blink,
)


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
        self.params = default_setting.DEFAULT_PARAMS.copy()
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
        self.raw_data.filter(self.filter_low, self.filter_high,
                             fir_design='firwin',
                             n_jobs=self.n_jobs)
        self.raw_data.resample(self.resample_rate, n_jobs=self.n_jobs)
        logger.info(f"Signal prepared with resample rate: {self.resample_rate} Hz")
        return self.raw_data

    def process_channel_data(self, channel, verbose=True):
        """Process data for a single channel."""
        core_process_channel_data(self, channel, verbose)

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
        core_process_all_channels(self)

    def select_representative_channel(self):
        """Select the best representative channel based on blink statistics."""
        return core_select_representative_channel(self)

    def get_representative_blink_data(self, ch_selected):
        """Retrieve blink data from the selected representative channel."""
        return core_get_representative_blink_data(self, ch_selected)

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

        return core_get_blink(self)


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
