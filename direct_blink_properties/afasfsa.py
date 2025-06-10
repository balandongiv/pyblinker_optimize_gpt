from direct_blink_properties.util import load_fif_and_annotations,extract_blink_durations
from direct_blink_properties.viz import generate_blink_reports,plot_with_annotation_lines
from pathlib import Path
# We will use subject S01 from the dataset
# fif_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif"
# zip_path = r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip"


# fif_path = r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1\S01_20170519_043933.fif'
# zip_path = r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final\S1\from_cvat\S01_20170519_043933.zip'
#
fif_path = r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\pyblink_ear_combine_ground_annot\S1\S01_20170519_043933_2.fif'
zip_path = r'C:\Users\balan\OneDrive - ums.edu.my\CVAT_visual_annotation\cvat_zip_final\S1\from_cvat\S01_20170519_043933_2.zip'
base_filename = f"{Path(fif_path).stem}_blink_report"
output_dir='blink_reports'
# Load data
raw, annotation_df = load_fif_and_annotations(fif_path, zip_path,use_cache=False)
# Extract blink intervals
frame_offset=5
video_fps=30
sfreq = raw.info['sfreq']
blink_df = extract_blink_durations(annotation_df,frame_offset,sfreq,video_fps)


# get the sampling rate

# Get overview about the time series data
raw.plot(
    picks=['Average EAR'
        # ,'E8'
           ],
    block=True,
    show_scrollbars=False,
    title='avg_ear Blink Signal'
)


# Get a plot by plotting the blink signal with the annotation lines
# ⚠️ WARNING: Only plotting the first 10 blinks for visual inspection
print("⚠️ WARNING: Only plotting the first 10 blink events...")
for _, row in  blink_df.head(10).iterrows():
    plot_with_annotation_lines(
        raw=raw,
        start_frame=row['startBlinks'],
        end_frame=row['endBlinks'],
        mid_frame=row['blinkmin'],
        picks='Average EAR',# some time we refer this as avg_ear
        sfreq=sfreq ,
    )

# Generate a report for the blink signal

generate_blink_reports(
    raw=raw,
    blink_df=blink_df,
    picks='Average EAR', # some time we refer this as avg_ear
    sfreq=sfreq ,
    output_dir=output_dir,
    base_filename=base_filename,
    max_events_per_report=40
)