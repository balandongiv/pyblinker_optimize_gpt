# Blink Detection and Analysis Pipeline Using pyblinkers

## 📌 Overview

This pipeline leverages the `pyblinkers` Python package to analyze blink events that have been manually annotated using the CVAT tool. Unlike conventional pipelines, which include automated filtering to remove questionable blink events, this approach directly uses annotations verified by human annotators. By doing so, we bypass traditional compliance-based filtering, resulting in accurate and reliable extraction of blink properties.

The primary goal of this pipeline is to precisely determine blink characteristics from video segments by utilizing accurate and detailed frame annotations, facilitating comprehensive blink analysis.

---

## 🔧 Step 1: Register Your Data Directories

In this step, you will specify and register distinct root directories for your annotation CSV files and EEG FIF files. These paths serve as the foundational references used throughout the pipeline to locate and process your datasets.

This structure is particularly beneficial for managing datasets stored across different drives or external storage devices, ensuring efficient and systematic access to data.

### 💡 Example

```python
from setting.project_path import set_data_paths

csv_root = r"G:\My Drive\cvat_annotations"
fif_root = r"D:\EEG_FIF_storage"

set_data_paths(csv_root=csv_root, fif_root=fif_root)
```

✅ Benefits of this setup:

* Paths are securely stored in a configuration JSON for seamless reuse.
* Facilitates independent loading and processing of annotations and EEG signals.
* Simplifies project management, especially with large-scale or distributed data.

---

## 🗂️ Detailed File Structure

### 📍 Annotation CSV Files

The annotations exported from CVAT are organized clearly by individual subjects:

```
csv_root/
├── S1/
│   └── annotations/
│       ├── S01_20170519_043933.zip
│       │   └── annotations/
│       │       └── default-annotations-human-imagelabels.csv
│       ├── S01_20170519_043933_2.zip
│       └── S01_20170519_043933_3.zip
├── S2/
└── S3/
```

### 📄 CSV Content Structure

Each annotation file includes entries like the following:

| ImageID       | Source | LabelName           | Confidence |
| ------------- | ------ | ------------------- | ---------- |
| frame\_000358 |        | HB\_CL\_left\_start | 1          |
| frame\_000361 |        | HB\_CL\_min         | 1          |
| frame\_000366 |        | HB\_CL\_right\_end  | 1          |
| frame\_000370 |        | M\_left\_start      | 1          |
| frame\_000372 |        | M\_min              | 1          |
| frame\_000376 |        | M\_right\_end       | 1          |

🔍 Important Notes:

* `ImageID` values are converted to integers for processing (e.g., "frame\_000358" → 358).
* Only complete sequences (start → min → end) are utilized for accurate blink property analysis.

### 🧠 EEG FIF Files

The EEG data files are stored independently, also organized by subject:

```
fif_root/
├── S1/
│   ├── S01_20170519_043933.fif
│   ├── S01_20170519_043933_2.fif
│   └── S01_20170519_043933_3.fif
├── S2/
└── S3/
```

Each FIF file contains multiple signal types including:

* **Ear**: Eye Aspect Ratio (EAR) signal, which tracks eyelid position dynamics.
* **EOG**: Electrooculography signals for eye movement tracking.
* **EEG**: Electroencephalography signals for brain activity monitoring.

These signals are grouped under `candidate_signal` and processed using the MNE-Python library.

---

## 🔁 Pipeline Workflow

### 🔹 Step 2: Extract Blink Durations from Annotation CSV

```python
import pandas as pd

def extract_blink_durations(annotation_df):
    blink_data = []
    for i in range(0, len(annotation_df) - 2, 3):
        start_label = annotation_df.iloc[i]['LabelName']
        mid_label = annotation_df.iloc[i+1]['LabelName']
        end_label = annotation_df.iloc[i+2]['LabelName']

        if start_label.endswith('_start') and end_label.endswith('_end'):
            blink_type = start_label.rsplit('_', 1)[0]
            blink_start = int(annotation_df.iloc[i]['ImageID'].replace('frame_', ''))
            blink_end = int(annotation_df.iloc[i+2]['ImageID'].replace('frame_', ''))
            blink_data.append({
                'blink_start': blink_start,
                'blink_end': blink_end,
                'blink_type': blink_type
            })

    return pd.DataFrame(blink_data)
```

**Example Output:**

| blink\_start | blink\_end | blink\_type |
| ------------ | ---------- | ----------- |
| 358          | 366        | HB\_CL      |
| 370          | 376        | M           |

### 🔹 Steps 3–5: Process Blink Properties

```python
from pyblinkers.extractBlinkProperties import BlinkProperties, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks

def process_blinks(candidate_signal, df, params):
    fitblinks = FitBlinks(candidate_signal, df, params)
    fitblinks.process_blink_candidate()
    df = fitblinks.frame_blinks

    blink_stats = get_blink_statistic(df, params['z_thresholds'], candidate_signal)

    # Optional filtering step (commented out by default)
    # good_blink_mask, df = get_good_blink_mask(df, blink_stats['bestMedian'], blink_stats['bestRobustStd'], params['z_thresholds'])

    df = BlinkProperties(candidate_signal, df, params['sfreq'], params).df
    return df, blink_stats
```

---

### 🔄 Step 6: Automated Looping Through Subjects and Files

To systematically process each subject along with their associated FIF and CSV files, utilize the following looping approach:

```python
import os
from your_utils import load_csv_annotations, load_fif_file

subjects = ["S1", "S2", "S3"]

for subject in subjects:
    csv_subject_dir = os.path.join(csv_root, subject)
    fif_subject_dir = os.path.join(fif_root, subject)

    for fif_file in os.listdir(fif_subject_dir):
        if fif_file.endswith(".fif"):
            fif_path = os.path.join(fif_subject_dir, fif_file)

            base_name = os.path.splitext(fif_file)[0]
            csv_path = os.path.join(csv_subject_dir, "annotations", f"{base_name}.zip", "annotations", "default-annotations-human-imagelabels.csv")

            annotation_df = load_csv_annotations(csv_path)
            blink_df = extract_blink_durations(annotation_df)

            candidate_signal = load_fif_file(fif_path)
            df, blink_stats = process_blinks(candidate_signal, blink_df, params)

            # Implement saving, further analysis, or visualization here
```

---

### 📊 Step 7: Visual Verification of Blink Events (Optional)

Although blink start and end frames are determined from human annotations, it's valuable to visually verify each detected blink event. After `fitblinks.frame_blinks`, several new columns are added, such as:

* Left zero crossing
* Right zero crossing
* Maximum peak

These can be used to generate blink-wise plots for validation.

You can either:

* Save each blink plot as a separate `.png` or `.jpg` using the blink start frame as filename (e.g., `358.png`).
* Use **MNE Report** to group visualizations into HTML files.

```python
def save_blink_plots(df, candidate_signal, out_dir):
    for _, row in df.iterrows():
        blink_id = row['blink_start']
        # Plot logic here...
        filename = os.path.join(out_dir, f"{blink_id}.png")
        plt.savefig(filename)
        plt.close()
```

For large datasets, MNE Reports should be split to avoid HTML overload. You can configure the max number of images per report:

```python
from mne import Report

def generate_mne_reports(image_paths, max_images_per_report, report_dir):
    for i in range(0, len(image_paths), max_images_per_report):
        report = Report()
        batch = image_paths[i:i+max_images_per_report]
        for img_path in batch:
            report.add_image(img_path, title=os.path.basename(img_path))
        report.save(os.path.join(report_dir, f"report_{i//max_images_per_report + 1}.html"), overwrite=True)
```

---

## ✅ Final Output

After successfully running the entire pipeline:

* A comprehensive DataFrame containing frame indices, blink types, and computed blink characteristics is generated.
* Optional blink plots can be saved individually or grouped via MNE reports.
* The dataset is optimized for subsequent statistical analysis, quality checks, modeling, or visualization purposes.
