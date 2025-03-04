Certainly, let's break down each of the t-statistics provided for blinking behavior analysis. These statistics are designed to capture various aspects of eye blinks, which can be indicative of different physiological and neurological conditions.

Here's a detailed explanation of each statistic, drawing from the provided table and code:

**1. EAR_Before_Blink_left_avg & EAR_Before_Blink_right_avg**

*   **Description:** These statistics represent the average Eye Aspect Ratio (EAR) of the left and right eyes, respectively, calculated over a 3-second period *immediately before* the onset of the first detected blink in the recording.
*   **Meaning:**  EAR is a measure of eye openness, typically ranging from 0 (completely closed) to 1 (fully open).  A higher EAR generally indicates wider open eyes.  These "before blink" averages provide a baseline EAR value right before a blink occurs. This can be useful to understand the typical eye openness state just prior to blinking.  Changes in this baseline EAR might be relevant in conditions affecting muscle tone or alertness.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio).  EAR is a ratio, hence dimensionless, and its value is constrained between 0 and 1.
*   **Code Context:**
    ```python
    end_l = get_blink_start(matched_blinks["left"], 0) # start frame of the first left blink
    end_r = get_blink_start(matched_blinks["right"], 0) # start frame of the first right blink
    start_l = max(end_l - 3 * fps, 0) # 3 seconds before the blink start, capped at 0
    start_r = max(end_r - 3 * fps, 0) # 3 seconds before the blink start, capped at 0
    statistics["EAR_Before_Blink_left_avg"]  = np.nanmean(ear_l[start_l:end_l]) # average EAR in the 3s window
    statistics["EAR_Before_Blink_right_avg"] = np.nanmean(ear_r[start_r:end_r]) # average EAR in the 3s window
    ```
    The code calculates the start frame of the first blink for each eye (`get_blink_start`). Then, it determines a 3-second window preceding this start frame. Finally, it computes the average EAR within this window using `np.nanmean`, handling potential NaN values in the `ear_l` and `ear_r` arrays.

**2. EAR_left_min, EAR_right_min, EAR_left_max, EAR_right_max**

*   **Description:** These statistics represent the minimum and maximum EAR values observed for the left and right eyes, respectively, throughout the entire recording time series.
*   **Meaning:**
    *   `EAR_min`:  The lowest EAR value indicates the point of maximum eye closure during the recording. A very low `EAR_min` suggests complete or near-complete eye closure during blinks.
    *   `EAR_max`: The highest EAR value represents the point of maximum eye openness.  `EAR_max` is expected to be high when the eyes are wide open between blinks.
        These min/max values provide the range of eye openness achieved during the recording.  Abnormalities in these ranges (e.g., consistently low `EAR_max` or high `EAR_min`) could be indicative of issues.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio). Same as EAR.
*   **Code Context:**
    ```python
    statistics["EAR_left_min"]  = np.nanmin(ear_l) # Minimum EAR value in the entire left eye EAR time series
    statistics["EAR_right_min"] = np.nanmin(ear_r) # Minimum EAR value in the entire right eye EAR time series
    statistics["EAR_left_max"]  = np.nanmax(ear_l) # Maximum EAR value in the entire left eye EAR time series
    statistics["EAR_right_max"] = np.nanmax(ear_r) # Maximum EAR value in the entire right eye EAR time series
    ```
    These are straightforward calculations using `np.nanmin` and `np.nanmax` to find the minimum and maximum EAR values across the entire `ear_l` and `ear_r` arrays, again handling potential NaN values.

**3. Partial_Blink_threshold_left & Partial_Blink_threshold_right**

*   **Description:** These are *threshold values* used to classify blinks as either "partial" or "complete" for the left and right eyes.
*   **Meaning:**  A blink is considered "complete" if the EAR value drops *below* this threshold during the blink. If the EAR value only drops *to* or *above* this threshold, the blink is classified as "partial."  These thresholds are crucial for distinguishing between different types of blinks. The specific value of the threshold is likely determined based on calibration or prior research to optimally differentiate partial from complete blinks.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio).  The threshold is applied to the EAR value, so it has the same unit and range as EAR.
*   **Code Context:**
    ```python
    statistics["Partial_Blink_threshold_left"]  = partial_threshold_l # Directly using the input threshold value
    statistics["Partial_Blink_threshold_right"] = partial_threshold_r # Directly using the input threshold value
    ```
    These statistics simply store the input `partial_threshold_l` and `partial_threshold_r` values, which are used elsewhere in the `JeFaPaTo` software to classify blinks.

**4. Prominence_min, Prominence_max, Prominence_avg**

*   **Description:** These statistics relate to the "prominence" of blinks. Prominence is a measure of how much a blink "stands out" from the surrounding EAR signal. It's calculated for each blink event (for both left and right eyes) and then the minimum, maximum, and average prominence values across all blinks are computed.
*   **Meaning:**  Prominence in signal processing often refers to the height of a peak relative to its surroundings. In the context of blinks and EAR, a higher prominence likely indicates a more pronounced blink, where the eye closes more significantly and for a longer duration relative to the baseline EAR.
    *   `Prominence_min`: The least prominent blink observed.
    *   `Prominence_max`: The most prominent blink observed.
    *   `Prominence_avg`: The average prominence of all blinks.
        These statistics can reflect the consistency and intensity of blinks. Reduced prominence might be associated with fatigue or certain neurological conditions.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio). Prominence is likely derived from the EAR signal, so it's expected to have a similar range.
*   **Code Context:**
    ```python
    prom_l = matched_blinks["left"]["prominance"] # Prominence values for left eye blinks
    prom_r = matched_blinks["right"]["prominance"] # Prominence values for right eye blinks
    prom = np.concatenate([prom_l, prom_r])      # Combine prominences from both eyes

    statistics["Prominence_min"] = np.nanmin(prom) # Minimum prominence across all blinks
    statistics["Prominence_max"] = np.nanmax(prom) # Maximum prominence across all blinks
    statistics["Prominence_avg"] = np.nanmean(prom) # Average prominence across all blinks
    ```
    The code retrieves pre-calculated prominence values from the `matched_blinks` DataFrame for both eyes, concatenates them, and then calculates the minimum, maximum, and average using `np.nanmin`, `np.nanmax`, and `np.nanmean`.

**5. Width_min, Width_max, Width_avg**

*   **Description:** These statistics relate to the "width" of blinks.  "Width" in this context likely refers to the duration of the blink event in frames or time units. It's calculated for each blink (left and right eye), and then the minimum, maximum, and average widths are computed across all blinks.
*   **Meaning:** Blink width (duration) is a key characteristic of blinking behavior.
    *   `Width_min`: The shortest blink duration observed.
    *   `Width_max`: The longest blink duration observed.
    *   `Width_avg`: The average blink duration.
        Blink duration can be affected by factors like fatigue, dryness, and neurological conditions.  Changes in blink width statistics can be clinically relevant.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio).  While the description mentions "width value," and the code uses "peak_internal_width," the unit/range suggests it's normalized or represented as a ratio. It's possible this "width" is a normalized measure related to the blink duration relative to some reference duration, or it could be a mislabeled unit in the table and actually represent time in frames or milliseconds internally before being potentially normalized for output.  *Based on the code calculating `blink_lengths_ms` using `peak_internal_width * 1000 / fps`, it's more likely that "peak_internal_width" is in frames.*
*   **Code Context:**
    ```python
    width_l = matched_blinks["left"]["peak_internal_width"] # Width values for left eye blinks (likely in frames)
    width_r = matched_blinks["right"]["peak_internal_width"] # Width values for right eye blinks (likely in frames)
    width = np.concatenate([width_l, width_r])          # Combine widths from both eyes

    statistics["Width_min"] = np.nanmin(width)         # Minimum width across all blinks
    statistics["Width_max"] = np.nanmax(width)         # Maximum width across all blinks
    statistics["Width_avg"] = np.nanmean(width)         # Average width across all blinks
    ```
    Similar to prominence, the code retrieves "peak_internal_width" values from `matched_blinks`, combines them, and calculates min, max, and average.

**6. Height_min, Height_max, Height_avg**

*   **Description:** These statistics relate to the "height" of blinks. "Height" likely refers to the depth of the blink, i.e., how much the EAR value decreases during the blink. It's calculated for each blink (left and right eye), and then the minimum, maximum, and average heights are computed across all blinks.
*   **Meaning:** Blink height (depth) reflects the extent of eye closure during a blink.
    *   `Height_min`: The shallowest blink (least eye closure).
    *   `Height_max`: The deepest blink (most eye closure).
    *   `Height_avg`: The average blink depth.
        Blink height, along with prominence, can indicate the intensity of blinks. Reduced blink height might be associated with incomplete blinks or certain conditions.
*   **Unit/Range:** $[0, 1]$ (dimensionless ratio). Similar to EAR and prominence, blink height is likely derived from the EAR signal and has a range of $[0, 1]$.
*   **Code Context:**
    ```python
    height_l = matched_blinks["left"]["peak_height"] # Height values for left eye blinks
    height_r = matched_blinks["right"]["peak_height"] # Height values for right eye blinks
    height = np.concatenate([height_l, height_r])        # Combine heights from both eyes

    statistics["Height_min"] = np.nanmin(height)        # Minimum height across all blinks
    statistics["Height_max"] = np.nanmax(height)        # Maximum height across all blinks
    statistics["Height_avg"] = np.nanmean(height)        # Average height across all blinks
    ```
    The code follows the same pattern as prominence and width, using "peak_height" from `matched_blinks` and calculating min, max, and average.

**7. Partial_Blink_Total_left & Partial_Blink_Total_right**

*   **Description:** These statistics count the total number of "partial" blinks detected for the left and right eyes throughout the entire recording.
*   **Meaning:**  As defined earlier, a "partial" blink is one where the EAR value does not drop below the `Partial_Blink_threshold`.  Counting partial blinks is important because an increased frequency of partial blinks can be a sign of eye strain, dryness, or certain neurological conditions.
*   **Unit/Range:** $\mathbb{N}$ (Natural numbers - non-negative integers).  These are counts, so they are whole numbers (0, 1, 2, ...).
*   **Code Context:**
    ```python
    partial_l = matched_blinks["left"][matched_blinks["left"]["blink_type"] == "partial"] # Filter for partial blinks in left eye
    partial_r = matched_blinks["right"][matched_blinks["right"]["blink_type"] == "partial"] # Filter for partial blinks in right eye

    statistics["Partial_Blink_Total_left"]  = len(partial_l) # Count of partial left blinks
    statistics["Partial_Blink_Total_right"] = len(partial_r) # Count of partial right blinks
    ```
    The code filters the `matched_blinks` DataFrame to select only rows where `blink_type` is "partial" for each eye and then uses `len()` to count the number of rows (i.e., partial blinks).

**8. Partial_Frequency_left_bpm & Partial_Frequency_right_bpm**

*   **Description:** These statistics calculate the frequency of "partial" blinks per minute (bpm - blinks per minute) for the left and right eyes, averaged over the entire recording duration.
*   **Meaning:** Blink frequency is a fundamental measure of blinking behavior.  Partial blink frequency specifically focuses on how often incomplete blinks occur per minute.  Elevated partial blink frequency can be clinically significant.
*   **Unit/Range:** $1/\text{min}$ (blinks per minute).
*   **Code Context:**
    ```python
    length_l_min = len(ear_l) / fps / 60 # Total recording length in minutes for left eye (based on EAR array length)
    length_r_min = len(ear_r) / fps / 60 # Total recording length in minutes for right eye (based on EAR array length)

    statistics["Partial_Frequency_left_bpm"]  = statistics["Partial_Blink_Total_left"] / length_l_min # Partial blinks per minute
    statistics["Partial_Frequency_right_bpm"] = statistics["Partial_Blink_Total_right"] / length_r_min # Partial blinks per minute
    ```
    The code first calculates the total recording duration in minutes (`length_l_min`, `length_r_min`) based on the length of the EAR arrays and the frames per second (`fps`). Then, it divides the total count of partial blinks (calculated in the previous step) by the recording duration in minutes to get the frequency in blinks per minute.

**9. Blink_Length_left_ms_avg, Blink_Length_left_ms_std, Blink_Length_right_ms_avg, Blink_Length_right_ms_std**

*   **Description:** These statistics describe the duration (length) of blinks in milliseconds (ms). They calculate the average (mean) and standard deviation (std) of blink lengths for both left and right eyes.
*   **Meaning:** Blink length (duration) is another crucial characteristic.
    *   `Blink_Length_avg`: The average duration of blinks.
    *   `Blink_Length_std`: The standard deviation of blink durations, indicating the variability in blink lengths. A higher standard deviation means blink durations are more inconsistent.
        Changes in average blink length or increased variability can be clinically relevant.
*   **Unit/Range:** Time in $ms$ (milliseconds).
*   **Code Context:**
    ```python
    blink_lengths_l_ms = matched_blinks["left"]["peak_internal_width"] * 1000 / fps # Convert width (frames) to milliseconds
    blink_lengths_r_ms = matched_blinks["right"]["peak_internal_width"] * 1000 / fps # Convert width (frames) to milliseconds

    statistics["Blink_Length_left_ms_avg"] = np.nanmean(blink_lengths_l_ms) # Average blink length in ms
    statistics["Blink_Length_left_ms_std"] = np.nanstd(blink_lengths_l_ms)  # Standard deviation of blink length in ms
    statistics["Blink_Length_right_ms_avg"] = np.nanmean(blink_lengths_r_ms) # Average blink length in ms
    statistics["Blink_Length_right_ms_std"] = np.nanstd(blink_lengths_r_ms) # Standard deviation of blink length in ms
    ```
    The code first converts the "peak_internal_width" (which is likely in frames) to milliseconds by multiplying by 1000 and dividing by `fps`. Then, it calculates the mean and standard deviation of these blink lengths using `np.nanmean` and `np.nanstd`.

**10. Partial_Blinks_min[NN]_left, Partial_Blinks_min[NN]_right**

*   **Description:** These statistics count the number of "partial" blinks specifically within each minute of the recording for the left and right eyes. `[NN]` is a placeholder for the minute number (e.g., `Partial_Blinks_min01_left` for the first minute, `Partial_Blinks_min02_left` for the second minute, and so on).
*   **Meaning:**  These minute-by-minute counts provide a temporal resolution to partial blink frequency. They can reveal if partial blink frequency changes over time within the recording. For example, an increase in partial blinks in later minutes might indicate fatigue setting in.
*   **Unit/Range:** $\mathbb{N}$ (Natural numbers). These are counts of blinks within a minute.
*   **Code Context:**
    ```python
    partial_l["minute"] = partial_l["apex_frame_og"]  / fps / 60 # Calculate minute of blink occurrence
    partial_times_l = pd.to_datetime(partial_l["minute"], unit='m', errors="ignore") # Convert minute to datetime (for grouping)
    partial_group_l = partial_l.groupby(partial_times_l.dt.minute) # Group partial blinks by minute

    i = 1
    for i, row in enumerate(partial_group_l.count()["minute"], start=1): # Iterate through minutes and counts
        statistics[f"Partial_Blinks_min{i:02d}_left"] = row # Store count for each minute
    while i <= math.ceil(length_l_min): # Ensure all minutes are accounted for, even if no blinks occurred
        statistics[f"Partial_Blinks_min{i:02d}_left"] = 0 # Set count to 0 for minutes with no blinks
        i += 1
    # ... (similar code for right eye) ...
    ```
    The code calculates the minute in which each partial blink occurs, groups the blinks by minute using pandas `groupby`, and then counts the number of blinks in each minute. It then stores these counts in the `statistics` dictionary with keys like "Partial_Blinks_min01_left", "Partial_Blinks_min02_left", etc. It also ensures that even minutes with zero partial blinks are included in the statistics with a count of 0.

**11. Complete_Blink_Total_left, Complete_Blink_Total_right, Complete_Frequency_left_bpm, Complete_Frequency_right_bpm**

*   **Description:** These statistics are analogous to the partial blink statistics (7 & 8), but they are for "complete" blinks.
    *   `Complete_Blink_Total`: Total count of complete blinks for each eye.
    *   `Complete_Frequency_bpm`: Frequency of complete blinks per minute for each eye.
*   **Meaning:** "Complete" blinks are those where the EAR value drops below the `Partial_Blink_threshold`.  Analyzing complete blink frequency and total count is essential for understanding normal blinking behavior and detecting deviations.
*   **Unit/Range:**
    *   `Complete_Blink_Total`: $\mathbb{N}$ (Natural numbers).
    *   `Complete_Frequency_bpm`: $1/\text{min}$.
*   **Code Context:**
    ```python
    complete_l = matched_blinks["left"][matched_blinks["left"]["blink_type"] == "complete"] # Filter for complete blinks
    complete_r = matched_blinks["right"][matched_blinks["right"]["blink_type"] == "complete"] # Filter for complete blinks

    statistics["Complete_Blink_Total_left"]  = len(complete_l) # Count of complete left blinks
    statistics["Complete_Blink_Total_right"] = len(complete_r) # Count of complete right blinks
    statistics["Complete_Frequency_left_bpm"]  = statistics["Complete_Blink_Total_left"] / length_l_min # Frequency
    statistics["Complete_Frequency_right_bpm"] = statistics["Complete_Blink_Total_right"] / length_r_min # Frequency
    ```
    The code is very similar to the partial blink calculations, just filtering for `blink_type == "complete"` instead of "partial".

**12. Complete_Blinks_min[NN]_left, Complete_Blinks_min[NN]_right**

*   **Description:**  Similar to partial blinks per minute (statistic 10), these statistics count the number of "complete" blinks within each minute of the recording for the left and right eyes.
*   **Meaning:** Minute-by-minute counts of complete blinks provide temporal resolution to complete blink frequency, allowing for analysis of how complete blink rate changes over time.
*   **Unit/Range:** $\mathbb{N}$ (Natural numbers).
*   **Code Context:**
    ```python
    complete_l["minute"] = complete_l["apex_frame_og"]  / fps / 60 # Calculate minute of blink
    complete_times_l = pd.to_datetime(complete_l["minute"], unit='m', errors="ignore") # Convert to datetime
    complete_group_l = complete_l.groupby(complete_times_l.dt.minute) # Group by minute

    for i, row in enumerate(complete_group_l.count()["minute"], start=1): # Iterate and count
        statistics[f"Complete_Blinks_min{i:02d}_left"] = row # Store count per minute
    while i <= math.ceil(length_l_min): # Ensure all minutes are included
        statistics[f"Complete_Blinks_min{i:02d}_left"] = 0 # Set 0 for minutes with no blinks
        i += 1
    # ... (similar code for right eye) ...
    ```
    Again, the code structure mirrors the partial blinks per minute calculation, but it operates on the `complete_l` and `complete_r` DataFrames to count complete blinks per minute.

**In Summary:**

These statistics comprehensively characterize blinking behavior by considering:

*   **Eye Aspect Ratio (EAR):**  Baseline openness, minimum and maximum openness.
*   **Blink Type:** Distinguishing between partial and complete blinks using thresholds.
*   **Blink Properties:** Prominence, width (duration), and height (depth) of blinks.
*   **Blink Frequency:** Total counts and frequencies (per minute) of both partial and complete blinks, both overall and minute-by-minute.
*   **Blink Duration Variability:** Standard deviation of blink lengths.

By analyzing these statistics, healthcare professionals can gain valuable insights into a patient's blinking patterns, which can be relevant for diagnosing and monitoring various medical conditions.