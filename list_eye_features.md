# Features derived from a 30-second epoch of 30Hz eye-tracking data
This document outlines a comprehensive set of features extracted from eyelid aperture data within 30-second epochs. These features are categorized by the aspect of blink behavior they capture – from individual blink event characteristics to the dynamics of the eye during inter-blink periods and overall signal complexity. Each feature is defined, its calculation method is briefly described, and its relevance, particularly in the context of fatigue or arousal, is discussed with supporting literature.

## 1. Blink Event Features (Aggregated per epoch)

These features quantify characteristics of individual blinks and are then aggregated (e.g., mean, standard deviation) over all blinks within a 30-second epoch.

### 1.1 Blink Frequency and Temporal Variability

This section details metrics related to how often blinks occur and the temporal spacing between them, providing insights into baseline blink rates and the regularity of blinking.

*   **blink_count**: Total number of blinks detected within the 30-second epoch.
*   **blink_rate**: Blinks per minute. Calculated as `blink_count * 2` for a 30-second epoch.
    *   **Rationale**: Blink rate is a fundamental measure. High rates can indicate cognitive load, anxiety, or ocular surface irritation, while abnormally low rates, or a decrease over time, are often linked to reduced arousal and fatigue. ([arxiv.org][1])
* 
### 1.2 Inter-Blink Interval (IBI) Features
The Inter-Blink Interval (IBI) is the duration between the end of one blink and the start of the next. A vector of IBIs is computed for each epoch.
    *   **Rationale**: IBI variability provides a rich signal mirroring changes in arousal, attention, and dopaminergic tone.

Below is a compact field-guide to various statistics derived from an epoch’s sequence of inter-blink intervals (IBIs). It explains what each statistic means, how it is calculated, and why analysts use it. Together, these give a layered view of central tendency, spread, short-term variability, and longer-range structure in blink timing.


#### 1.2.1. Basic Descriptive Statistics for IBIs

| Metric                            | Definition (on an epoch’s IBI vector `{I₁ … Iₙ}`) | Practical meaning                                                                                                      |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Mean (μ)**                      | Arithmetic average,   μ = ∑Iᵢ / n                 | Typical time between blinks – converts directly to blink rate (1/μ). ([arxiv.org][1])                                  |
| **Standard deviation (σ)**        | σ = √\[ ∑(Iᵢ – μ)² / (n–1) ]                      | Overall dispersion around the mean – higher values signal irregular blinking. ([lifewire.com][2])                      |
| **Median**                        | 50-th percentile of the sorted IBI list           | Resistant to outliers from unusually long “micro-sleep” blinks. ([lifewire.com][2])                                    |
| **Min / Max**                     | Extremes of the epoch’s IBI distribution          | Fastest and slowest blink spacing within the window. ([lifewire.com][2])                                               |
| **Coefficient of variation (CV)** | CV = σ / μ (unit-less)                            | Normalises spread to the mean so you can compare subjects with different baseline blink rates. ([en.wikipedia.org][3]) | 
---
#### 1.2.2. Short-Term Variability Metrics for IBI
##### 1.2.2.1 RMSSD (Root Mean Square of Successive Differences)


* Computes the typical jump from one blink interval to the next.
*  A large RMSSD means IBIs swing rapidly; a small RMSSD means fairly even pacing.  Originally a heart-rate-variability (HRV) measure – it maps well to blink timing because both are point-processes. ([pmc.ncbi.nlm.nih.gov][4], [pmc.ncbi.nlm.nih.gov][4], [biopac.com][5], [pubmed.ncbi.nlm.nih.gov][6])

$$
\text{RMSSD} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n-1}(I_{i+1}-I_i)^2}
$$

**Interpretation tip**: when you normalise RMSSD by the mean IBI you obtain a dimensionless index akin to short-term CV.

##### 1.2.2.2  Poincaré plot metrics (SD1 & SD2)

A Poincaré plot places each IBI against the next (Iᵢ vs Iᵢ₊₁).  Fitting an ellipse gives two orthogonal spreads:

* **SD1** – dispersion **perpendicular** to the line of identity; mirrors short-range, beat-to-beat variability (≈ RMSSD / √2).
* **SD2** – dispersion **along** the line; reflects longer-term oscillations in blink timing.

The SD1/SD2 ratio is often read as a “randomness” index – values ↑1 suggest erratic blinking, values ≪1 imply quasi-periodic behaviour. ([researchgate.net][7], [researchgate.net][7], [sciendo.com][8], [biomedical-engineering-online.biomedcentral.com][9])

---

#### 1.2.2 Complexity & memory-structure metrics

##### 1.2.2.1 Permutation entropy (PE)

PE turns short overlapping fragments of the IBI series into **ordinal patterns** (e.g., increasing, decreasing, plateau) and measures the Shannon entropy of their distribution.

* **High PE (→ log (k!))**: many patterns used with similar probability → unpredictable, high-complexity blink stream.
* **Low PE**: a few patterns dominate → strongly regular or highly constrained blinking.

It is noise-robust, works on short windows, and tracks attention/fatigue shifts faster than spectral measures. Choose embedding dimension *m* (3–5) and delay τ based on sampling rate. ([aptech.com][10], [arxiv.org][11], [valeman.medium.com][12])
* Practical Note: Choose embedding dimension m (typically 3–5) and delay τ based on sampling rate.

##### 1.2.2.2 Hurst exponent (H)

H gauges **long-range dependence**:

* **H ≈ 0.5** → the IBI series behaves like uncorrelated white noise (no memory).
* **H > 0.5** → persistent trend: long intervals tend to follow long intervals (possible drowsiness drift).
* **H < 0.5** → anti-persistent, mean-reverting behaviour (blink-rate quickly pulls back to baseline).

Computed via rescaled-range (R/S), DFA or variance-time methods on each epoch. ([macrosynergy.com][13], [pubsonline.informs.org][14])


##### 1.2.3 Implementation Notes

* Pre-compute the IBI series (from blink-onset and offset timestamps to successive differences of onsets or offsets).
* For every 30-s epoch, calculate the descriptives, variability, and complexity metrics.
* Aggregate across the session (mean, median, slope vs. time) to build subject-level features.
* Log each function’s entry/exit; wrap loops with tqdm; and keep DEBUG traces of intermediate vectors (e.g., successive‐difference array) for easy troubleshooting.

* These metrics distil complementary aspects of blink timing – from simple rate descriptors to subtle markers of short-term fluctuation and long-memory trends – giving your downstream models a richer view of ocular behaviour during tasks such as drowsy-driving assessment.

[1]: https://arxiv.org/pdf/1605.02037?utm_source=chatgpt.com "[PDF] Blink Rate Variability during resting and reading sessions - arXiv"
[2]: https://www.lifewire.com/how-to-find-variance-in-excel-4690114?utm_source=chatgpt.com "How to Calculate and Find Variance in Excel"
[3]: https://en.wikipedia.org/wiki/Coefficient_of_variation?utm_source=chatgpt.com "Coefficient of variation - Wikipedia"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/?utm_source=chatgpt.com "An Overview of Heart Rate Variability Metrics and Norms - PMC"
[5]: https://www.biopac.com/application/ecg-cardiology/advanced-feature/rmssd-for-hrv-analysis/?utm_source=chatgpt.com "RMSSD for HRV Analysis - ECG: Cardiology - BIOPAC"
[6]: https://pubmed.ncbi.nlm.nih.gov/15787862/?utm_source=chatgpt.com "Filter properties of root mean square successive difference (RMSSD ..."
[7]: https://www.researchgate.net/figure/The-Poincare-plot-SD1-and-SD2-standard-deviations-of-the-scattergram_fig1_290416554?utm_source=chatgpt.com "The Poincaré plot. SD1 and SD2 – standard deviations of the ..."
[8]: https://sciendo.com/pdf/10.2478/slgr-2013-0031?utm_source=chatgpt.com "[PDF] Poincaré Plots in Analysis of Selected Biomedical Signals - Sciendo"
[9]: https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-10-17?utm_source=chatgpt.com "Sensitivity of temporal heart rate variability in Poincaré plot to ..."
[10]: https://www.aptech.com/blog/permutation-entropy/?utm_source=chatgpt.com "Permutation Entropy - Aptech"
[11]: https://arxiv.org/abs/1905.06443?utm_source=chatgpt.com "On the Automatic Parameter Selection for Permutation Entropy"
[12]: https://valeman.medium.com/unlocking-predictive-power-harnessing-permutation-entropy-for-superior-time-series-forecasting-7772b52f4ec4?utm_source=chatgpt.com "Unlocking Predictive Power: Harnessing Permutation Entropy for ..."
[13]: https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/?utm_source=chatgpt.com "Detecting trends and mean reversion with the Hurst exponent"
[14]: https://pubsonline.informs.org/do/10.1287/LYTX.2012.04.05/full/?utm_source=chatgpt.com "The Hurst Exponent: Predictability of Time Series | Analytics Magazine"


### 1.3 Blink Morphology (Waveform Shape)
This section provides a detailed deep-dive into blink-waveform morphology. For each feature, a concise mathematical definition, a practical recipe for computing it from an eyelid-aperture (or EOG) waveform, and the empirical link to fatigue or drowsiness reported in the literature are given.

* blink_duration: Mean, std, median, min, max, cv, iqr
    * blink_duration_ratio: Ratio of longest to shortest duration
* time_to_peak/time_from_peak_to_end: Mean, std, cv
* blink_rise_time_25_75 / blink_fall_time: Mean, std, cv
* blink_fwhm: Mean, std, cv
* blink_amplitude: Mean, std, median, min, max, cv, skewness, kurtosis
* blink_area: Mean, std, cv
* blink_half_area_time: Mean, std, cv
* blink_asymmetry: Ratio of rise to fall time/slope (mean, std)
* blink_skewness/blink_kurtosis: Mean, std
* blink_inflection_count: Mean, std
  Below is a detailed deep-dive into **blink-waveform morphology**.
  For each feature I give (1) a concise mathematical definition, (2) a practical recipe for computing it from an eyelid-aperture (or EOG) waveform, and (3) the empirical link to fatigue or drowsiness reported in the literature.  Citations are placed after every claim that comes from a source.

---

### 1.3.1 Blink-Duration Metrics

#### 1.3.1.1 Per-blink duration

* **Definition** – The time elapsed from blink start to blink end: $d_i=t_{end,i}-t_{start,i}$.
* **Aggregate statistics** within a 30 s epoch:

$$
\begin{aligned}
\mu_d &=\tfrac{1}{n}\sum_{i=1}^{n} d_i\,,\qquad
\sigma_d &=\sqrt{\tfrac{1}{n-1}\sum_{i=1}^{n}(d_i-\mu_d)^2},\\[4pt]
\mathrm{CV}_d &=\sigma_d/\mu_d,\qquad
\text{IQR}_d &=Q_{75}-Q_{25}\, .
\end{aligned}
$$

*   **Rationale**: Longer mean duration ($\mu_d$) and larger spread ($\sigma_d$) appear reliably after sleep loss and during night-time driving, indicating increased drowsiness. ([pmc.ncbi.nlm.nih.gov][1], [iovs.arvojournals.org][2])

#### 1.3.1.2 Duration-ratio
*   **Definition**: The ratio of the longest blink duration to the shortest blink duration within the epoch:

$$
\text{Ratio}=\frac{\max\{d_i\}}{\min\{d_i\}}
$$

*   **Rationale**: Extreme ratios (> 5) can signify “slow” or micro-sleep blinks embedded in otherwise normal blinks, which are predictive of performance decrements like out-of-lane events. ([pmc.ncbi.nlm.nih.gov][3], [jcsm.aasm.org][4])

---

#### 1.3.2 Temporal Landmarks inside a Blink

#### 1.3.2.1 Time-to-peak & time-from-peak-to-end
*   **Definition**: Let $t_{pk,i}$ be the sample of maximum eyelid closure for blink $i$. 
  * Time-to-peak (rise time): $t^{\uparrow}_i=t_{pk,i}-t_{start,i}$
  *   Time-from-peak-to-end (fall time / reopening time): $t^{\downarrow}_i=t_{end,i}-t_{pk,i}$



*   **Rationale**: Slower eyelid closure (increased $t^{\uparrow}$) but even slower reopening (increased $t^{\downarrow}$) produce **asymmetry** typical of fatigued eyelids. ([ietresearch.onlinelibrary.wiley.com][5], [pmc.ncbi.nlm.nih.gov][6])
#### 1.3.2.2 Asymmetry index
**Definition**: A ratio comparing the closing and opening phases of a blink:
$$
\mathrm{AI}_i=\frac{t^{\uparrow}_i}{t^{\downarrow}_i}
\quad\text{or}\quad
\frac{|s^{\uparrow}_i|}{|s^{\downarrow}_i|}
$$

where $s^{\uparrow}$, $s^{\downarrow}$ are average slopes.


*   **Rationale**: Values much less than 1 (indicating a significantly slower fall/reopening phase) increase with cumulative wakefulness, serving as a robust indicator of fatigue. ([personales.upv.es][7], [diva-portal.org][8])
---

### 1.3.3 Rise & Fall Time Windows

#### 1.3.3.1 25 %–75 % rise time


*   **Definition**: The time taken for the eyelid aperture to go from 25% to 75% of its peak closure during the closing phase (rise time), and vice-versa for the reopening phase (fall time).
*   **Rationale**: Prolongation of these times reflects slowed orbicularis oculi (closing) and levator palpebrae (opening) muscle activity as alertness wanes. ([pmc.ncbi.nlm.nih.gov][6], [pmc.ncbi.nlm.nih.gov][9])


#### 1.3.3..2 Full-Width at Half-Maximum (FWHM)


*   **Definition**: The duration for which the eyelid aperture remains below 50% of its maximum closure during a blink:
    $$
    \text{FWHM}_i = t_{50\%,\text{down}} - t_{50\%,\text{up}}
    $$
    (Figure-based implementations typically use eyelid distance from fully open).
*   **Rationale**: FWHM expands during periods of night work and monotonous tasks, often preceding microsleeps by approximately 20 seconds, indicating prolonged closure. ([researchgate.net][10], [pmc.ncbi.nlm.nih.gov][9])


---

### 1.3.4 Amplitude- and Area-Based Measures

#### 1.3.4.1 Amplitude

*   **Definition**: $A_i = \max(\text{aperture baseline}) - \min(\text{aperture})$ during blink $i$. This represents the maximum extent of eyelid closure.
*   **Aggregate Moments**: Mean, max, plus higher-order descriptors aggregated per epoch:
    $$
    \text{Skewness} =
    \frac{\tfrac{1}{n}\sum_{i}(A_i-\mu_A)^3}{\sigma_A^{3}},\quad
    \text{Kurtosis} =
    \frac{\tfrac{1}{n}\sum_{i}(A_i-\mu_A)^4}{\sigma_A^{4}}-3 .
    $$
*   **Rationale**: Blink amplitude typically **shrinks** with eyelid fatigue. Simultaneously, the kurtosis of amplitude distribution often drops as sharp blinks are replaced by wider, flatter ones. ([mdpi.com][11], [arxiv.org][12])

#### 1.3.4.2 Blink area & half-area time

*   **Definition**:
    *   **Area ($S_i$)**: The integral of the absolute eyelid aperture deviation from baseline during a blink, $S_i=\int |x_i(t)|\,dt$ or $\sum |x(t)|\,\Delta t$. Represents the total "amount" of closure.
    *   **Half-Area Time**: The moment within the blink when the cumulative area reaches 50% of the total blink area ($S_i$).
*   **Rationale**: An increasing half-area time mirrors the general slowing of lid kinematics in drowsiness, as it takes longer to accumulate half the total closure area. ([mdpi.com][13], [pmc.ncbi.nlm.nih.gov][14])

---

### 1.3.5 Shape-Descriptor Features

#### 1.3.5.1 Global skewness & kurtosis of the **waveform**

*   **Definition**: Computed over the entire normalized blink trace $x(t)$ (or deviation from baseline):
    $$
    \text{skew}(x)=\frac{\mathbb{E}[(x-\mu)^3]}{\sigma^3},\quad
    \text{kurt}(x)=\frac{\mathbb{E}[(x-\mu)^4]}{\sigma^4}-3 .
    $$
*   **Rationale**: Fatigued blinks typically show **positive skew** (indicating a sluggish reopening phase) and **platykurtic** peaks (flatter, less sharp peaks), reflecting reduced neuromuscular control. ([mdpi.com][11], [pmc.ncbi.nlm.nih.gov][3])


#### 1.3.5.2 Number of inflection points


*   **Definition**: The count of zero-crossings of the second derivative $x''(t)$ within a single blink waveform.
*   **Rationale**: Voluntary or “reflex” blinks often contain extra kinematic phases, increasing the count. In contrast, spontaneous drowsy blinks tend to become smoother with reduced control, resulting in fewer inflection points. ([personales.upv.es][7], [media.isbweb.org][15])


---

### 1.3.6 Fatigue Links in a Nutshell

| Feature family      | Typical fatigue signature     | Key evidence                                                                     |
| ------------------- | ----------------------------- | -------------------------------------------------------------------------------- |
| Duration & FWHM     | ↑ mean, ↑ max, ↑ FWHM         | field & simulator driving ([pmc.ncbi.nlm.nih.gov][1], [pmc.ncbi.nlm.nih.gov][3]) |
| Rise/Fall asymmetry | Rise ≈ steady, Fall ↑         | lab eyelid EMG ([ietresearch.onlinelibrary.wiley.com][5])                        |
| Amplitude & area    | ↓ amplitude, ↑ half-area time | optical-sensor eyeglass ([ietresearch.onlinelibrary.wiley.com][5])               |
| Skew / kurtosis     | +skew, −kurtosis              | MDPI Sensors & high-speed video ([mdpi.com][11], [pmc.ncbi.nlm.nih.gov][6])      |
| Inflection count    | ↓ (count)                     | voluntary vs. spontaneous comparison ([personales.upv.es][7])                    |

---

### 1.3.7 Implementation Pointers

1. **Detect landmarks** (start, peak, end) via threshold on velocity (Savitzky-Golay filtered) ([mdpi.com][13]).
2. Log entry/exit of each computation function, and wrap epoch loops with `tqdm`.
3. Store per-blink vectors in a `pandas.DataFrame`; then derive epoch-statistics by `.agg(['mean','std','max', ...])`.

Together, these waveform-shape features capture the **slowing, flattening and asymmetry** that eyelids exhibit as central nervous system arousal drops, providing sensitive predictors for driver-fatigue and attention analytics.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3174834/?utm_source=chatgpt.com "The Characteristics of Sleepiness During Real Driving at Night—A ..."
[2]: https://iovs.arvojournals.org/article.aspx?articleid=2188061&utm_source=chatgpt.com "Blink Frequency and Duration during Perimetry and Their ... - IOVS"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6760410/?utm_source=chatgpt.com "Eye-Blink Parameters Detect On-Road Track-Driving Impairment ..."
[4]: https://jcsm.aasm.org/doi/abs/10.5664/jcsm.7918?utm_source=chatgpt.com "Eye-Blink Parameters Detect On-Road Track-Driving Impairment ..."
[5]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/mnl.2017.0136?utm_source=chatgpt.com "Fatigue evaluation by detecting blink behaviour using eyeglass ..."
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5937736/?utm_source=chatgpt.com "Blinking characterization from high speed video records. Application ..."
[7]: https://personales.upv.es/thinkmind/dl/conferences/achi/achi_2013/achi_2013_17_20_20212.pdf?utm_source=chatgpt.com "[PDF] Paper Title (use style: paper title) - UPV"
[8]: https://www.diva-portal.org/smash/get/diva2%3A673983/FULLTEXT01.pdf?utm_source=chatgpt.com "[PDF] Blink behaviour based drowsiness detection - DiVA portal"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9920860/?utm_source=chatgpt.com "System and Method for Driver Drowsiness Detection Using ..."
[10]: https://www.researchgate.net/figure/Blink-peak-width-calculation-using-the-full-width-at-half-maximum-fwhm_fig3_324827676?utm_source=chatgpt.com "Blink peak width calculation using the full width at half maximum ..."
[11]: https://www.mdpi.com/1424-8220/23/11/5339?utm_source=chatgpt.com "A Hardware-Based Configurable Algorithm for Eye Blink Signal ..."
[12]: https://arxiv.org/pdf/2009.13276?utm_source=chatgpt.com "[PDF] Driver Drowsiness Classification Based on Eye Blink and Head ..."
[13]: https://www.mdpi.com/2078-2489/9/4/93?utm_source=chatgpt.com "Robust Eye Blink Detection Based on Eye Landmarks and Savitzky ..."
[14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8998332/?utm_source=chatgpt.com "Relationship between Eye Blink Frequency and Incremental ..."
[15]: https://media.isbweb.org/images/conf/2007/ISB/0214.pdf?utm_source=chatgpt.com "[PDF] SUBMISSION GUIDELINES FOR 2007 ISB CONGRESS"


## 1.4.  Blink Kinematics (Movement Dynamics)

This section focuses on **blink-kinematic features**—velocity, acceleration, jerk, and the Amplitude-Velocity Ratio (AVR). After defining each quantity and its summary statistics (max, mean, SD, CV), it describes how they are extracted from an eyelid-displacement trace and why they shift when a person becomes fatigued.

---

### Summary

Blink kinematics are obtained by successively differentiating the eyelid-aperture signal $x(t)$.

* **Velocity** $v(t)=\tfrac{dx}{dt}$ captures how fast the lid moves.
* **Acceleration** $a(t)=\tfrac{d^2x}{dt^2}$ describes how quickly that speed changes.
* **Jerk** $j(t)=\tfrac{d^3x}{dt^3}$ reports the smoothness of the motion.
* **amplitude_velocity_ratio AVR** compares motion *extent* (amplitude) to its *peak velocity* and rises when the same closure distance is achieved more slowly.

Across many studies, fatigue slows the lid in both directions, flattens acceleration peaks, and increases AVR—making these metrics prime inputs for drowsiness-detection systems. ([pmc.ncbi.nlm.nih.gov][1], [pmc.ncbi.nlm.nih.gov][2], [pubmed.ncbi.nlm.nih.gov][3], [pubmed.ncbi.nlm.nih.gov][4])

---

### 1.4.1 Blink Velocity

For each blink $i$, the **instantaneous velocity** is the first derivative of aperture:

$$
v_i(t)=\frac{dx_i(t)}{dt}.
$$

Key scalars are extracted:

$$
v_{\text{max},i}=\max_t|v_i(t)|,\qquad
\bar v_i=\frac{1}{T_i}\int_{t_{s,i}}^{t_{e,i}}|v_i(t)|dt,
$$

with $T_i$ the blink duration.



Given an epoch’s $n$ blinks, compute

$$
\mu_v=\frac{1}{n}\sum_{i=1}^{n}v_{\text{max},i},\;
\sigma_v=\sqrt{\tfrac{1}{n-1}\sum (v_{\text{max},i}-\mu_v)^2},\;
\text{CV}_v=\sigma_v/\mu_v.
$$



Field and simulator studies show peak-closure velocity drops by 20–40 % after sleep restriction or long driving sessions ([pmc.ncbi.nlm.nih.gov][1], [pmc.ncbi.nlm.nih.gov][5], [tobii.com][6], [pubmed.ncbi.nlm.nih.gov][7], [pubmed.ncbi.nlm.nih.gov][4]), reflecting slower orbicularis-oculi activation.  Lower velocity lengthens PERCLOS and raises accident risk.

---

### 1.4.2 Blink Acceleration

Acceleration is the second derivative of aperture:

$$
a_i(t)=\frac{d^2x_i(t)}{dt^2}=\frac{dv_i(t)}{dt}.
$$

Extract $a_{\text{max},i}$ (absolute peak), $\bar a_i$ (mean magnitude), then aggregate as for velocity.



Reduced neural drive and viscoelastic damping of eyelid tissues attenuate acceleration peaks; drowsy participants show 25–50 % smaller $a_{\text{max}}$ and larger inter-subject variance ([pmc.ncbi.nlm.nih.gov][2], [jcsm.aasm.org][8], [pubmed.ncbi.nlm.nih.gov][9], [pmc.ncbi.nlm.nih.gov][10]).  Lower acceleration is a precursor to slow, incomplete closures linked with microsleeps.

---

#### 1.4.3. Blink Jerk

Jerk is the third derivative of aperture, measuring how quickly acceleration changes:

$$
j_i(t)=\frac{d^3x_i(t)}{dt^3}=\frac{da_i(t)}{dt}.
$$

Peak jerk $j_{\text{max},i}$ quantifies how abruptly muscle force is applied or released.

* Some practical notes on computing jerk:
    * Use a Savitzky–Golay filter to smooth the aperture signal before differentiating, as jerk is sensitive to noise.
    * Compute jerk as the third derivative of the aperture signal, or as the second derivative of velocity.
    * Aggregate jerk statistics per epoch: mean, max, standard deviation, and coefficient of variation (CV).




High-speed-video work reports smoother (lower-jerk) profiles as the oculomotor system tires, especially during reopening ([pmc.ncbi.nlm.nih.gov][5], [asmedigitalcollection.asme.org][11], [arxiv.org][12]).  Decreased jerk correlates with increased reaction-time lapses in sustained-attention tasks.

---

#### 1.4.4 Amplitude-Velocity Ratio (AVR)

For a blink of amplitude $A_i=\max x_i - \min x_i$ and peak velocity $v_{\text{max},i}$,

$$
\text{AVR}_i=\frac{A_i}{v_{\text{max},i}}.
$$

Positive AVR (closing), negative AVR (reopening) are often analysed separately.

######  1.4.4.1  Epoch descriptors

$$
\mu_{\text{AVR}},\;\sigma_{\text{AVR}},\;
\text{CV}_{\text{AVR}}=\sigma_{\text{AVR}}/\mu_{\text{AVR}}.
$$



AVR rises because $v_{\text{max}}$ diminishes faster than amplitude when alertness drops ([researchgate.net][13], [pmc.ncbi.nlm.nih.gov][2], [pubmed.ncbi.nlm.nih.gov][3], [jcsm.aasm.org][14], [asmedigitalcollection.asme.org][11]).  Thresholds around AVR > 0.25 s have predicted out-of-lane driving events with ≥80 % sensitivity ([jcsm.aasm.org][8], [sciencedirect.com][15]).



---

### Key References

1. Eye-blink parameters detect on-road driving impairment ([pmc.ncbi.nlm.nih.gov][1])
2. Amplitude-velocity ratio of blinks & drowsiness monitoring ([researchgate.net][13])
3. Accuracy of eyelid movement parameters for drowsiness detection ([pmc.ncbi.nlm.nih.gov][2])
4. Effects of drowsiness on eyelid velocity (PubMed) ([pubmed.ncbi.nlm.nih.gov][3])
5. Tobii article on blink dynamics & fatigue ([tobii.com][6])
6. Naturalistic driving: blink measures & OSA ([sciencedirect.com][15])
7. Electrooculographic indices of pilot fatigue ([pubmed.ncbi.nlm.nih.gov][7])
8. Shift-work sleepiness & AVRs ([pubmed.ncbi.nlm.nih.gov][4])
9. Review of blinking kinematics (Physiology) ([pubmed.ncbi.nlm.nih.gov][9])
10. ASME high-speed eyelid-motion tracking ([asmedigitalcollection.asme.org][11])

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6760410/?utm_source=chatgpt.com "Eye-Blink Parameters Detect On-Road Track-Driving Impairment ..."
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3836343/?utm_source=chatgpt.com "The Accuracy of Eyelid Movement Parameters for Drowsiness ..."
[3]: https://pubmed.ncbi.nlm.nih.gov/33352535/?utm_source=chatgpt.com "The effects of unintentional drowsiness on the velocity of eyelid ..."
[4]: https://pubmed.ncbi.nlm.nih.gov/26094925/?utm_source=chatgpt.com "Ocular Measures of Sleepiness Are Increased in Night Shift Workers ..."
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5937736/?utm_source=chatgpt.com "Blinking characterization from high speed video records. Application ..."
[6]: https://www.tobii.com/resource-center/learn-articles/blinks-a-hidden-gem-in-eye-tracking-research?utm_source=chatgpt.com "Blinks - a hidden gem in eye tracking research - Tobii"
[7]: https://pubmed.ncbi.nlm.nih.gov/8652752/?utm_source=chatgpt.com "Electrooculographic and performance indices of fatigue during ..."
[8]: https://jcsm.aasm.org/doi/abs/10.5664/jcsm.7918?utm_source=chatgpt.com "Eye-Blink Parameters Detect On-Road Track-Driving Impairment ..."
[9]: https://pubmed.ncbi.nlm.nih.gov/12612018/?utm_source=chatgpt.com "Eyelid movements: behavioral studies of blinking in humans under ..."
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6187843/?utm_source=chatgpt.com "Fatigue Assessment by Blink Detected with Attachable Optical ..."
[11]: https://asmedigitalcollection.asme.org/biomechanical/article/147/1/014503/1208594/Eyelid-Motion-Tracking-During-Blinking-Using-High?utm_source=chatgpt.com "Eyelid Motion Tracking During Blinking Using High-Speed Imaging ..."
[12]: https://arxiv.org/abs/2407.02222?utm_source=chatgpt.com "Detecting Driver Fatigue With Eye Blink Behavior"
[13]: https://www.researchgate.net/publication/289510542_The_amplitude-velocity_ratio_of_blinks_A_new_method_for_monitoring_drowsiness?utm_source=chatgpt.com "(PDF) The amplitude-velocity ratio of blinks: A new method for ..."
[14]: https://jcsm.aasm.org/doi/pdf/10.5664/jcsm.7918?utm_source=chatgpt.com "Eye-Blink Parameters Detect On-Road Track-Driving Impairment ..."
[15]: https://www.sciencedirect.com/science/article/abs/pii/S2352721821000577?utm_source=chatgpt.com "Eye blink parameters to indicate drowsiness during naturalistic ..."

### 1.5. Blink Energy and Complexity Features

This section describes features that quantify the overall "effort" or intricate movement patterns within individual blinks.

This section describes features that quantify the overall "effort" or intricate movement patterns within individual blinks.

*   **blink_signal_energy (Mean, std, cv)**
    *   **Definition**: The integral of the squared eyelid aperture signal over the duration of each blink. $E_i = \int_{t_{start,i}}^{t_{end,i}} x_i(t)^2 dt$ or $\sum x_i(t)^2 \Delta t$.
    *   **Rationale**: Represents the total power or intensity of the eyelid movement during closure and reopening. Lower energy blinks might indicate reduced neuromuscular effort associated with fatigue.
    *   **Citation**: Common in signal processing for activity quantification; applied to blinks in fatigue studies [General signal processing, e.g., similar to acoustic energy metrics].

*   **teager_kaiser_energy (Mean, std, cv)**
    *   **Definition**: Applies the Teager-Kaiser Energy Operator ($\Psi[x(t)] = x(t)^2 - x(t-1)x(t+1)$) to the eyelid aperture signal for each blink, then aggregates the results.
    *   **Rationale**: Measures the instantaneous energy of oscillations, capturing rapid changes in amplitude and frequency. It is sensitive to subtle variations in motor control and could reflect instability or compensatory mechanisms in fatigued states.
    *   **Citation**: Widely used in biosignal processing (e.g., EMG, speech) [Nonlinear signal processing, e.g., similar to muscle activity].

*   **blink_line_length (Mean, std, cv)**
    *   **Definition**: The sum of absolute differences between consecutive eyelid aperture samples during a blink: $\sum_{t=t_{start,i}}^{t_{end,i}-1} |x_i(t+1) - x_i(t)|$. This represents the total path length traversed by the eyelid.
    *   **Rationale**: A measure of the total kinematic activity or "wobble" within the blink. Increased line length could suggest more erratic or less smooth movements, potentially due to impaired motor control in fatigue.
    *   **Citation**: Used in EMG analysis for muscle activation duration [Bioelectrical signal analysis, e.g., similar to EMG line length].

*   **Integral of absolute velocity (Mean, std, cv)**
    *   **Definition**: The integral of the absolute instantaneous velocity of the eyelid during a blink: $\int_{t_{start,i}}^{t_{end,i}} |v_i(t)| dt$. This is also a measure of total path length.
    *   **Rationale**: Quantifies the total distance the eyelid travels during a blink, irrespective of direction. It is a robust measure of overall eyelid excursion and kinetic effort. A reduction might signify shallower, less complete blinks in fatigue.
    *   **Citation**: Related to kinematic measures of movement quality [General biomechanics, e.g., similar to total displacement].


## 2. Ocular Activity During Inter-Blink Periods (Aggregated per epoch)

These features describe the state and dynamics of the eye during periods when it is considered "open," between distinct blink events.

### 2.1. Baseline and Drift

*   **baseline_mean**: Mean eyelid position between blinks.
    *   **Definition**: The average eyelid aperture (e.g., percentage of full opening) within an epoch, specifically calculated during periods when no blink is detected.
    *   **Rationale**: A decrease in this mean value can indicate partial lid closures or a generally lower "open" state, a common sign of developing drowsiness or reduced vigilance.
    *   **Citation**: Fundamental in PERCLOS-related research [General fatigue research].
*   **baseline_drift**: Slope of linear fit to baseline.
    *   **Definition**: The slope of a linear regression line fitted to the eyelid aperture signal during the inter-blink periods across the epoch.
    *   **Rationale**: A negative slope indicates a slow, gradual drooping of the eyelid, characteristic of increasing drowsiness or cumulative fatigue, even before full blinks become prolonged.
    *   **Citation**: Related to sustained eyelid closure detection [Drowsiness detection literature].
*   **baseline_std/baseline_mad**: Variability of baseline.
    *   **Definition**: Standard deviation (std) or Median Absolute Deviation (MAD) of the eyelid aperture signal during inter-blink periods.
    *   **Rationale**: Increased variability suggests instability in oculomotor control, which can arise from mental fatigue or reduced vigilance. It reflects the "steadiness" of eye opening.
    *   **Citation**: Links to ocular motor control stability [Oculomotor fatigue literature].
*   **low_freq_baseline_power (<0.1 Hz)**: Power in low frequency band.
    *   **Definition**: The power spectral density of the eyelid aperture signal (during open periods) in the very low frequency band (e.g., 0.01-0.1 Hz).
    *   **Rationale**: Reflects slow, "rolling" eyelid movements or very subtle partial closures not captured as discrete blinks. Increased power in this band can be a marker of emerging drowsiness or instability in tonic eyelid position.
    *   **Citation**: Similar to slow eye movement analysis in fatigue [Eyelid oscillation and fatigue research].

### 2.2. Eye Opening and Closure Dynamics

*   **perclos**: Percentage of eyelid closure.
    *   **Definition**: The percentage of time within an epoch that the eyelid is closed or nearly closed (e.g., below a certain threshold like 70-80% of individual maximum opening).
    *   **Rationale**: A gold-standard measure of drowsiness. High PERCLOS values (e.g., >80% closure for >1.5-2 seconds) are strongly correlated with impaired performance, increased accident risk, and microsleeps.
    *   **Citation**: Widely used in driver fatigue research [SAE J2386 standard, numerous studies].
*   **eye_opening_rms**: Root mean square of eyelid opening amplitude.
    *   **Definition**: The Root Mean Square (RMS) of the eyelid aperture signal during inter-blink periods. This measures the overall fluctuation or "noisiness" of the open eye signal.
    *   **Rationale**: Could capture subtle, non-blink related movements or micro-fluctuations in eyelid position that are not captured by mean or standard deviation. Increased RMS might indicate oculomotor instability in fatigue.
    *   **Citation**: Often used in general signal noise analysis; application to eyelid stability [Similar to physiological tremor quantification].
*   **micropause_count**: Count of partial closures (100ms - 300ms).
    *   **Definition**: The number of instances within an epoch where the eyelid partially closes (e.g., drops below 50% of maximum opening) for a brief duration (e.g., 100-300ms) but doesn't meet the criteria for a full blink.
    *   **Rationale**: Micropauses are indicators of momentary lapses in attention or early stages of drowsiness, reflecting partial lid control loss and a struggle to maintain full eye opening.
    *   **Citation**: Specific research on partial blinks and vigilance [Micropause research in drowsiness].

### 2.3. Inter-Blink Signal Complexity

*   **inter_blink_variance/inter_blink_mad**:
    *   **Definition**: Variance or Median Absolute Deviation (MAD) of the eyelid aperture signal *only* during the inter-blink periods (i.e., when the eye is considered open).
    *   **Rationale**: Quantifies the overall dispersion of the eyelid position when supposedly stable. Increased variability suggests reduced stability of oculomotor control, often seen in fatigue, or the presence of subtle lid movements not classified as blinks.
    *   **Citation**: General statistical measures applied to eyelid stability [Similar to postural sway analysis].
*   **non_blink_spectral_entropy**:
    *   **Definition**: Shannon entropy applied to the normalized power spectrum of the eyelid aperture signal, specifically calculated from the periods *between* blinks.
    *   **Rationale**: High entropy indicates a broad, flat spectrum (more random or irregular behavior), while low entropy suggests dominant frequencies (more regular or periodic behavior). Changes can indicate shifts in underlying neural control patterns in fatigue.
    *   **Citation**: Used in biosignal analysis for complexity and predictability [Similar to EEG spectral entropy in fatigue].
*   **approximate_entropy / sample_entropy**:
    *   **Definition**: Measures the predictability and regularity of the eyelid aperture time series *during inter-blink periods*. Lower values indicate more regularity and predictability.
    *   **Rationale**: These metrics often decrease with increasing drowsiness, as the oculomotor system becomes more regular and less adaptive in its open-eye state, reflecting a shift towards more automatic or constrained control.
    *   **Citation**: Common in physiological signal analysis for complexity and regularity (e.g., HRV, EEG) [Physiological signal complexity in fatigue].
*   **Zero-crossing rate of 1st derivative**:
    *   **Definition**: The number of times the first derivative (velocity) of the eyelid aperture signal crosses zero *within inter-blink periods*.
    *   **Rationale**: Detects rapid changes in the direction of lid movement. A high rate could indicate eyelid tremor, micro-oscillations, or unstable eye opening, which can be associated with fatigue or stress.
    *   **Citation**: Used in tremor analysis; application to eyelid micro-movements [Biomedical signal processing for tremors].




## 3. Frequency Domain Features (Per Epoch)

These features are derived by analyzing the frequency content of the eyelid aperture signal over the entire epoch, providing insights into rhythmic components of ocular activity.

### 3.1. Blink-Related Rhythms

*   **blink_rate_peak_frequency (0.1-0.5 Hz)**: Dominant blink periodicity.
    *   **Definition**: The dominant frequency component in the power spectrum derived from the sequence of blink onset times (e.g., by treating blink occurrences as a point process and computing its power spectrum).
    *   **Rationale**: Identifies a preferred blinking rhythm. Changes in this rhythm (e.g., towards slower, more periodic blinks) can indicate shifts in cognitive state or arousal, as the internal "blink pacemaker" adjusts.
    *   **Citation**: Research on blink pacemakers and vigilance [Oculomotor control and circadian rhythms].
*   **blink_rate_peak_power**: Power at dominant blink periodicity.
    *   **Definition**: The power (amplitude squared) at the identified `blink_rate_peak_frequency`.
    *   **Rationale**: Indicates the strength or prominence of the dominant blinking rhythm. Higher power suggests a more synchronized and regular blinking pattern, potentially reflecting a more automatic or less variable state.
    *   **Citation**: Similar to above, quantifies rhythmic strength [Power spectral analysis in biological rhythms].

### 3.2. General Eye Movement Rhythms

These features are computed from the power spectrum of the raw eyelid aperture signal (or its envelope) over the entire epoch.

*   **broadband_power (0.5-2 Hz)**: Voluntary/stress-related movements.
    *   **Definition**: Total power spectral density of the raw eyelid aperture signal within a broad frequency band (e.g., 0.5 Hz to 2 Hz), chosen to capture various non-blink ocular movements.
    *   **Rationale**: This range might capture various slow eyelid movements, large eye movements, or artifacts from head movements. Changes in this power could reflect increased restlessness, gross motor instability, or specific task-related ocular adjustments in response to fatigue or stress.
    *   **Citation**: General spectral analysis of physiological signals; specific band relevance might be empirical [Physiological artifacts in eye tracking].
*   **spectral_centroid (0.5-2 Hz)**:
    *   **Definition**: The "center of mass" of the power spectrum within the specified frequency band (e.g., 0.5-2 Hz), calculated as $\sum (f \cdot P(f)) / \sum P(f)$.
    *   **Rationale**: A higher centroid indicates that more power is concentrated at higher frequencies within the band. This can reflect changes in the speed or "sharpness" of general ocular movements, potentially indicating an increase in faster, less controlled movements.
    *   **Citation**: Common in audio signal processing, adapted for biosignals [Signal processing spectral moments].
*   **total_energy (2-13 Hz)**: Rapid micro-blinks vs. noise.
    *   **Definition**: Total power in the eyelid aperture signal within a higher frequency band (e.g., 2 Hz to 13 Hz).
    *   **Rationale**: This band might capture very rapid micro-movements, subtle tremors, or fine motor control fluctuations of the eyelid. Increased energy here could signal heightened physiological noise, or specific compensatory mechanisms in fatigue (e.g., micro-tremors).
    *   **Citation**: Eyelid tremor/oscillation studies [Physiological tremor and fatigue].
*   **spectral_entropy (2-13 Hz)**:
    *   **Definition**: Shannon entropy of the normalized power spectrum within the 2-13 Hz band.
    *   **Rationale**: Measures the "flatness" or "randomness" of the power distribution in this higher frequency range. A higher entropy indicates a more uniform distribution of power across frequencies (more "noise-like"), while lower entropy suggests more distinct peaks (more structured activity).
    *   **Citation**: General signal processing, specific to biological signals [Time-series entropy].
*   **1/f_slope**: Slope of power spectrum in log-log space.
    *   **Definition**: The exponent $\alpha$ in the power spectrum relationship $P(f) \propto 1/f^\alpha$, typically derived from a linear fit to the log-log plot of power vs. frequency for the raw eyelid signal (or its envelope) across a wide frequency range.
    *   **Rationale**: Characterizes the fractal or self-similar nature of the signal. A shift from white noise ($\alpha \approx 0$) towards pink noise ($\alpha \approx 1$) or Brownian noise ($\alpha \approx 2$) can indicate changes in underlying physiological control, potentially linked to arousal, attention, and cognitive state.
    *   **Citation**: Used in physiological signals (e.g., EEG, HRV) to characterize complexity and long-range correlations [Fractal analysis in biological signals].
*   **band_power_ratios**: Ratios of power in different frequency bands.
    *   **Definition**: Provide a normalized comparison of power distribution across different physiological rhythms (e.g., (0.01-0.1 Hz) / (0.1-0.5 Hz) or (0.5-2 Hz) / (2-13 Hz)).
    *   **Rationale**: Specific ratios might be more sensitive indicators of fatigue or stress than absolute power in a single band, as they capture relative shifts in ocular dynamics.
    *   **Citation**: Common in EEG and HRV analysis for assessing state changes [Ratio metrics in spectral analysis].
*   **wavelet_packet_energy (D1-D4 levels)**: Wavelet-based energy features.
    *   **Definition**: Energy coefficients derived from a Discrete Wavelet Packet Transform (DWPT) of the eyelid aperture signal, at specified decomposition levels (e.g., D1-D4 corresponding to specific frequency ranges).
    *   **Rationale**: Wavelets offer a multi-resolution analysis, capturing transient events and non-stationary patterns more effectively than traditional FFT. Energy in specific wavelet bands can reflect different types of lid movements or physiological states at different time scales, providing a richer temporal-frequency signature of fatigue.
    *   **Citation**: Used in time-series analysis for complex and non-stationary signals, particularly biosignals [Wavelet analysis in biomedical signals].


## 4. Advanced and Experimental Features

These features represent more complex or less commonly adopted measures, often requiring specific signal processing techniques or reflecting subtle aspects of oculomotor control.

*   **blink_magnitude_spectrum (FFT of individual blinks)**
    *   **Definition**: The magnitude (amplitude) spectrum obtained by applying a Fast Fourier Transform (FFT) to the detrended eyelid aperture waveform of each *individual* blink.
    *   **Rationale**: Characterizes the frequency components present within a single blink event. Changes due to fatigue might alter the smoothness or speed of the blink, which would be reflected in the higher frequency content of its spectrum (e.g., a "sharper" blink having more high-frequency components).
    *   **Citation**: Experimental application of spectral analysis to individual events [Similar to single-trial EEG analysis].
*   **blink_wavelet_coefficients**
    *   **Definition**: Coefficients derived from applying a Continuous or Discrete Wavelet Transform (CWT/DWT) to the individual eyelid aperture waveform of each blink.
    *   **Rationale**: Provides a time-frequency representation of the blink. Wavelet coefficients at different scales (frequencies) and positions (time within blink) can capture fine-grained details of blink kinematics, such as sudden changes in velocity or specific oscillatory components within the closing/opening phases, potentially revealing subtle fatigue markers.
    *   **Citation**: Advanced signal processing for transient events [Wavelet analysis of short-duration signals].
*   **clustered_blink_count**: Blinks occurring very close together.
    *   **Definition**: The number of times multiple blinks (e.g., three or more) occur within a very short, pre-defined time window (e.g., 2-3 blinks within 1-2 seconds), going beyond simple microbursts (pairs). This implies a more concentrated series of blinks.
    *   **Rationale**: Suggests periods of intense ocular activity, possibly due to eye strain, dry eyes, or a specific compensatory coping mechanism for drowsiness (e.g., repeated attempts to clear vision or re-engage).
    *   **Citation**: Could be related to compensatory blinking patterns [Blinking patterns in ocular discomfort].
*   **microblink_ratio**
    *   **Definition**: The proportion of total blinks within an epoch that are classified as "microblinks" (i.e., blinks with a very short duration, e.g., < 100ms or < 50% of typical blink duration).
    *   **Rationale**: Microblinks are often associated with states of high sustained attention or cognitive load, where a full closure might be disruptive. An increase in this ratio could indicate a focused state, while a decrease might signify a shift towards longer, drowsy blinks.
    *   **Citation**: Research on microblinks and cognitive load [Microblink studies in attention].
*   **blink_burstiness_index: Variance(IBI) / Mean(IBI)**
    *   **Definition**: The ratio of the variance of inter-blink intervals (IBI) to their mean. This is equivalent to the squared coefficient of variation for IBIs (`CV^2`).
    *   **Rationale**: For a purely random (Poisson) process, this ratio is 1. Deviations indicate "burstiness" (>1, implying blinks occur in clusters) or regularity (<1, implying more evenly spaced blinks). A high burstiness index suggests that blinks are occurring in clusters rather than being evenly spaced, which can be a sign of fluctuating attention or the onset of drowsiness (e.g., periods of normal blinking interspersed with periods of very short or very long IBIs).
    *   **Citation**: Used in neuroscience for spike train analysis, applicable to point processes [Neuroscience burstiness metrics].
*   **pupil_jump_indicator**: Sudden large baseline shifts (potentially saccades).
    *   **Definition**: A count or measure of sudden, large, transient changes in pupil diameter (e.g., >20% change in <100ms) not directly correlated with blinks, assuming pupil data is available from the eye-tracker.
    *   **Rationale**: Often indicates rapid eye movements (saccades), attention shifts, or physiological responses not directly related to eyelid activity. Can be used to infer cognitive processes, arousal state, or potential data artifacts.
    *   **Citation**: Pupilometry research on saccades and attention [Pupillometry and cognitive load].




## 5. Higher-Level Analysis and Practical Considerations

Beyond epoch-level feature extraction, these methods provide context and deeper insights into ocular behavior across longer timescales or in relation to external events.

*   **Session-level histograms (e.g., IBI distribution)**:
    *   **Application**: Visualizing the distribution of features (like IBIs, blink durations) over an entire experimental session provides a macroscopic view, revealing overall patterns, multi-modal distributions, or shifts in central tendency.
*   **Trend-over-time analysis (e.g., blink rate vs. time)**:
    *   **Application**: Tracking how features change over the duration of a task or session (e.g., using moving averages or linear regression). This is critical for assessing cumulative fatigue, adaptation, or learning effects.
*   **Event-triggered averages (aligned to external events)**:
    *   **Application**: Averaging feature values (or raw signal segments) time-locked to specific external events (e.g., critical incidents, task switches, stimuli presentations). This helps elucidate how ocular dynamics respond to and are modulated by environmental cues or cognitive demands.

These advanced analytical approaches allow for a comprehensive understanding of ocular behavior, bridging the gap between raw eye-tracking data and interpretable insights for various applications, particularly in human factors and vigilance monitoring.


Great. I’ll conduct a deep dive into academic and peer-reviewed literature to uncover innovative time-series features, composite metrics, and analytical techniques—including those using EEG, EOG, or video-based facial monitoring—that correlate blink behavior with fatigue or drowsiness.

I’ll focus on features or methodologies that are not already represented in your reference material, particularly those with novelty in signal processing, machine learning, or real-time system design. I’ll let you know once the comprehensive report is ready.


# Novel Time-Series Blink Features and Drowsiness Detection Techniques

## Advanced Blink-Derived Time-Series Features

**Amplitude-Velocity Ratio (AVR) and Blink Speed:** A key novel feature is the ratio of blink amplitude to peak closing velocity (AVR). In alert individuals, blink amplitude and peak closing speed are tightly correlated (large blinks close faster), but with drowsiness **blinks become abnormally slow for their size**. Johns et al. introduced AVR as a calibration-free index of this slowing: the amplitude/velocity ratio (with units of time) increases as eyelid movements become sluggish during fatigue. Studies show that a **higher AVR strongly indicates drowsiness**, even more than basic blink rate or duration. For example, recent explainable-machine-learning analysis found **blink duration and AVR to be the two most influential EOG features** for detecting fatigue; blinks longer than \~0.15 s and AVR above \~0.075 were clear predictors of a drowsy state. The practical value of AVR is highlighted in real-time systems like Johns’ **Drowsiness Score (JDS)** – a composite 0–10 scale computed from blink velocity metrics and other eyelid parameters. JDS leverages AVR and related measures to continuously score driver alertness (scores ≥4.5 indicate caution, ≥5 high risk). By capturing slowed eyelid dynamics without needing eye-calibration, AVR-based indices have proven effective for real-time monitoring.

**Eyelid Closure and Opening Dynamics:** Beyond total blink duration, researchers have explored the **separate phases of blink motion**. Fatigue affects the eyelid closing phase and reopening phase differently. Novel features include the **peak closing velocity (PCV)** and peak opening velocity of the eyelids, as well as the durations of each phase. Under drowsiness, the closing phase often slows and lengthens, sometimes with a noticeable delay before reopening. For instance, one system evaluated **Maximum Closing Duration (MCD)** – the longest single eyelid closure interval – as an indicator of microsleeps. Similarly, **Closing Time** and **Opening Time** can be measured per blink (e.g. time from blink start to fully closed, and closed to open). A 2008 study combining eyelid and heart signals observed that in sleepy states **blinks exceeded 400 ms and eyelid reopening time increased significantly**, reflecting very sluggish eye-opening after a blink. These fine-grained time measures provide insight into how drowsiness induces partial or delayed lid movements. **Closing vs. opening speed asymmetry** has also been studied: drowsy blinks often close more slowly and reopen weakly, so features like the ratio of closing to opening velocity can capture this change. Overall, **blink phase metrics** (peak speeds and phase durations) add nuance beyond total blink length, and have been used alongside basic blink rate to improve detection of early fatigue-induced eyelid behavior changes.

**Prolonged Blinks and Microsleep Events:** Drowsiness is often marked by occasional **prolonged eyelid closures**, essentially very long blinks or “microsleeps.” Metrics to capture these events include **PERCLOS** (percentage of time eyes are ≥80% closed) and the **count/rate of Long Eye Closures (LEC)** beyond a threshold (e.g. 500 ms). Such measures go beyond average blink duration by focusing on extreme events. Research shows these prolonged-blink measures correlate strongly with performance lapses: for example, the frequency of LEC events had one of the highest odds ratios for predicting lane departures in drowsy driving. In one study, having LEC rates in the top 5% corresponded to a **10–32× increase in risk of an out-of-lane incident** in a 5-minute window. Thus, features like “number of blinks > X ms” or longest blink duration in a period are powerful fatigue indicators. These can be treated as time-series features by updating counts over moving windows (e.g. per 30 s or 1 min). Microsleep events are essentially binary flags of severe drowsiness; when integrated into composite drowsiness indices, they significantly boost predictive ability (e.g. JDS incorporates the proportion of long blinks as one component). In real-time systems, detecting a single prolonged closure often triggers immediate alerts, since it likely indicates the onset of sleep. Overall, **microsleep detection metrics** (like LEC count or maximum eye-closure duration) complement continuous measures and provide a safety-critical signal of extreme fatigue.

**Complex Blink Waveform Metrics:** Recent work has extracted **detailed shape descriptors** from the eyelid aperture waveform, treating each blink’s time-series as a rich source of features. One notable approach is the *BLINKER* algorithm (Tarafder et al. 2022), which identified over 25 blink waveform features from EEG-based blink signals. These include: blink durations measured at various thresholds (e.g. **Duration Half-Maximum** – the width of the blink at half amplitude), and multiple **Amplitude–Velocity Ratios** computed over different blink segments (for instance, using only the closing half vs. opening half of the blink). They also define features like **Closing Time (CT)** and **Reopening Time (RT)** under different definitions – e.g. time from start to peak (“Zero” crossing points) vs. time within a fitted triangular “tent” model of the blink. By fitting each blink to an approximate triangular waveform (“tent”), one can measure slopes and intersection points, yielding features such as **CTT/RTT (Closing/Reopening Time Tent)** and corresponding amplitude measures. Tarafder et al. found that several of these novel features (e.g. **Negative Amplitude/Velocity Ratio (NAVR)** for the closing phase and **Closing Time Tent (CTT)**) were among the top predictors of drowsiness in their ensemble model. The practical implication is that subtle differences in blink *shape* – not just overall length – carry information about fatigue. For instance, a drowsy blink might have a slower deceleration (reflected in tent slope) or a plateau near full closure. By quantifying these with features like “Time Shut” (time spent ≥90% closed) or peak-to-baseline slope, systems can detect gradations of drowsiness that simpler metrics miss. Such waveform analysis requires high-fidelity eyelid signals (from EOG or infrared eye-trackers) and signal processing to find inflection points, but it enables **composite morphology metrics** that correlate strongly with fatigue levels. As these features become more common, they enrich blink-based drowsiness models with a deeper, time-series-informed understanding of eyelid motion.

**Blink Timing Variability and Patterns:** Another set of novel time-series features relates to the **variability and regularity of blinking over time**. Rather than just the mean blink rate or interval, researchers examine how blink intervals fluctuate. Fatigue can induce an **irregular blinking pattern** – for example, sequences of rapid blinks followed by long pauses as a driver’s eyes fight to stay open. To capture this, features like the **standard deviation or coefficient of variation of inter-blink intervals (IBI)** in a time window are used. A high variability or erratic blink rhythm may signal changing alertness. Some studies have gone further to apply non-linear dynamics metrics to blink sequences, such as **entropy measures** or **fractal dimension** of the blink-rate time-series. For instance, methods inspired by gaze entropy have been considered: Shiferaw et al. showed that increased unpredictability (higher entropy) in gaze fixations correlates with drowsiness, suggesting that a similar approach to blink timing could be informative. While direct application of entropy to blinks is less common, an analogous idea is **blink transition probability** – how likely the blink rate is to switch states (e.g., from normal to bursty). Hidden Markov Models (HMMs) have been proposed to model such state transitions in blinking behavior. By modeling blink sequences probabilistically, HMM-based features can include the likelihood of being in a “drowsy blink state” at a given time. In summary, **time-series pattern features** – whether simple statistical variability or more complex sequence modeling outputs – represent a novel way to quantify *how* blinking unfolds over time, rather than just averaging it. These features leverage the temporal structure of blink data, offering additional predictive power for fatigue beyond static blink counts. Early signs like an **“irregular blink pattern”** have indeed been noted as warning flags in fatigue detection frameworks, validating the importance of these time-series variability metrics.

## Composite and Multi-Modal Blink-Related Metrics

**Composite Blink Indices (Multi-Feature):** As mentioned, the **John’s Drowsiness Score (JDS)** is an example of a composite metric entirely derived from multiple blink features. It combines blink duration, velocity, and other eyelid metrics into a single continuous score. The novelty of such indices is in their **fusion of several blink parameters**, rather than using one in isolation. JDS, for instance, was developed alongside the Optalert™ system and reflects a weighted combination of blink rate, a series of amplitude/velocity ratios, and the presence of long closures. By scaling these into a 0–10 score, it provides a real-time fatigue level that correlates with performance impairments. In practice, composite blink scores smooth out noise in individual features and capture a broader physiological picture. Other composite blink-based metrics include ratios like **blink frequency to fixation duration**, or combined thresholds (e.g. “blink >500 ms AND PERCLOS >30%” as a severe drowsiness indicator). These metrics often serve as the final decision variable in driver monitoring systems. They highlight the trend in recent research to **blend multiple eyelid measures** to improve robustness. While the user’s prior catalog covers many single blink features, composite indices (especially those scaling multiple features into one fatigue score) represent a novel layer of analysis aimed at practical, **actionable drowsiness scoring**.

**Blink-EOG and EEG Fusion:** Blinks are inherently linked to other physiological signals, and modern approaches exploit this by fusing blink data with brain activity measures. A novel strategy is to extract **blink information directly from EEG signals** (which often contain blink artifacts) and use it for drowsiness detection. Tarafder et al. (2022) asked: *Can we detect drowsiness from ocular indices in an EEG without a separate eye tracker?*. Using their BLINKER tool, they identified blink events and features from frontal EEG leads, essentially performing EOG analysis on the EEG. This multi-modal twist allows blink behavior to be combined with EEG metrics like alpha/theta power. They found that incorporating **blink duration and other blink artifact features alongside EEG spectral features improved accuracy**. In fact, their explainable model showed interactions: e.g. when EEG theta power is high *and* blink duration is long, the probability of drowsiness is highest. Other studies have explicitly combined **EOG and EEG signals** for detection. For example, one technique used **EEG + EOG in a helmet-mounted system** and showed that blink duration measured from EOG and simultaneous brainwave changes together distinguished alert vs. drowsy states better than either alone. The **practical novelty** here is treating blinks as one channel in a larger physiological monitoring context – effectively a *multi-modal feature*. By fusing ocular and neural features, systems can catch both the behavioral signs (slow blinks) and the underlying central nervous changes (e.g. rising theta waves) of fatigue. This fusion has achieved higher accuracies; one multimodal study reported \~83.5% accuracy using combined EEG+EOG+ECG features, outperforming any single modality. In summary, **blink–EEG composite features** (like “blink rate during high-theta EEG”) provide a richer indicator of drowsiness than blink behavior alone, and represent a cutting-edge direction in fatigue research.

**Blink and Heart-Rate Variability (HRV):** Another composite approach pairs blink metrics with cardiovascular signals. Fatigue and reduced alertness often manifest in autonomic changes (like increased heart rate variability). **Blink-HRV composite features** leverage these connections. As one example, Bergasa et al. (2008) developed a **wearable system (a “helmet”) that monitored both blinking and HRV** to detect driver fatigue. In their results, a combination of **prolonged blink duration (>400 ms)** and certain HRV patterns reliably signaled sleepiness. Specifically, during drowsiness they observed **longer blinks alongside elevated low-frequency HRV components**, indicating sympathetic nervous system changes. By using a composite rule (e.g., “blink duration + RR-interval variance”), they achieved better prediction of drowsy episodes than either blinks or heart signals alone. The **novelty** lies in correlating an ocular behavior (blink) with a physiological state measure (heart rhythm). This kind of sensor fusion has practical implications: if a driver’s eye camera reports slow blinks at the same time a wearable heart monitor reports high variability or sudden drops in heart rate, the confidence in a drowsiness alarm is much higher. Some real-time driver monitoring frameworks are beginning to incorporate such multi-sensor data fusion to reduce false alarms. **HRV-blink composites** essentially treat blink events as triggers or modifiers in a larger context of driver state. While the user’s reference material focused on eyelid data alone, introducing heart-rate features (or other vitals like skin conductance) linked in time to blink behavior is a noteworthy extension in recent research.

**Blink Behavior with Other Facial Cues:** In video-based monitoring, blink features are often combined with **other observable behaviors** – forming composite visual indicators. For instance, **blinking + yawning frequency** is a common combination: these are treated as two facets of fatigue in tandem. A driver showing both an elevated blink rate and frequent yawns is far more likely to be drowsy than one showing only one symptom. Some systems compute a joint fatigue score from these (e.g. summing normalized blink rate and yawn count). Another composite is **blink metrics with eye-gaze measures**. Research by Jackson et al. and others introduced **gaze entropy** and **fixation stability** alongside blink rate. In practice, this might mean if blinks are slowing and the driver’s gaze is becoming more erratic (higher entropy), the system’s drowsiness index spikes. **Head pose nodding** is yet another cue – slight head bobs often co-occur with microsleeps. Systems have combined blink duration with detected head-nod events to improve detection latency (the head nod might precede or coincide with a long blink). An IEEE study by Bergasa et al. used a fuzzy logic controller on multiple cues (eye closures, **EAR** – Eye Aspect Ratio for blinks, plus mouth openness for yawns, etc.). The fusion of these “blink-related behaviors” produced a more robust drowsiness level output than any single cue alone. In summary, **multi-modal composite features involving blinks** span a range from other facial behaviors (yawn, gaze, head tilt) to even vehicle data (steering variability combined with blink rate). The trend is to incorporate blinks as one vital component of a holistic drowsiness detection metric. This multi-faceted approach is especially useful in real-time driving conditions, where relying on one signal can be error-prone due to noise (e.g. poor lighting affecting eye camera). By combining blink behavior with corroborating cues, systems achieve both higher accuracy and earlier detection of fatigue states.

## Time-Series Analytical Techniques and ML Pipelines

**Sequential Blink Pattern Modeling (HMMs and State Models):** Rather than extracting a fixed set of blink features over a window, one analytical technique is to model the **temporal sequence of blinks as a stochastic process**. Hidden Markov Models (HMMs) have been employed to represent different drowsiness states with distinct blink patterns. For example, Eyosiyas et al. (2014) proposed an HMM-based dynamic model that ingests facial observations (blink duration, eye closure state) over time to probabilistically infer the driver’s state. The HMM’s hidden states can correspond to “alert”, “slightly drowsy”, “very drowsy”, etc., with blink-related observations causing transitions. Such a model inherently captures the **time dependence** – e.g. a run of long blinks increases probability of transitioning to the drowsy state. In another approach, Martello et al. used a finite-state machine where the duration of eye closure determined state transitions (short closures keep system in normal state, an accumulation of longer closures pushes it into warning state). The novelty of these methods is treating blinking as a **time-evolving signal** to be modeled continuously, rather than resetting each epoch. They often use **sequential likelihoods** or **cumulative sums** as features – for instance, the likelihood that a given blink sequence would occur if the driver were drowsy vs. alert. One study defined a **“blink sequence score”**: it slides a window of T consecutive blinks and classifies that sequence as alert/drowsy using learned models. By voting across multiple sequences in a video, the system achieved more stable predictions. This voting on blink sequences is effectively an HMM-like smoothing. The practical impact is that **early subtle signs** can be caught; a single slow blink might not trigger an alarm, but an HMM can accumulate evidence over several blinks and change state once a pattern is recognized. As a result, **HMM and sequence models** improve both sensitivity and specificity, reducing false alarms that single-blink metrics might produce and catching gradual onset of fatigue that threshold-based methods might miss.

**Deep Learning on Eye-Blink Time Series:** With the rise of deep learning, modern pipelines increasingly handle blink data as a time-series input to neural networks. One novel pipeline uses **recurrent neural networks (RNNs)** or sequence-aware models to automatically learn blink features. For instance, Koushkestani et al. created a baseline *temporal* model that feeds a sliding window of consecutive blink events into an RNN to predict drowsiness probability. Their network looks at the pattern of inter-blink intervals and durations in the sequence, effectively learning complex features like “bursts of quick blinks followed by a slow blink.” This approach outperformed models that only used aggregate features, highlighting that the **network-extracted temporal features** carry additional information. Similarly, some works integrate CNNs with RNNs: e.g. a CNN may first process each video frame or short eye image sequence to detect blinks and eye aspect ratio, then an LSTM (a type of RNN) processes the timeline of those features to classify drowsiness. This two-stage deep learning captures spatial blink features and their temporal evolution. An example is a system by Park et al., which (after frame-level CNN processing) classified fatigue with an LSTM over time – they emphasized that incorporating the **temporal context of blinks** is crucial, since classifying each frame independently ignores subtle blink-duration changes. Indeed, approaches that neglect temporal pooling require very overt signs (like eyes fully closed for many frames) to detect drowsiness. By contrast, sequence models can detect a slow drift in eye closure over consecutive blinks. Recently, Transformer-based models (attention networks) have also been applied to blink sequences. Hassan et al. (2025) used a Vision Transformer to classify eye state (open/closed) frame by frame, then implemented a **real-time drowsiness scoring** that essentially counts the length of eye-closure sequences to trigger an alarm. The Transformer’s strength lies in capturing long-range dependencies – e.g. it could learn that a blink lasting >50 frames is essentially “prolonged closure.” These deep learning pipelines are novel in that they **learn composite blink features implicitly** (through network weights) rather than requiring hand-crafted features. They have achieved impressive accuracy (e.g. >95% for eye state classification, and substantial improvements in drowsiness detection rates). The use of deep models also facilitates **real-time implementation** on embedded devices as some networks can be optimized for speed. In summary, advanced ML pipelines treat blink behavior as a temporal signal to be learned, enabling discovery of complex blink patterns (e.g. a certain blink cadence that precedes a lapse) that might not be obvious or included in traditional feature sets.

**Real-Time Drowsiness Detection Systems:** Bringing these features and models into real-world use has led to innovations in real-time processing. For example, Ji et al. employed an **Extended Kalman Filter (EKF)** to continuously track eye landmarks and smooth the eye-closure signal, enabling robust measurement of blink frequency and duration in real time. This kind of filtering is an analytical technique to handle noisy time-series (video frame-by-frame) and still extract blink metrics accurately on the fly. Other systems use adaptive thresholds that update based on recent blink statistics – effectively a short-term time-series adaptation to account for individual differences or lighting changes. Many commercial or experimental driver-monitoring systems now run on 30 Hz video or wearable sensors and compute blink features in a rolling window (e.g. every second, compute the last 30 seconds of features). The **machine learning pipeline then operates in an online fashion**, sometimes with incremental model updates or memory of past state. An example pipeline might be: detect blinks per frame, update a buffer of recent blink intervals, compute features (blink rate, PERCLOS, etc.) for the last minute, then apply a classifier (say an SVM or a small neural network) to decide the drowsiness level. If the system uses a sequence model (LSTM), it may directly ingest the last N blinks without needing to recompute handcrafted features each time. The novelty in some recent real-time systems is also the incorporation of **explainability and personalization**. The 2024 study by Watling et al. used SHAP values in real time to identify which feature (e.g. blink duration spike) triggered an alert. This helps with user acceptance (explaining “you are tired because your blinks slowed by X”). Other pipelines allow a calibration phase to learn a driver’s baseline blink pattern and then detect deviations over time. This personalized time-series modeling can reduce false positives, as each driver’s normal blink rate variability is taken into account. In short, real-time drowsiness detection has evolved to incorporate **sophisticated time-series processing** – from Kalman filters for smoother blink signals to on-the-fly sequence classification and even hybrid systems fusing multiple sensor streams in real time. These advancements ensure that the novel blink features and models discussed (AVR, microsleep counts, RNN sequence models, etc.) can be deployed in practical settings, delivering timely warnings with high confidence.

## Conclusion

In conclusion, recent academic research has greatly expanded the toolkit for correlating blink behavior with fatigue. Novel time-series blink features – such as amplitude–velocity ratios, detailed eyelid movement dynamics, prolonged-blink event counts, and blink-pattern variability metrics – go well beyond traditional blink rate and duration measures. These features capture the subtle slowing and irregularity of blinking that accompanies drowsiness, and they have demonstrated strong predictive power in experiments. Moreover, composite metrics and multi-modal approaches integrate blink data with other physiological and behavioral signals (EEG, HRV, yawning, etc.), yielding more robust indicators than any single cue alone. Finally, advanced analytical techniques – from HMM state modeling to deep learning sequence networks – now leverage the full temporal information in blink time-series, enabling real-time drowsiness detection systems that are both accurate and responsive. The **novelty** of these methods lies in treating eyelid behavior not as isolated blinks, but as a rich, time-dependent signal reflective of the brain’s dwindling alertness. Practically, implementing these insights means drivers and operators can be monitored using a combination of eyelid metrics that can detect fatigue earlier and more reliably, ultimately improving safety. The research surveyed here underscores that blinks, when analyzed with modern time-series techniques, are indeed a window into the state of the weary brain – providing a sophisticated basis for next-generation drowsiness detection systems.

**References:** (Academic and peer-reviewed sources for the above content have been cited in-text in the format【source†lines】 to ensure traceability of data and claims.)
