This markdown describes the functions used for calculating various blink properties. These calculations are based on the `BlinkProperties` class in `pyblinkers/extractBlinkProperties.py`.

## Blink Property Calculations

### Durations

The following durations are calculated in the `set_blink_duration` method:

-   **`durationBase`**: The time difference between the right and left base points of the blink.
    ```
    (rightBase - leftBase) / srate
    ```
-   **`durationZero`**: The time difference between the right and left zero-crossing points of the blink.
    ```
    (rightZero - leftZero) / srate
    ```
-   **`durationTent`**: The time difference between the right and left x-intercepts of the lines fitted to the blink slopes.
    ```
    (rightXIntercept - leftXIntercept) / srate
    ```
-   **`durationHalfBase`**: The time difference between the points on the rising and falling slopes of the blink at half the amplitude, relative to the base.
    ```
    (rightBaseHalfHeight - leftBaseHalfHeight + 1) / srate
    ```
-   **`durationHalfZero`**: The time difference between the points on the rising and falling slopes of the blink at half the amplitude, relative to the zero-crossings.
    ```
    (rightZeroHalfHeight - leftZeroHalfHeight + 1) / srate
    ```

### Amplitude-Velocity Ratios

These ratios are calculated to describe the relationship between the blink's amplitude and its velocity.

-   **`posAmpVelRatioZero` / `negAmpVelRatioZero`**: The ratio of the maximum blink amplitude to the maximum (positive) and minimum (negative) velocity between the zero-crossings. The positive ratio is calculated from `leftZero` to `maxFrame`, and the negative ratio from `maxFrame` to `rightZero`.
    ```python
    # Simplified concept
    100 * abs(maxValue / extreme_velocity) / srate
    ```
-   **`posAmpVelRatioBase` / `negAmpVelRatioBase`**: The ratio of the maximum blink amplitude to the maximum (positive) and minimum (negative) velocity between the base points. The positive ratio is calculated from `leftBase` to `maxFrame`, and the negative ratio from `maxFrame` to `rightBase`.
-   **`posAmpVelRatioTent` / `negAmpVelRatioTent`**: The ratio of the maximum blink amplitude to the average velocity of the left and right slopes of the "tent" fit.
    ```python
    # For the positive (left) side
    100 * abs(maxValue / averLeftVelocity) / srate
    # For the negative (right) side
    100 * abs(maxValue / averRightVelocity) / srate
    ```

### Shut Times

These properties measure the duration the eye is considered "shut" during a blink.

-   **`closingTimeZero` / `reopeningTimeZero`**: The time it takes from the left zero-crossing to the blink peak (`closing`) and from the peak to the right zero-crossing (`reopening`).
    ```python
    closingTimeZero = (maxFrame - leftZero) / srate
    reopeningTimeZero = (rightZero - maxFrame) / srate
    ```
-   **`closingTimeTent` / `reopeningTimeTent`**: The time it takes from the left x-intercept to the intersection point of the fitted lines (`closing`) and from the intersection to the right x-intercept (`reopening`).
    ```python
    closingTimeTent = (xIntersect - leftXIntercept) / srate
    reopeningTimeTent = (rightXIntercept - xIntersect) / srate
    ```
-   **`timeShutBase`**: The duration for which the blink signal is above a certain fraction (`shutAmpFraction`) of its maximum amplitude, measured between the base points.
-   **`timeShutZero`**: The duration for which the blink signal is above a certain fraction (`shutAmpFraction`) of its maximum amplitude, measured between the zero-crossing points.
-   **`timeShutTent`**: The duration for which the blink signal is above a certain fraction (`shutAmpFraction`) of its maximum amplitude, measured between the x-intercepts of the fitted lines.

### Peak Timings and Amplitudes

These properties relate to the peak of the blink.

-   **`peakMaxBlink`**: The maximum amplitude of the blink signal (`maxValue`).
-   **`peakMaxTent`**: The amplitude at the intersection of the fitted lines (`yIntersect`).
-   **`peakTimeBlink`**: The time at which the blink reaches its maximum amplitude.
    ```
    maxFrame / srate
    ```
-   **`peakTimeTent`**: The time of the intersection of the fitted lines.
    ```
    xIntersect / srate
    ```

### Inter-Blink Intervals

These properties measure the time between consecutive blinks.

-   **`interBlinkMaxAmp`**: The time difference between the peaks of consecutive blinks.
-   **`interBlinkMaxVelBase`**: The time difference between the points of maximum positive velocity (relative to the base) of consecutive blinks.
-   **`interBlinkMaxVelZero`**: The time difference between the points of maximum positive velocity (relative to the zero-crossings) of consecutive blinks.

### Other Properties

-   **`peaksPosVelBase`**: The sample index of the maximum positive velocity between `leftBase` and `maxFrame`.
-   **`peaksPosVelZero`**: The sample index of the maximum positive velocity between `leftZero` and `maxFrame`.