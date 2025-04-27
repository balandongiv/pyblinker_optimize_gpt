import warnings
import numpy as np
import pandas as pd


def _to_ints(*args):
    """
    Convert multiple values to native Python int.

    This is used to ensure all indexing-related variables are type-safe.
    When using `pandas.apply`, mixed dtypes (like float32 and int32) can result in float64 values
    being passed to functions, which may cause issues with NumPy indexing.

    Accepts any number of inputs and returns their `int()` equivalents in order.
    """
    return tuple(int(x) for x in args)

def get_line_intersection_slope(x_intersect, y_intersect, left_x_intersect, right_x_intersect):
    """
    Original logic retained. Computes slopes at the intersection point.
    """
    # Local variable usage here is minimal since there's only two lines.
    left_slope = y_intersect / (x_intersect - left_x_intersect)
    right_slope = y_intersect / (x_intersect - right_x_intersect)
    return left_slope, right_slope


def get_average_velocity(p_left, p_right, x_left, x_right):
    """
    Original logic retained. Computes average velocities.
    """
    # Using local references is possible, but it's already short.
    aver_left_velocity = p_left.coef[1] / np.std(x_left)
    aver_right_velocity = p_right.coef[1] / np.std(x_right)
    return aver_left_velocity, aver_right_velocity



def compute_outer_bounds(df: pd.DataFrame, data_size: int) -> pd.DataFrame:
    """
    Computes 'outerStarts' and 'outerEnds' for each row in a DataFrame based on the 'maxFrame' column.

    Each blink potential range is defined as the interval:
        [outerStarts, outerEnds)

    Where:
    - 'outerStarts' is the maxFrame of the **previous** row (or 0 if first row)
    - 'outerEnds' is the maxFrame of the **next** row (or data_size if last row)

    This creates a window around each blink's maxFrame to define its surrounding range.

    ASCII Example:
        Input:
            index | maxFrame
            ------|---------
              0   |   10
              1   |   25
              2   |   40
              3   |   60

        Output after compute_outer_bounds(df, data_size=80):
            index | maxFrame | outerStarts | outerEnds
            ------|----------|-------------|-----------
              0   |   10     |     0       |    25
              1   |   25     |    10       |    40
              2   |   40     |    25       |    60
              3   |   60     |    40       |    80

        Visualized as:
            Row 0: [ 0  ---> 25 ]
            Row 1: [10  ---> 40 ]
            Row 2: [25  ---> 60 ]
            Row 3: [40  ---> 80 ]

    Args:
        df (pd.DataFrame): A DataFrame that must contain a 'maxFrame' column.
        data_size (int): The total length of the candidate signal (used for the final 'outerEnd').

    Returns:
        pd.DataFrame: A new DataFrame with added 'outerStarts' and 'outerEnds' columns.
    """
    df = df.copy()
    df['outerStarts'] = df['maxFrame'].shift(1, fill_value=0)
    df['outerEnds'] = df['maxFrame'].shift(-1, fill_value=data_size)
    return df



def left_right_zero_crossing(candidate_signal, max_frame, outer_starts, outer_ends):
    """
    Identify the left zero crossing and right zero crossing of the signal
    between outer_starts->max_frame and max_frame->outer_ends.
    Signal timeline (index):

    |----------------|----------------|----------------|
    0          outer_starts      max_frame       outer_ends

    candidate_signal = [...- +, +, +, 0, +, +, +, +, +, -,...]
                            ^               ^
                    left_zero        right_zero

    The goal:
    - Find the first negative value BEFORE `max_frame` (from `outer_starts` to `max_frame`) → left_zero
    - Find the first negative value AFTER `max_frame` (from `max_frame` to `outer_ends`) → right_zero

    If not found:
    - For left_zero, we extend the search beyond the outer_starts, in this case,we assume the start_idx (i.e.,previous max frame of previous blink is imperfect), so we will find the first negative, but is this a good approach?
    - For right_zero, fallback to searching [max_frame, end_of_signal]

    - If no negative values are found, set left_zero or right_zero to np.nan. This especially true when
    we deal with epoch format, as the signal windows might be small, therefore,we cannot extend the search window
    to extreme.
    """

    start_idx,end_idx,m_frame=_to_ints(outer_starts,outer_ends,max_frame)

    # Left side search
    left_range = np.arange(start_idx, m_frame)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)

    if s_ind_left_zero.size > 0:
        left_zero = left_range[s_ind_left_zero[-1]]
    else:
        # There is instance where there is no negative crossing found within the stipulated range,
        # In this case, in left_range
        full_left_range = np.arange(0, m_frame).astype(int)
        left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
        if left_neg_idx.size > 0:
            left_zero = full_left_range[left_neg_idx[-1]]
        else:
            # No negative values found, set a default fallback (e.g., 0 or np.nan)
            left_zero = np.nan # or np.nan, depending on your use case

    # Right side search
    right_range = np.arange(m_frame, end_idx)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)

    if s_ind_right_zero.size > 0:
        right_zero = right_range[s_ind_right_zero[0]]
    else:
        # Extreme remedy by extending beyond outer_ends to the max signal length

        extreme_outer = np.arange(m_frame, candidate_signal.shape[0]).astype(int)
        s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
        if s_ind_right_zero_ex.size > 0:
            right_zero = extreme_outer[s_ind_right_zero_ex[0]]
        else:
            right_zero=np.nan



    return left_zero, right_zero


def get_up_down_stroke(max_frame, left_zero, right_zero):
    """
    Compute the place of maximum positive and negative velocities.
    up_stroke is the interval between left_zero and max_frame,
    down_stroke is the interval between max_frame and right_zero.
    """
    m_frame, l_zero, r_zero = _to_ints(max_frame, left_zero, right_zero)
    up_stroke = np.arange(l_zero, m_frame + 1)
    down_stroke = np.arange(m_frame, r_zero + 1)
    return up_stroke, down_stroke


def _max_pos_vel_frame(blink_velocity, max_frame, left_zero, right_zero):
    """
    In the context of *blink_velocity* time series,
    the `max_pos_vel_frame` and `max_neg_vel_frame` represent the indices where
    the *blink_velocity* reaches its maximum positive value and maximum negative value, respectively.
    """

    m_frame, l_zero, r_zero = _to_ints(max_frame, left_zero, right_zero)
    up_stroke, down_stroke = get_up_down_stroke(m_frame, l_zero, r_zero)

    # Maximum positive velocity in the up_stroke region
    max_pos_vel_idx = np.argmax(blink_velocity[up_stroke])
    max_pos_vel_frame = up_stroke[max_pos_vel_idx]


    # Maximum negative velocity in the down_stroke region, if it exists
    if down_stroke.size > 0:
        # Case: down_stroke contains only one index and it equals the last index of blink_velocity
        if down_stroke.size == 1 and down_stroke[0] == len(blink_velocity) - 1:
            max_neg_vel_frame = np.nan
        else:
            # Remove the last index if it's in down_stroke
            down_stroke = down_stroke[down_stroke != len(blink_velocity)]

            if down_stroke.size > 0:
                max_neg_vel_idx = np.argmin(blink_velocity[down_stroke])
                max_neg_vel_frame = down_stroke[max_neg_vel_idx]
            else:
                warnings.warn('Force nan but require further investigation why it happened like this')
                max_neg_vel_frame = np.nan
    else:
        warnings.warn('Force nan but require further investigation why it happened like this')
        max_neg_vel_frame = np.nan

    return max_pos_vel_frame, max_neg_vel_frame



def _get_left_base(blink_velocity, left_outer, max_pos_vel_frame):
    """
    Determine the left base index from left_outer to max_pos_vel_frame
    by searching for where blink_velocity crosses <= 0.
    """

    l_outer, m_pos_vel = _to_ints(left_outer, max_pos_vel_frame)
    left_range = np.arange(l_outer, m_pos_vel + 1)
    reversed_velocity = np.flip(blink_velocity[left_range])

    left_base_index = int(np.argmax(reversed_velocity <= 0))
    left_base = m_pos_vel - left_base_index - 1
    return left_base


def _get_right_base(candidate_signal, blink_velocity, right_outer, max_neg_vel_frame):
    """
    Determine the right base index from max_neg_vel_frame to right_outer
    by searching for where blink_velocity crosses >= 0.
    """

    r_outer, m_neg_vel = _to_ints(right_outer, max_neg_vel_frame)

    # Ensure boundaries are valid
    if m_neg_vel > r_outer:
        return None

    max_size = candidate_signal.size
    end_idx = min(r_outer, max_size)
    right_range = np.arange(m_neg_vel, end_idx)

    if right_range.size == 0:
        return None

    # Avoid out-of-bounds indexing for blink_velocity
    if right_range[-1] >= blink_velocity.size:
        right_range = right_range[:-1]
        if right_range.size == 0 or right_range[-1] >= blink_velocity.size:
            # TODO: Handle this case more gracefully
            raise ValueError('Please strategies how to address this')

    right_base_velocity = blink_velocity[right_range]
    right_base_index = int(np.argmax(right_base_velocity >= 0))
    right_base = m_neg_vel + right_base_index + 1
    return right_base



def _get_half_height(candidate_signal, max_frame, left_zero, right_zero, left_base, right_outer):
    """
    left_base_half_height:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the left base value.
    right_base_half_height:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the right base value.
    """

    m_frame, l_zero, r_zero, l_base, r_outer = _to_ints(
        max_frame, left_zero, right_zero, left_base, right_outer
    )
    # Halfway point (vertical) from candidate_signal[max_frame] to candidate_signal[left_base]
    max_val = candidate_signal[m_frame]
    left_base_val = candidate_signal[l_base]
    half_height_val = max_val - 0.5 * (max_val - left_base_val)

    # Left side half-height from base
    left_range = np.arange(l_base, m_frame + 1)
    left_vals = candidate_signal[left_range]
    left_index = int(np.argmax(left_vals >= half_height_val))
    left_base_half_height = l_base + left_index + 1

    # Right side half-height from base
    right_range = np.arange(m_frame, r_outer + 1)
    try:
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame
        )
    except IndexError:
        # If out-of-bounds, reduce range by 1
        right_range = np.arange(m_frame, r_outer)
        right_base_half_height = min(
            r_outer,
            np.argmax(candidate_signal[right_range] <= half_height_val) + m_frame
        )

    # Now compute the left and right half-height frames from zero
    # Halfway from candidate_signal[max_frame] down to 0 (the "zero" crossing region).
    # left_zero_half_height
    zero_half_val = 0.5 * max_val
    left_zero_range = np.arange(l_zero, m_frame + 1)
    left_zero_index = int(np.argmax(candidate_signal[left_zero_range] >= zero_half_val))
    left_zero_half_height = l_zero + left_zero_index + 1

    # right_zero_half_height
    right_zero_range = np.arange(m_frame, r_zero + 1)
    right_zero_index = int(np.argmax(candidate_signal[right_zero_range] <= zero_half_val))
    right_zero_half_height = min(r_outer, m_frame + right_zero_index)

    return left_zero_half_height, right_zero_half_height, left_base_half_height, right_base_half_height


def get_left_range(left_zero, max_frame, candidate_signal, blink_top, blink_bottom):
    """
    Identify the left blink range based on blink_top/blink_bottom thresholds
    within candidate_signal.
    """

    l_zero,m_frame = _to_ints(left_zero,max_frame)
    blink_range = np.arange(l_zero, m_frame + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    # Indices where candidate_signal < blink_top
    top_idx = np.where(cand_slice < blink_top)[0]
    blink_top_point_idx = top_idx[-1]  # the last occurrence

    # Indices where candidate_signal > blink_bottom
    bottom_idx = np.flatnonzero(cand_slice > blink_bottom)
    blink_bottom_point_idx = bottom_idx[0]  # the first occurrence

    blink_top_point_l_x = blink_range[blink_top_point_idx]
    blink_top_point_l_y = candidate_signal[blink_top_point_l_x]

    blink_bottom_point_l_x = blink_range[blink_bottom_point_idx]
    blink_bottom_point_l_y = candidate_signal[blink_bottom_point_l_x]

    left_range = [blink_bottom_point_l_x, blink_top_point_l_x]

    return left_range, blink_top_point_l_x, blink_top_point_l_y, blink_bottom_point_l_x, blink_bottom_point_l_y


def get_right_range(max_frame, right_zero, candidate_signal, blink_top, blink_bottom):
    """
    Identify the right blink range based on blink_top/blink_bottom thresholds
    within candidate_signal.
    """

    m_frame,r_zero= _to_ints(max_frame, right_zero)
    blink_range = np.arange(m_frame, r_zero + 1, dtype=int)
    cand_slice = candidate_signal[blink_range]

    # Indices where candidate_signal < blink_top
    top_mask = (cand_slice < blink_top)
    blink_top_point_r = np.argmax(top_mask)  # first True

    # Indices where candidate_signal > blink_bottom
    bottom_mask = (cand_slice > blink_bottom)
    bottom_idx = np.where(bottom_mask)[0]
    blink_bottom_point_r = bottom_idx[-1]  # last True

    blink_top_point_r_x = blink_range[blink_top_point_r]
    blink_top_point_r_y = candidate_signal[blink_top_point_r_x]

    blink_bottom_point_r_x = blink_range[blink_bottom_point_r]
    blink_bottom_point_r_y = candidate_signal[blink_bottom_point_r_x]

    right_range = [blink_range[blink_top_point_r], blink_range[blink_bottom_point_r]]

    return (right_range,
            blink_top_point_r_x, blink_top_point_r_y,
            blink_bottom_point_r_x, blink_bottom_point_r_y)


def compute_fit_range(candidate_signal, max_frame, left_zero, right_zero, base_fraction, top_bottom=None):
    """
    Computes x_left, x_right, left_range, right_range,
    plus optional top/bottom blink points,
    for the candidate_signal around a blink event.
    """

    m_frame,l_zero,r_zero= _to_ints(max_frame, left_zero, right_zero)
    # Compute the blink_top/blink_bottom for thresholding
    blink_height = candidate_signal[m_frame] - candidate_signal[l_zero]
    blink_top = candidate_signal[m_frame] - base_fraction * blink_height
    blink_bottom = candidate_signal[l_zero] + base_fraction * blink_height

    (left_range,
     blink_top_point_l_x, blink_top_point_l_y,
     blink_bottom_point_l_x, blink_bottom_point_l_y) = get_left_range(l_zero, m_frame, candidate_signal, blink_top, blink_bottom)

    (right_range,
     blink_top_point_r_x, blink_top_point_r_y,
     blink_bottom_point_r_x, blink_bottom_point_r_y) = get_right_range(m_frame, r_zero, candidate_signal, blink_top, blink_bottom)

    # Create arrays for fitting
    x_left = np.arange(left_range[0], left_range[1] + 1, dtype=int)  # +1 to include the last index
    x_right = np.arange(right_range[0], right_range[1] + 1, dtype=int)

    # Replace empty arrays with np.nan for consistency
    if x_left.size == 0:
        x_left = np.nan
    if x_right.size == 0:
        x_right = np.nan

    if top_bottom is None:
        # Return minimal information
        warnings.warn('To modify this so that all function return the top_bottom point')
        return x_left, x_right, left_range, right_range
    else:
        # Return extended info including top/bottom points
        return (x_left, x_right, left_range, right_range,
                blink_bottom_point_l_y, blink_bottom_point_l_x,
                blink_top_point_l_y, blink_top_point_l_x,
                blink_bottom_point_r_x, blink_bottom_point_r_y,
                blink_top_point_r_x, blink_top_point_r_y)