def filter_blink_amplitude_ratios(df, params):
    """
    Filter rows based on blink amplitude ratio range.
    If no rows remain, fallback to the row with the highest number of good blinks.
    """
    min_ratio = params['blinkAmpRange_1']
    max_ratio = params['blinkAmpRange_2']

    filtered_df = df[
        (df['blinkAmpRatio'] >= min_ratio) &
        (df['blinkAmpRatio'] <= max_ratio)
        ]

    if not filtered_df.empty:
        filtered_df = filtered_df.copy()
        filtered_df.loc[:, 'select'] = True
        return filtered_df

    # Fallback: select the best available candidate
    max_idx = df['numberGoodBlinks'].idxmax()
    df['select'] = False  # Ensure the column exists
    df.loc[max_idx, ['select', 'status']] = [
        True,
        "Blink amplitude too low — selected row with highest numberGoodBlinks."
    ]

    return df




def filter_good_blinks(df, params):
    """
    Find the ones that meet the minimum good blink threshold.
    Filter rows based on number of good blinks.
    If no rows remain, set status and select the row with max numberGoodBlinks.
    """
    threshold=params['minGoodBlinks']
    # Filter DataFrame based on minimum good blinks
    filtered_df = df[df['numberGoodBlinks'] > params['minGoodBlinks']]

    if not filtered_df.empty:
        filtered_df.loc[:, 'select'] = True
        return filtered_df

    # Fallback: no rows meet threshold, select the best available
    max_idx = df['numberGoodBlinks'].idxmax()
    df['select'] = False  # Ensure column exists
    df.loc[max_idx, ['select', 'status']] = [True, f"Fewer than {threshold} minimum Good Blinks were found"]

    return df




def filter_good_ratio(df, params):
    """
    Filter rows based on good ratio threshold.
    If none meet the threshold, select the row with the highest number of good blinks.
    """
    threshold = params['goodRatioThreshold']
    filtered_df = df[df['goodRatio'] >= threshold]

    if not filtered_df.empty:
        # filtered_df['select'] = True
        filtered_df.loc[:, 'select'] = True
        return filtered_df

    # Fallback: no rows pass the threshold
    max_idx = df['numberGoodBlinks'].idxmax()
    df['select'] = False  # Ensure column exists
    df.loc[max_idx, ['select', 'status']] = [
        True,
        "Good ratio too low — selecting row with max number of good blinks."
    ]

    return df




def select_max_good_blinks(df):
    """
    Ensure that the row with the maximum number of good blinks is selected
    if no row is currently marked as selected.
    """
    if 'select' in df.columns and df['select'].any():
        df['select'] = True
        return df  # Selection already exists

    # No selection yet — fallback to the row with max number of good blinks
    max_idx = df['numberGoodBlinks'].idxmax()
    df['select'] = False  # Ensure the column exists
    df.loc[max_idx, ['select', 'status']] = [True, "Complete all checking"]

    return df




def channel_selection(channel_blink_stats, params):
    channel_blink_stats=filter_blink_amplitude_ratios(channel_blink_stats, params)

    # Return early if exactly one True in 'select' column
    if channel_blink_stats['select'].sum() == 1:
        return channel_blink_stats


    channel_blink_stats = filter_good_blinks(channel_blink_stats, params)
    # Return early if exactly one True in 'select' column
    if channel_blink_stats['select'].sum() == 1:
        return channel_blink_stats

    channel_blink_stats = filter_good_ratio(channel_blink_stats, params)

    # Return early if exactly one True in 'select' column
    if channel_blink_stats['select'].sum() == 1:
        return channel_blink_stats

    signal_data_output = select_max_good_blinks(channel_blink_stats)



    return signal_data_output

