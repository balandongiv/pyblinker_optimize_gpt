import pickle

from pyblinkers.extractBlinkProperties import BlinkProperties, get_good_blink_mask
from pyblinkers.extractBlinkProperties import get_blink_statistic
from pyblinkers.fit_blink import FitBlinks

with open("fitblinks_debug.pkl", "rb") as f:
    debug_data = pickle.load(f)

# Access variables
candidate_signal = debug_data["candidate_signal"]
df = debug_data["df"]
params = debug_data["params"]

# Store initial length
prev_len = len(df)
print(f"Initial df length: {prev_len}")

# STEP 2: Fit blinks
fitblinks = FitBlinks(
    candidate_signal=candidate_signal,
    df=df,
    params=params)
fitblinks.process_blink_candidate()
df = fitblinks.frame_blinks

# Check df length after STEP 2
if len(df) < prev_len:
    print(f"STEP 2: df length decreased from {prev_len} to {len(df)}")
elif len(df) == prev_len:
    print(f"STEP 2: df length maintained at {len(df)}")
else:
    print(f"STEP 2: df length increased from {prev_len} to {len(df)}")
prev_len = len(df)

# STEP 3: Extract blink statistics
blink_stats = get_blink_statistic(
    df, params['z_thresholds'],
    signal=candidate_signal
)
blink_stats['ch'] = 'ch'
# df is not changed here, so we skip checking length

# STEP 4: Get good blink mask _ori
# good_blink_mask, df = get_good_blink_mask(
#     df,
#     blink_stats['bestMedian'],
#     blink_stats['bestRobustStd'],
#     params['z_thresholds']
# )
#
# # Check df length after STEP 4
# if len(df) < prev_len:
#     print(f"STEP 4: df length decreased from {prev_len} to {len(df)}")
# elif len(df) == prev_len:
#     print(f"STEP 4: df length maintained at {len(df)}")
# else:
#     print(f"STEP 4: df length increased from {prev_len} to {len(df)}")
# prev_len = len(df)

# STEP 5: Compute blink properties
df = BlinkProperties(
    candidate_signal,
    df,
    params['sfreq'],
    params
).df

# Check df length after STEP 5
if len(df) < prev_len:
    print(f"STEP 5: df length decreased from {prev_len} to {len(df)}")
elif len(df) == prev_len:
    print(f"STEP 5: df length maintained at {len(df)}")
else:
    print(f"STEP 5: df length increased from {prev_len} to {len(df)}")

h = 1
