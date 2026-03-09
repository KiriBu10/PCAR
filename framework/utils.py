from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import pandas as pd


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)

def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

def test_mae_difference(df1: pd.DataFrame, df2: pd.DataFrame, paired: bool = True):
    """
    Test if the MAE values between two dataframes are significantly different.

    Args:
        df1 (pd.DataFrame): First dataframe with 'MAE' column.
        df2 (pd.DataFrame): Second dataframe with 'MAE' column.
        paired (bool): Whether the samples are paired (default True).

    Returns:
        statistic (float): t-statistic
        p_value (float): p-value
    """
    mae1 = df1['test_mae']
    mae2 = df2['test_mae']

    if paired:
        statistic, p_value = ttest_rel(mae1, mae2)
    else:
        statistic, p_value = ttest_ind(mae1, mae2)

    return statistic, p_value


def find_parallel_sequences(df_a: pd.DataFrame, df_b: pd.DataFrame, considered_timeframe: float,
                            activity_to_int: dict, process_to_int: dict) -> pd.DataFrame:
    df_a = df_a.copy()
    parallel_sequences_list = []

    # Pre-compute case_end_times from df_b for each (process, case_id)
    case_end_times = df_b.groupby(['process', 'case_id'])['timestamp'].max().to_dict()

    for _, row in df_a.iterrows():
        prefix_time = row['prefix_time']
        start_time = prefix_time - considered_timeframe
        considered_case_id = row['case_id'] 

        # Filter B for events within timeframe and before prefix_time
        df_b_filtered = df_b[(df_b['timestamp'] >= start_time) &
                             (df_b['timestamp'] <= prefix_time)&
                             (df_b['case_id'] != considered_case_id)]

        # Group by process and case_id to build parallel sequences
        grouped = df_b_filtered.groupby(['process', 'case_id'])

        parallel_seqs = []
        for (process, case_id), group in grouped:
            group_sorted = group.sort_values('timestamp')
            activity_sequence = group_sorted['activity'].tolist()
            timestamps = group_sorted['timestamp'].tolist()
            first_timestamp = timestamps[0]
            time_deltas = [0.0] + [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

            # Fetch the true case_end_time for this process + case_id
            case_end_time = case_end_times.get((process, case_id), None)

            if case_end_time is not None:
                remaining_case_time = case_end_time - prefix_time
            else:
                remaining_case_time = None

            parallel_seqs.append({
                'process_ints': [process_to_int[process]] * len(activity_sequence),
                'case_id': case_id,
                'activity_sequence': activity_sequence,
                'activity_sequence_int': [activity_to_int[i] for i in activity_sequence],
                'timestamps': timestamps,
                'timestamps_case_related':[t-first_timestamp for t in timestamps],
                'time_deltas': time_deltas,
                'case_end_time': case_end_time,
                'remaining_case_time': remaining_case_time
            })

        parallel_sequences_list.append(parallel_seqs)

    df_a['parallel_sequences'] = parallel_sequences_list
    return df_a


# drop-in replacement for attach_parallel_sequences (same signature/return)
import heapq, bisect
from typing import List, Dict, Any

def attach_parallel_sequences_fast(prefix_df: pd.DataFrame,
                                   case_meta: dict,
                                   intervals,
                                   max_parallel=10) -> pd.DataFrame:
    case_ids_arr, starts, ends = intervals  # (np.ndarray, np.ndarray, np.ndarray)
    rows_out: List[List[Dict[str, Any]]] = []

    # local helpers for speed
    times_abs_of = {cid: cm["times_abs"] for cid, cm in case_meta.items()}
    times_rel_of = {cid: cm["times_rel"] for cid, cm in case_meta.items()}
    acts_int_of  = {cid: cm["acts_int"]  for cid, cm in case_meta.items()}
    proc_int_of  = {cid: cm["proc_int"]  for cid, cm in case_meta.items()}
    proc_str_of  = {cid: cm["process"]   for cid, cm in case_meta.items()}

    for r in prefix_df.itertuples(index=False):
        # expects columns case_id, prefix_abs_time
        cid = getattr(r, "case_id")
        t_abs = float(getattr(r, "prefix_abs_time"))

        # concurrent cases (mask in O(n) over intervals; arrays are contiguous & fast)
        m = (starts <= t_abs) & (ends > t_abs) & (case_ids_arr != cid)
        ocases = case_ids_arr[m]

        if not len(ocases):
            rows_out.append([])
            continue

        # --- pass 1: keep only top-K most progressed at t_abs ---
        heap = []  # min-heap of (score, cid, cut_idx)
        push = heapq.heappush; pop = heapq.heappop
        K = max_parallel

        for oc in ocases:
            ta = times_abs_of[oc]
            # rightmost event <= t_abs
            cut = bisect.bisect_right(ta, t_abs) - 1
            if cut < 0:
                continue
            score = float(times_rel_of[oc][cut])  # “progress”
            if len(heap) < K:
                push(heap, (score, oc, cut))
            else:
                if score > heap[0][0]:
                    pop(heap); push(heap, (score, oc, cut))

        if not heap:
            rows_out.append([])
            continue

        # --- pass 2: materialize only the winners ---
        winners = sorted(heap, key=lambda x: -x[0])  # optional, stable order
        par_list = []
        for _, oc, cut in winners:
            acts = acts_int_of[oc][:cut+1].tolist()
            t_rel = times_rel_of[oc][:cut+1].astype(float).tolist()
            if not acts:
                continue
            par_list.append({
                "activity_sequence_int": acts,
                "timestamps_case_related": t_rel,
                "process_int": int(proc_int_of[oc]),
                "process": proc_str_of[oc],
            })
        rows_out.append(par_list)

    out = prefix_df.copy()
    out["parallel_sequences"] = rows_out
    return out



############################# PREPROCESSING ######################################
def filter_complete_cases(event_log: pd.DataFrame) -> pd.DataFrame:
    # Identify case_ids where at least one row has status == 'complete'
    complete_case_ids = event_log[event_log['status'] == 'COMPLETE']['case_id'].unique()

    # Filter the original log to keep only those case_ids
    complete_cases = event_log[event_log['case_id'].isin(complete_case_ids)]

    return complete_cases

import pandas as pd

def count_unique_traces(df):
    """
    Count the number of unique activity sequences (traces) in an event log.

    Parameters:
    df (pd.DataFrame): Event log with at least 'case_id', 'timestamp', and 'activity' columns.

    Returns:
    int: Number of unique traces.
    """
    # Ensure correct sorting within each case
    df_sorted = df.sort_values(by=['case_id', 'timestamp'])

    # Group by case_id and aggregate activity sequence as a tuple
    trace_series = df_sorted.groupby('case_id')['activity'].apply(tuple)

    # Find number of unique traces
    unique_traces = trace_series.nunique()

    return unique_traces


def build_activity_vocab(df):
    activities = set()
    for seq in df['activity'].unique():
        activities.add(seq)
    activity_to_int = {act: idx + 1 for idx, act in enumerate(sorted(activities))}
    int_to_activity = {v: k for k, v in activity_to_int.items()}
    return activity_to_int, int_to_activity

def build_process_vocab(df):
    processes = set()
    for seq in df['process'].unique():
        processes.add(seq)
    process_to_int = {act: idx + 1 for idx, act in enumerate(sorted(processes))}
    int_to_process = {v: k for k, v in process_to_int.items()}
    return process_to_int, int_to_process

def encode_prefix_data(df, activity_to_int, process_to_int):
    df['prefix_int'] = df['prefix'].apply(lambda seq: [activity_to_int[a] for a in seq])
    df['next_activity_int'] = df['next_activity'].apply(lambda a: activity_to_int[a])
    df['process_int'] = df['process'].apply(lambda a: process_to_int[a])
    return df


def get_event_log_statistics(event_log: pd.DataFrame) -> dict:
    # Group by case
    grouped = event_log.groupby('case_id')

    # Number of cases
    num_cases = grouped.ngroups

    # Unique activities
    num_unique_activities = event_log['activity'].nunique()

    # Case lengths
    case_lengths = grouped.size()
    min_case_length = case_lengths.min()
    max_case_length = case_lengths.max()
    avg_case_length = case_lengths.mean()

    # Case durations (max timestamp - min timestamp per case)
    case_durations = grouped['timestamp'].agg(lambda x: x.max() - x.min())
    min_case_duration = case_durations.min()
    max_case_duration = case_durations.max()
    avg_case_duration = case_durations.mean()

    return {
        "Number of Cases": num_cases,
        "Number of Unique Activities": num_unique_activities,
        "Min Case Length": int(min_case_length),
        "Max Case Length": int(max_case_length),
        "Avg Case Length": round(avg_case_length, 2),
        "Min Case Duration": round(min_case_duration, 2),
        "Max Case Duration": round(max_case_duration, 2),
        "Avg Case Duration": round(avg_case_duration, 2),
    }

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Tuple
def preprocess_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['case_id', 'timestamp'])
    return df

def split_cases(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_cases = df['case_id'].unique()
    train_cases, test_cases = train_test_split(unique_cases, test_size=1-train_ratio, random_state=42)
    val_cases, test_cases = train_test_split(test_cases, test_size=(1 - train_ratio - val_ratio) / (1 - train_ratio), random_state=42)
    
    train_df = df[df['case_id'].isin(train_cases)]
    val_df = df[df['case_id'].isin(val_cases)]
    test_df = df[df['case_id'].isin(test_cases)]
    
    return train_df, val_df, test_df


from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
def split_cases_temporal(df: pd.DataFrame, train_ratio=0.5, val_ratio=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Step 1: Compute the start timestamp for each case
    case_start_times = df.groupby('case_id')['timestamp'].min().reset_index()
    case_start_times = case_start_times.sort_values(by='timestamp')  # Step 2: Sort by start time

    # Step 3: Determine split indices
    num_cases = len(case_start_times)
    num_train = int(num_cases * train_ratio)
    num_val = int(num_cases * val_ratio)

    # Get case_ids for each split
    train_case_ids = case_start_times.iloc[:num_train]['case_id']
    val_case_ids = case_start_times.iloc[num_train:num_train + num_val]['case_id']
    test_case_ids = case_start_times.iloc[num_train + num_val:]['case_id']

    # Step 4: Filter the original df by case_ids
    train_df = df[df['case_id'].isin(train_case_ids)]
    val_df = df[df['case_id'].isin(val_case_ids)]
    test_df = df[df['case_id'].isin(test_case_ids)]

    return train_df, val_df, test_df
# from typing import Tuple
# import pandas as pd

# def get_ordered_activity_sequence(df: pd.DataFrame) -> str:
#     return '>'.join(df.sort_values('timestamp')['activity'].tolist())

# def split_cases_temporal(df: pd.DataFrame, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2, gen_ratio=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     assert abs(train_ratio + val_ratio + test_ratio + gen_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

#     # Step 1: Compute the start timestamp for each case
#     case_start_times = df.groupby('case_id')['timestamp'].min().reset_index()
#     case_start_times = case_start_times.sort_values(by='timestamp')  # Step 2: Sort by start time

#     # Step 3: Determine split indices
#     num_cases = len(case_start_times)
#     num_train = int(num_cases * train_ratio)
#     num_val = int(num_cases * val_ratio)
#     num_test = int(num_cases * test_ratio)

#     # Step 4: Get case_ids for each split (gen will be filled later)
#     train_case_ids = case_start_times.iloc[:num_train]['case_id']
#     val_case_ids = case_start_times.iloc[num_train:num_train + num_val]['case_id']
#     test_case_ids = case_start_times.iloc[num_train + num_val:num_train + num_val + num_test]['case_id']
#     remaining_case_ids = case_start_times.iloc[num_train + num_val + num_test:]['case_id']

#     # Step 5: Create intermediate datasets
#     train_df = df[df['case_id'].isin(train_case_ids)]
#     val_df = df[df['case_id'].isin(val_case_ids)]
#     test_df = df[df['case_id'].isin(test_case_ids)]
#     remaining_df = df[df['case_id'].isin(remaining_case_ids)]

#     # Step 6: Build set of seen sequences from train/val/test
#     seen_sequences = set()
#     for group in df[df['case_id'].isin(pd.concat([train_case_ids, val_case_ids, test_case_ids]))].groupby('case_id'):
#         sequence = get_ordered_activity_sequence(group[1])
#         seen_sequences.add(sequence)

#     # Step 7: Add only new unique sequences to test_gen_df
#     test_gen_case_ids = []
#     for case_id, group in remaining_df.groupby('case_id'):
#         sequence = get_ordered_activity_sequence(group)
#         if sequence not in seen_sequences:
#             test_gen_case_ids.append(case_id)
#             seen_sequences.add(sequence)  # prevent duplicates in test_gen_df

#     test_gen_df = remaining_df[remaining_df['case_id'].isin(test_gen_case_ids)]

#     return train_df, val_df, test_df, test_gen_df



import pandas as pd
import random

def generate_prefix_data(df: pd.DataFrame, sample_percent: float = 1.0) -> pd.DataFrame:
    prefix_data = []

    for case_id, group in df.groupby('case_id'):
        group = group.sort_values('timestamp')
        activities = group['activity'].tolist()
        timestamps = group['timestamp'].tolist()
        process_names = group['process'].tolist()

        
        case_end_time = timestamps[-1]

        prefixes = []

        for i in range(1, len(activities)):
            prefix = activities[:i]
            next_activity = activities[i]
            process_name = process_names[i]
            prefix_time = timestamps[i - 1]
            remaining_time = case_end_time - prefix_time
            list_timestamps = timestamps[:i]
            timestamp_case = prefix_time - timestamps[0]
            list_timestamps_case = [t-timestamps[0] for t in list_timestamps]
            time_deltas = [0.0] + [t2 - t1 for t1, t2 in zip(list_timestamps[:-1], list_timestamps[1:])]

            prefixes.append({
                'process': process_name,
                'case_id': case_id,
                'prefix': prefix,
                'timestamps': list_timestamps,
                'time_deltas': time_deltas,
                'next_activity': next_activity,
                'remaining_time': remaining_time,
                'prefix_time': prefix_time,
                'case_end_time': case_end_time,
                'timestamp_case':timestamp_case,
                'list_timestamps_case':list_timestamps_case
            })

        # Sample x% of prefixes for the case
        sample_size = max(1, int(len(prefixes) * sample_percent))
        prefix_data.extend(random.sample(prefixes, sample_size))

    return pd.DataFrame(prefix_data)


# def generate_prefix_data(df: pd.DataFrame) -> pd.DataFrame:
#     prefix_data = []

#     for case_id, group in df.groupby('case_id'):
#         group = group.sort_values('timestamp')
#         activities = group['activity'].tolist()
#         timestamps = group['timestamp'].tolist()
#         process_names = group['process'].tolist()

#         # Calculate time deltas (first delta is 0 or None)
#         time_deltas = [0.0] + [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
#         # Timestamp of last event in the case
#         case_end_time = timestamps[-1]

#         for i in range(1, len(activities)):
#             prefix = activities[:i]
#             next_activity = activities[i]
#             process_name = process_names[i]
#             prefix_time = timestamps[i - 1]  
#             remaining_time = case_end_time - prefix_time 
#             list_timestamps = timestamps[:i]

#             prefix_data.append({
#                 'process':process_name,
#                 'case_id': case_id,
#                 'prefix': prefix,
#                 'timestamps':list_timestamps,
#                 'time_deltas':time_deltas,
#                 'next_activity': next_activity,
#                 'remaining_time': remaining_time,
#                 'prefix_time':prefix_time,
#                 'case_end_time':case_end_time,

#             })

#     return pd.DataFrame(prefix_data)


def prepare_datasets(df: pd.DataFrame, temporal=False, sample_percent=1):
    df = preprocess_log(df)
    if temporal: 
        train_df, val_df, test_df = split_cases_temporal(df)
    else:
        train_df, val_df, test_df = split_cases(df)

    train_prefix = generate_prefix_data(train_df, sample_percent)
    val_prefix = generate_prefix_data(val_df, sample_percent)
    test_prefix = generate_prefix_data(test_df, sample_percent)
    #test_gen_prefix = generate_prefix_data(test_df, sample_percent=1)
    
    return train_prefix, val_prefix, test_prefix#, test_gen_prefix

def get_unseen_prefix_subset(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # Convert lists to tuples for set operations
    train_prefixes = set(tuple(p) for p in train_df['prefix'])
    val_prefixes = set(tuple(p) for p in val_df['prefix'])
    
    seen_prefixes = train_prefixes.union(val_prefixes)

    # Filter test set for unseen prefixes
    unseen_test_df = test_df[~test_df['prefix'].apply(lambda p: tuple(p) in seen_prefixes)].copy()

    return unseen_test_df




def numeric_case_id_series(series: pd.Series) -> pd.Series:
    """
    Keep only digits from each case_id. If a value has no digits,
    assign a stable surrogate integer based on that original string.
    Returns a string Series of digits (so you can cast to int if you like).
    """
    s = series.astype(str)

    # keep only digits
    digits = s.str.replace(r"\D", "", regex=True)

    # rows with no digits -> assign stable surrogate ids
    no_digits_mask = digits.eq("")
    if no_digits_mask.any():
        # factorize returns stable integer codes
        codes, uniques = pd.factorize(s[no_digits_mask], sort=True)
        # offset codes to avoid colliding with real numeric ids
        # choose a large prefix like "9" * 6 to avoid overlap
        surrogate = pd.Series(("900000" + (codes + 1).astype(str)),
                              index=digits[no_digits_mask].index)
        digits.loc[no_digits_mask] = surrogate

    return digits