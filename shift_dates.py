import os, datetime
import pandas as pd

# Project root directory
project_root = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.join(project_root, 'data')

# Determine the target start date: January 1 of current year
now = datetime.datetime.now()
start_of_year = datetime.datetime(now.year, 1, 1)

# List of CSV files to adjust
csv_files = [
    'trust_scores.csv',
    'evidence_reports.csv',
    'feature_vectors.csv',
    'drift_results.csv',
    'policy_results.csv',
    'detection_metrics.json',  # this is json, will be ignored
]

for fname in csv_files:
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        print(f'Skipping missing file: {fname}')
        continue
    if fname.endswith('.json'):
        # Not handling JSON files for now
        continue
    df = pd.read_csv(path)
    # Identify the timestamp column (common names)
    time_col = None
    for col in ['window', 'timestamp', 'window_start']:
        if col in df.columns:
            time_col = col
            break
    if not time_col:
        print(f'No time column found in {fname}')
        continue
    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    # Compute offset to map earliest timestamp to start_of_year
    earliest = df[time_col].min()
    offset = start_of_year - earliest
    df[time_col] = df[time_col] + offset
    # Write back
    df.to_csv(path, index=False)
    print(f'Updated {fname}')
