import os
from getpass import getpass
from pathlib import Path

import pandas as pd
from scipy.stats import wasserstein_distance
from differential_privacy import mask_string_columns, apply_differential_privacy
import tkinter as tk
from tkinter import filedialog


CORRECT_PASSCODE = os.environ.get("VERIFICATOR_PASSCODE", "secure123")
LOCK_EPSILON = 0.01  # Small epsilon => large noise, locks down data.


def _apply_lockdown_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Mask string columns and inject heavy differential privacy noise."""
    masked_df = mask_string_columns(df)
    numeric_cols = masked_df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        return masked_df

    epsilons = {col: LOCK_EPSILON for col in numeric_cols}
    return apply_differential_privacy(masked_df, epsilons)


def _enforce_passcode(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Prompt for passcode and apply DP noise on failure."""
    passcode = getpass("Enter verificator passcode: ")
    if passcode == CORRECT_PASSCODE:
        print("Passcode accepted. Proceeding with verification...")
        return df

    print("Invalid passcode. Applying differential privacy noise to lock the dataset.")
    locked_df = _apply_lockdown_noise(df)

    target_path = Path(file_path).with_name(Path(file_path).stem + "_locked.csv")
    try:
        locked_df.to_csv(target_path, index=False)
        print(f"Noised copy written to '{target_path}'. Original data remains unchanged.")
    except Exception as exc:
        print(f"Could not persist locked dataset ({exc}). Continuing with in-memory noise only.")

    print("Dataset has been noised and is now unusable for meaningful analysis.")
    return locked_df

# File chooser dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an anonymized CSV dataset",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load dataset
df = pd.read_csv(file_path)
df = _enforce_passcode(df, file_path)


# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Show available columns
print("\nAvailable columns in dataset:")
print(df.columns.tolist())

# Get quasi-identifiers from user
qids_input = input("\nEnter quasi-identifier columns (comma-separated): ")
qids = [col.strip().lower() for col in qids_input.split(",") if col.strip().lower() in df.columns]
if not qids:
    print("No valid quasi-identifier columns selected. Exiting.")
    exit()

# Get sensitive attribute from user
sensitive_col = input("Enter sensitive attribute column: ").strip().lower()
if sensitive_col not in df.columns:
    print("Invalid sensitive attribute. Exiting.")
    exit()

# -------- K-Anonymity --------
def check_k_anonymity(df, qids, k):
    groups = df.groupby(qids)
    return groups.size().min() >= k

# -------- L-Diversity --------
def check_l_diversity(df, qids, sensitive_col, l):
    diversity = df.groupby(qids)[sensitive_col].nunique()
    return (diversity >= l).all()

# -------- T-Closeness --------
def check_t_closeness(df, qids, sensitive_col, t):
    overall_dist = df[sensitive_col].value_counts(normalize=True)
    for _, group in df.groupby(qids):
        group_dist = group[sensitive_col].value_counts(normalize=True)
        group_dist = group_dist.reindex(overall_dist.index, fill_value=0)
        dist = wasserstein_distance(overall_dist.values, group_dist.values)
        if dist > t:
            return False
    return True

# -------- Run Checks --------
k = 4
l = 3
t = 0.2

print("k-Anonymity:", check_k_anonymity(df, qids, k))
print("l-Diversity:", check_l_diversity(df, qids, sensitive_col, l))
print("t-Closeness:", check_t_closeness(df, qids, sensitive_col, t))
