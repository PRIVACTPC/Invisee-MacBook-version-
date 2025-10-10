import pandas as pd
from scipy.stats import wasserstein_distance
import tkinter as tk
from tkinter import filedialog

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
df = pd.read_csv(file_paths


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