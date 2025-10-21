#!/usr/bin/env python
"""
Run inside each Docker container:
1.  Read CSV with numeric/date features only.
2.  Fit a trivial linear regression on (features -> target).
3.  Save JSON with intercept and coefficients.

Usage:
    python compute_node.py /data/input.csv /data/output.json
"""
import sys
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import numpy as np
import os

# Adjust import path to find data_handler.py in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from data_handler import sanitize

if len(sys.argv) != 3:
    print("Usage: python compute_node.py /data/input.csv /data/output.json")
    sys.exit(1)

in_csv, out_json = sys.argv[1:]

df = pd.read_csv(in_csv)
df = sanitize(df)

# last column is the target (convention from deployer)
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

 # Build a pipeline: standardise each feature then fit a ridge regression.
model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
    ]
).fit(X, y)

# Retrieve scaler and ridge objects
scaler = model.named_steps["scaler"]
ridge = model.named_steps["ridge"]

# De‑scale coefficients back to original feature units
coef = ridge.coef_ / scaler.scale_
intercept = ridge.intercept_ - np.dot(coef, scaler.mean_)

payload = {
    "intercept": float(intercept),
    "coefficients": {col: float(c) for col, c in zip(X.columns, coef)},
}

with open(out_json, "w") as f:
    json.dump(payload, f)

print("local model written")
