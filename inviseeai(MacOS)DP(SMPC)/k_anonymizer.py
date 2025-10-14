"""
k_anonymizer.py

Helpers for a numeric/datetime-based k-anonymity workflow using global recoding.

* `k_anonymity` – metric for minimum equivalence class size.
* `kanonymize` – transforms a DataRepository to satisfy k-anonymity.

Non-numeric columns remain unchanged; row count is preserved.
"""

from typing import Sequence
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

from data_repository import DataRepository
from anonypy import Preserver


def k_anonymity(frame: pd.DataFrame, quasi_cols: Sequence[str]) -> int:
    """Return the minimum equivalence-class size for *quasi_cols*."""
    if not quasi_cols:
        raise ValueError("quasi_cols must not be empty")
    return frame.groupby(list(quasi_cols), dropna=False).size().min()


def _microaggregate(df: pd.DataFrame, quasi_cols: Sequence[str], k: int) -> pd.DataFrame:
    data = df[quasi_cols].copy()

    for col in quasi_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].round(0).astype("Int64")
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = pd.to_datetime(data[col]).dt.floor('D')

    global_min_max = {}
    global_date_min_max = {}
    for col in quasi_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            s_vals = data[col].dropna()
            if not s_vals.empty:
                global_min_max[col] = (int(s_vals.min()), int(s_vals.max()))
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            s_vals = pd.to_datetime(data[col].dropna())
            if not s_vals.empty:
                global_date_min_max[col] = (s_vals.min(), s_vals.max())

    scores = pd.Series(0.0, index=df.index)
    for col in quasi_cols:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            z = (series - series.mean()) / (series.std(ddof=0) or 1)
            scores += z
        elif pd.api.types.is_datetime64_any_dtype(series):
            ts = pd.to_datetime(series).view("int64")
            z = (ts - ts.mean()) / (ts.std(ddof=0) or 1)
            scores += z

    ordered = scores.sort_values().index.to_list()
    n = len(ordered)
    groups = [ordered[i:i+k] for i in range(0, n, k)]
    if len(groups) > 1 and len(groups[-1]) < k:
        groups[-2].extend(groups[-1])
        groups.pop()

    df_out = df.copy()
    for col in quasi_cols:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = df_out[col].astype("object")

    for grp in groups:
        sub = data.loc[grp, quasi_cols]
        agg_vals = {}
        for col in quasi_cols:
            s = sub[col]
            if pd.api.types.is_numeric_dtype(s):
                s_vals = s.dropna()
                mn, mx = (int(s_vals.min()), int(s_vals.max())) if not s_vals.empty else ("", "")
                rep = f"{mn}" if mn == mx else f"{mn}–{mx}" if mn != "" and mx != "" else ""
            elif pd.api.types.is_datetime64_any_dtype(s):
                s_vals = pd.to_datetime(s.dropna())
                mn, mx = (s_vals.min(), s_vals.max()) if not s_vals.empty else ("", "")
                mn_str = mn.strftime('%Y-%m-%d') if mn else ""
                mx_str = mx.strftime('%Y-%m-%d') if mx else ""
                rep = mn_str if mn_str == mx_str else f"{mn_str} – {mx_str}" if mn_str and mx_str else ""
            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                rep = s.mode(dropna=True).iat[0] if not s.dropna().empty else ""
            agg_vals[col] = rep
        for col, val in agg_vals.items():
            df_out.loc[grp, col] = val

    return df_out


def kanonymize(repo: DataRepository, quasi_cols: Sequence[str], k: int, strategy: str = "micro") -> DataRepository:
    if strategy == "micro":
        return _kanonymize_with_micro(repo, quasi_cols, k)
    elif strategy == "anonypy":
        return _kanonymize_with_anonypy(repo, quasi_cols, k)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'micro' or 'anonypy'.")


def _kanonymize_with_micro(repo: DataRepository, quasi_cols: Sequence[str], k: int) -> DataRepository:
    if k <= 1:
        return repo

    df = repo.frame.copy()
    df_anon = _microaggregate(df, quasi_cols, k)

    repo._frame = df_anon
    setattr(repo, "_k_anonymized", True)
    setattr(repo, "_k", k)
    setattr(repo, "_k_quasi", list(quasi_cols))
    return repo


def _kanonymize_with_anonypy(repo: DataRepository, quasi_cols: Sequence[str], k: int) -> DataRepository:
    df = repo.frame.copy()
    for col in quasi_cols:
        if not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    mapping = {val: idx for idx, val in enumerate(sorted(df[col].dropna().unique()))}
                    df[col] = df[col].map(mapping)

    pres = Preserver(df, quasi_cols, sensitive_column=None, by="records")
    rows = pres.anonymize_k_anonymity(k=k)
    df_anon = pd.DataFrame(rows).reset_index(drop=True)

    repo._frame = df_anon
    setattr(repo, "_k_anonymized", True)
    setattr(repo, "_k", k)
    setattr(repo, "_k_quasi", list(quasi_cols))
    return repo


def deanonymize(repo: DataRepository) -> DataRepository:
    """
    Restore original quasi-identifier values from backup columns created during k-anonymization.
    """
    if not hasattr(repo, "_k_backup_cols"):
        raise ValueError("No k-anonymization backup columns found for de-anonymization.")

    backup_cols = repo._k_backup_cols
    df = repo.frame.copy()

    for col, bkp_col in backup_cols.items():
        if bkp_col in df.columns:
            df[col] = df[bkp_col]
            df.drop(columns=[bkp_col], inplace=True)

    clone = repo.clone()
    clone._frame = df
    if hasattr(clone, "_k_backup_cols"):
        delattr(clone, "_k_backup_cols")
    return clone
