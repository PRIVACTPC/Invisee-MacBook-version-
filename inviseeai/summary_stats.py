from typing import Any
import pandas as pd
import numpy as np

from data_repository import DataRepository
from k_anonymizer import k_anonymity

def simple_summary(repo: DataRepository | pd.DataFrame) -> dict[str, Any]:
    if isinstance(repo, pd.DataFrame):
        df = repo
    else:
        df = repo.frame

    summary = {}

    quasi_cols = list(getattr(repo, "_k_quasi", []))
    k_applied = getattr(repo, "_k_anonymized", False) or getattr(repo, "_k", None) is not None
    l_applied = getattr(repo, "_l", None) is not None

    summary["_dataset"] = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "k-anonymized": k_applied,
        "l-diversity": l_applied,
        "scattered_columns": [],
        "k-value": None,
        "l-value": getattr(repo, "_l", None),
        "l-diverse": getattr(repo, "_l_diverse", None),
    }

    if k_applied and quasi_cols:
        if getattr(repo, "_k", None) is not None:
            summary["_dataset"]["k-value"] = getattr(repo, "_k")
        else:
            try:
                summary["_dataset"]["k-value"] = k_anonymity(df, quasi_cols)
            except Exception:
                summary["_dataset"]["k-value"] = "Error"

    if hasattr(repo, "_original_frame"):
        orig_df = getattr(repo, "_original_frame")
        for col in df.columns:
            if col in orig_df.columns and not df[col].equals(orig_df[col]):
                summary["_dataset"]["scattered_columns"].append(col)

    for col in df.columns:
        s = df[col]
        col_summary = {
            "dtype": str(s.dtype),
            "missing_values": int(s.isna().sum()),
            "non_nulls": int(s.notna().sum()),
            "unique_values": int(s.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(s):
            col_summary.update({
                "mean": round(s.mean(), 2),
                "std": round(s.std(), 2),
            })

        vc = s.value_counts(dropna=True).head(10)
        if not vc.empty:
            col_summary["value_counts"] = vc.to_dict()

        summary[col] = col_summary

    return summary


def format_summary_table(summary: dict[str, Any]) -> str:
    """
    Format a summary dictionary into a readable multi-line string table.
    """
    from textwrap import indent

    lines = []

    # Display overall dataset info
    dataset_info = summary.pop("_dataset", {})
    lines.append("=== Dataset Info ===")
    for key, val in dataset_info.items():
        lines.append(f"{key:<20}: {val}")
    lines.append("")

    # Display per-column stats
    for col, stats in summary.items():
        lines.append(f"--- Column: {col} ---")
        for key, val in stats.items():
            lines.append(f"  {key:<15}: {val}")
        lines.append("")

    return "\n".join(lines)


# (Optional) This should ideally be moved to l_diversity.py
def vectorized_l_contextual_mask(df: pd.DataFrame, columns: list[str], l: int) -> pd.DataFrame:
    """
    Apply vectorized left-to-right ℓ-diversity masking across specified columns.
    """
    df = df.copy()
    for col in columns:
        values = df[col].astype(str)
        max_len = values.str.len().max()
        padded = values.str.pad(width=max_len, side='right', fillchar=' ')
        arr = padded.apply(list).to_numpy()
        mat = np.stack(arr)

        prev = np.roll(mat, shift=1, axis=0)
        next_ = np.roll(mat, shift=-1, axis=0)

        mask = np.full_like(mat, False, dtype=bool)
        for offset in range(l):
            match_prev = (mat == prev) & (mat != "*")
            match_next = (mat == next_) & (mat != "*")
            mask |= match_prev | match_next

        mask &= (mat != ' ') & (mat != '*')
        mat[mask] = '*'

        df[col] = ["".join(row).rstrip() for row in mat]
    return df
