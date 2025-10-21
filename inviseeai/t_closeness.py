import pandas as pd

try:
    from data_repository import DataRepository
except Exception:  # pragma: no cover - optional dependency
    DataRepository = None


def compute_distribution(series, *, dropna: bool = True, exclude_mask: str | None = "***"):
    """
    Return a normalized value distribution for a column.
    - dropna: if True, remove NaN/None before computing the distribution
    - exclude_mask: if set (e.g., "***"), exclude that placeholder from the distribution
    """
    s = series
    if dropna:
        s = s.dropna()
    if exclude_mask is not None:
        s = s[s != exclude_mask]
    if s.empty:
        # Return an empty Series of float to keep downstream alignment simple
        return pd.Series(dtype=float)
    return s.value_counts(normalize=True).sort_index()


def total_variation_distance(p, q):
    """Compute the total variation distance (TVD) between two distributions.
    Expects aligned numeric arrays/iterables of equal length.
    """
    import numpy as _np
    p = _np.asarray(p, dtype=float)
    q = _np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("TVD inputs must have the same shape after alignment.")
    return 0.5 * _np.abs(p - q).sum()


def t_closeness_check(df, quasi_cols, sensitive_col, t):
    """
    Return a list of violating index groups (one per violating equivalence class).
    Groups with TVD > t are considered privacy violations.
    """
    global_dist = compute_distribution(df[sensitive_col], dropna=True, exclude_mask="***")
    violations = []

    # If there is no meaningful global distribution, there can be no violations
    if global_dist.empty:
        return violations

    for name, group in df.groupby(quasi_cols):
        group_dist = compute_distribution(group[sensitive_col], dropna=True, exclude_mask="***")

        # Align indices so that all categories match
        aligned_index = global_dist.index.union(group_dist.index)
        global_aligned = global_dist.reindex(aligned_index, fill_value=0.0)
        group_aligned = group_dist.reindex(aligned_index, fill_value=0.0)

        tvd = total_variation_distance(global_aligned.values, group_aligned.values)

        print(f"[DEBUG] Group {name} --> TVD = {tvd:.4f}")

        # Standard logic: mask if TVD > t
        if tvd > t:
            violations.append(group.index)

    return violations


def _ensure_mutable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame that can be mutated in-place."""
    if getattr(df.attrs, "get", None) and df.attrs.get("_frozen"):
        return df.copy()
    try:
        arr = df.to_numpy(copy=False)
        if hasattr(arr, "flags") and not arr.flags.writeable:
            return df.copy()
    except Exception:
        pass
    return df


def apply_t_closeness(data, quasi_cols, sensitive_col, t, *, source_df=None, inplace: bool = False):
    """
    Apply t-closeness masking by replacing sensitive values with "***"
    in any group that violates the t-closeness threshold.
    Accepts either a pandas DataFrame or a DataRepository. When a DataRepository is
    provided, the masked frame is written to a clone by default (unless inplace=True).
    If *source_df* is supplied it will be used as the starting frame—useful for chaining
    after ℓ-diversity masking without reverting to the pre-ℓ state.
    """
    repo = None
    if DataRepository is not None and isinstance(data, DataRepository):
        repo = data
        base_df = source_df if source_df is not None else repo.frame
    else:
        base_df = source_df if source_df is not None else data
        if not isinstance(base_df, pd.DataFrame):
            raise TypeError("apply_t_closeness expects a pandas DataFrame or DataRepository as input.")

    working_df = _ensure_mutable_frame(base_df)
    violating_groups = t_closeness_check(working_df, quasi_cols, sensitive_col, t)

    for group_index in violating_groups:
        working_df.loc[list(group_index), sensitive_col] = "***"

    if repo is not None:
        target_repo = repo if inplace else repo.clone()
        target_repo._frame = working_df
        setattr(target_repo, "_t", t)
        setattr(target_repo, "_t_quasi", list(quasi_cols))
        setattr(target_repo, "_t_sensitive", sensitive_col)
        return target_repo

    return working_df
