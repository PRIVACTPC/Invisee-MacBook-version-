import math
import pandas as pd
from typing import Sequence
try:
    from data_repository import DataRepository
except Exception:
    DataRepository = None  # optional, for in-place repo finalization


def mask_similar_from_end(base: str, neighbor: str, l: int) -> str:
    """
    Mask any runs of 1 to ℓ identical characters (at the same positions from the right)
    between base and neighbor. Already masked characters ('*') are preserved.
    """
    base, neighbor = str(base), str(neighbor)
    masked = list(base)
    run = 0
    for i in reversed(range(min(len(base), len(neighbor)))):
        if base[i] == neighbor[i] and base[i] != "*":
            run += 1
        else:
            if 0 < run <= l:
                for j in range(i + 1, i + 1 + run):
                    if masked[j] != "*":
                        masked[j] = "*"
            run = 0
    if 0 < run <= l:
        for j in range(len(base) - run, len(base)):
            if masked[j] != "*":
                masked[j] = "*"
    return "".join(masked)



# --- Helper: right-to-left charwise masking for contextual masking ---
def _mask_rtl_charwise(base: str, neighbor: str, l: int) -> str:
    """Mask up to ℓ rightmost characters in *base* where *neighbor* matches at the same right-aligned position.
    - Ignores non-alphanumeric characters and existing '*'.
    - Does not require contiguous runs; evaluates each of the last ℓ positions independently.
    """
    base = str(base)
    neighbor = str(neighbor)
    limit = min(l, len(base), len(neighbor))
    if limit <= 0:
        return base
    chars = list(base)
    for offset in range(1, limit + 1):
        c = base[-offset]
        n = neighbor[-offset]
        # skip positions we already masked here or that are non-alphanumeric in base
        if c == '*' or not c.isalnum():
            continue
        # If neighbor is already masked at this position, propagate the mask (monotonicity)
        if n == '*':
            chars[-offset] = '*'
            continue
        # Otherwise, mask when characters match case-insensitively
        if c.casefold() == n.casefold():
            chars[-offset] = '*'
    return ''.join(chars)

def contextual_mask_column(series: pd.Series, l: int) -> pd.Series:
    """
    Apply ℓ-diverse contextual masking on a single column using adjacent rows.
    For each row, compare with previous and next values; at each of the last ℓ positions,
    if the character matches (and is alphanumeric), mask that character with '*'.
    """
    vals = series.astype(str).tolist()
    out = []
    for i, v in enumerate(vals):
        masked = v
        if i > 0:
            masked = _mask_rtl_charwise(masked, vals[i - 1], l)
        if i < len(vals) - 1:
            masked = _mask_rtl_charwise(masked, vals[i + 1], l)
        out.append(masked)
    return pd.Series(out, index=series.index)


def l_diversity_mask(
    df: pd.DataFrame,
    sensitive_col: str,
    quasi_cols: list,
    l: int,
    sensitivity_level: str = "medium",
    sentinel: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Perform ℓ-diversity contextual masking:
    - Mask sensitive_col with fixed intensity.
    - Apply ℓ-based positional masking to quasi_cols (column-wise, bidirectional).
    This function now modifies the provided DataFrame in-place (columns are reassigned) and returns the same object.

    Notes:
    - Accepts additional keyword arguments (e.g., 'sentinel') for backward compatibility; they are ignored.
    - Parameter `sentinel` is accepted for backward compatibility and ignored.
    """
    level_map = {"light": 25, "medium": 50, "heavy": 75, "sensitive": 100}
    if sensitivity_level not in level_map:
        raise ValueError("Invalid sensitivity level. Use light, medium, heavy, or sensitive.")

    def mask_percentage(s: str, pct: int) -> str:
        s = str(s)
        if pct >= 100:
            return "*" * len(s)
        to_mask = max(1, round(len(s) * pct / 100))
        return s[:len(s) - to_mask] + "*" * to_mask

    # No copy: operate directly on the provided DataFrame.

    # Mask sensitive column with fixed percentage
    df[sensitive_col] = df[sensitive_col].apply(lambda x: mask_percentage(x, level_map[sensitivity_level]))

    # Apply column-wise positional masking to quasi-identifiers
    if quasi_cols:
        right_to_left_global_l_mask(df, quasi_cols, l)

    freeze_dataframe_inplace(df)
    return df


def vectorized_l_contextual_mask(df: pd.DataFrame, columns: list[str], l: int) -> pd.DataFrame:
    """
    Mask up to the last ℓ characters of each value in `columns` when they match the previous or next row at the same
    (right-aligned) positions, using the same deterministic rule as `contextual_mask_column`.
    Modifies the provided DataFrame in-place and returns it.
    """
    # No copy: operate directly on the provided DataFrame.
    for col in columns:
        series = df[col].astype(str)
        vals = series.tolist()
        out = []
        for i, v in enumerate(vals):
            masked = v
            if i > 0:
                masked = _mask_rtl_charwise(masked, vals[i - 1], l)
            if i < len(vals) - 1:
                masked = _mask_rtl_charwise(masked, vals[i + 1], l)
            out.append(masked)
        df[col] = pd.Series(out, index=series.index)
    freeze_dataframe_inplace(df)
    return df


def consistent_vectorized_l_contextual_mask(df: pd.DataFrame, columns: list[str], l: int) -> pd.DataFrame:
    """
    Apply contextual ℓ-diversity masking consistently: if a value is masked once, it is masked the same way everywhere.
    Modifies the provided DataFrame in-place and returns it.
    """
    # No copy: operate directly on the provided DataFrame.
    for col in columns:
        series = df[col].astype(str)
        vals = series.tolist()
        # First pass: compute context-aware masked outputs per row
        row_masked = []
        for i, v in enumerate(vals):
            masked = v
            if i > 0:
                masked = _mask_rtl_charwise(masked, vals[i - 1], l)
            if i < len(vals) - 1:
                masked = _mask_rtl_charwise(masked, vals[i + 1], l)
            row_masked.append(masked)
        # Build a union mask per unique original value
        value_to_masked: dict[str, str] = {}
        for original, masked in zip(vals, row_masked):
            key = original.casefold()
            if key not in value_to_masked:
                value_to_masked[key] = masked
            else:
                prev = value_to_masked[key]
                if len(prev) != len(original):
                    prev = prev[:len(original)].ljust(len(original), '*')
                if len(masked) != len(original):
                    masked = masked[:len(original)].ljust(len(original), '*')
                merged = []
                for pc, mc, oc in zip(prev, masked, original):
                    merged.append('*' if (pc == '*' or mc == '*') else oc)
                value_to_masked[key] = ''.join(merged)
        # Second pass: apply consistent mapping by casefold key
        df[col] = pd.Series([value_to_masked[v.casefold()] for v in vals], index=series.index)
    freeze_dataframe_inplace(df)
    return df


def l_mask_by_value_similarity(df: pd.DataFrame, columns: list[str], l: int) -> pd.DataFrame:
    """
    Apply ℓ-character masking globally:
    - Find all identical values (even far apart).
    - Mask the *last l characters* of each value.
    - Apply consistently across the entire dataset.
    Modifies the provided DataFrame in-place and returns it.
    """
    # No copy: operate directly on the provided DataFrame.

    for col in columns:
        series = df[col].astype(str)
        masked_map = {}

        for val in series.unique():
            if len(val) <= l:
                masked = "*" * len(val)
            else:
                masked = val[:-l] + "*" * l
            masked_map[val] = masked

        df[col] = series.map(masked_map)

    freeze_dataframe_inplace(df)
    return df

# -------------------------
# Freeze (make DataFrame read-only in memory)
# -------------------------

def freeze_dataframe_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Make underlying arrays non‑writeable to prevent in‑place mutation downstream."""
    for col in df.columns:
        try:
            arr = df[col].to_numpy(copy=False)
            if hasattr(arr, "setflags"):
                arr.setflags(write=False)
        except Exception:
            pass
    df.attrs["_frozen"] = True
    return df

# -------------------------
# New right-to-left global masking utilities (strings & numbers)
# -------------------------

def _as_str(x) -> str:
    """Safe string conversion that keeps empty string for NaN."""
    return "" if pd.isna(x) else str(x)

def _int_digit_count_of_number(v: float) -> int:
    """Count integer digits of |v|. Values in (0,1) have 1 integer digit."""
    a = abs(v)
    int_part = int(a)
    return 1 if int_part == 0 else len(str(int_part))

def _to_float_relaxed(s: str) -> float | None:
    """Relaxed numeric parser: returns float for int/float-like strings; otherwise None.
    Accepts forms like '12', '-3', '4.5', '.75', '1.', '  2  '."""
    try:
        s = s.strip()
        if s == "":
            return None
        # Normalize leading dot (e.g., '.75') and trailing dot (e.g., '1.') are handled by float()
        return float(s)
    except Exception:
        return None

def _normalize_cell(x) -> tuple[str, bool, float | None, int]:
    """Return a tuple: (text, is_numeric, float_value_or_None, int_digit_count_if_numeric)
    - text: safe string ('' for NaN)
    - is_numeric: True if text parses as a float via _to_float_relaxed
    - float_value_or_None: parsed float or None
    - int_digit_count_if_numeric: integer digit count for |value| (1 for [0,1)) else 0 when non-numeric
    """
    text = _as_str(x)
    val = _to_float_relaxed(text)
    if val is None:
        return text, False, None, 0
    return text, True, val, _int_digit_count_of_number(val)


def mask_numbers_and_floats_by_scaled_range(series: pd.Series, l: int) -> pd.Series:
    """
    Mask whole numeric values (integers or floats) when *any other* numeric value in the
    column is within an inclusive threshold determined by the count of integer digits.

    Threshold rule (same scaling as before, now extended to floats):
      k = integer digit count of |x|  (e.g., 0.42->1, 7->1, 42->2, 999->3, 12000->5)
      threshold(x) = l * 10^(max(k-2, 0))

    If |x - y| <= threshold(x) OR |x - y| <= threshold(y), then BOTH x and y are masked.

    Non-numeric entries pass through unchanged.
    """
    vals = [_as_str(x) for x in series.tolist()]

    # Collect numeric entries with metadata: (index, value(float), k, original_length)
    num_records = []
    for idx, x in enumerate(vals):
        text, is_num, v, k = _normalize_cell(x)
        if is_num:
            num_records.append((idx, v, k, len(text)))

    to_mask = set()
    n = len(num_records)
    for a in range(n):
        idx_a, v_a, k_a, _ = num_records[a]
        thr_a = l * (10 ** max(k_a - 2, 0))
        for b in range(a + 1, n):
            idx_b, v_b, k_b, _ = num_records[b]
            thr_b = l * (10 ** max(k_b - 2, 0))
            if abs(v_a - v_b) <= thr_a or abs(v_a - v_b) <= thr_b:
                to_mask.add(idx_a)
                to_mask.add(idx_b)

    out = list(vals)
    for i in to_mask:
        out[i] = "*" * len(vals[i])

    return pd.Series(out, index=series.index)


def mask_values_by_rtl_positional_cover(series: pd.Series, l: int) -> pd.Series:
    """
    Bidirectional positional cover across the column for up to ℓ characters from either end.

    Rule (per direction):
    - For each position offset n = 1..ℓ (right-aligned or left-aligned), group rows by the
      **same alphanumeric character** at that offset (ignoring non-alphanumerics and '*').
    - If a character appears at that offset in ≥ 2 rows, mask a scaled portion of the
      suffix/prefix ending at that offset (length grows with n and ℓ).
    - Propagation: if any row already has '*' at that offset, the same suffix/prefix range is
      masked for **all** alphanumeric groups encountered at that offset.

    Works for strings, integers, and floats (using their string forms). Existing '*'
    are preserved and can only expand; they are never used for grouping. Masking is
    relaxed so that only part of the suffix/prefix is covered (scaled by ℓ), except for
    short numeric identifiers (pure digits of length 2-3) which remain fully covered.
    """
    vals = [_as_str(x) for x in series.tolist()]

    if not vals:
        return pd.Series([], dtype=series.dtype, index=series.index)

    buffers = [list(s) for s in vals]
    max_len = max(len(s) for s in vals)
    scan_limit = min(l, max_len)
    relaxed_ratio = min(0.9, 0.35 + 0.05 * max(1, min(l, 10)))

    def _cover(direction: str) -> None:
        for offset in range(1, scan_limit + 1):
            star_present = False
            groups: dict[str, list[int]] = {}

            for row_idx, buf in enumerate(buffers):
                if len(buf) < offset:
                    continue
                pos = -offset if direction == "right" else offset - 1
                ch = buf[pos]
                if ch == '*':
                    star_present = True
                    continue
                if not ch.isalnum():
                    continue
                groups.setdefault(ch.casefold(), []).append(row_idx)

            if not groups and not star_present:
                continue

            def iter_rows():
                if star_present:
                    return groups.values()
                return (rows for rows in groups.values() if len(rows) >= 2)

            for rows in iter_rows():
                for row_idx in rows:
                    original = vals[row_idx]
                    buf = buffers[row_idx]
                    is_short_numeric = original.isdigit() and 2 <= len(original) <= 3

                    if direction == "right":
                        cover_len = offset if is_short_numeric else max(
                            1, min(offset, math.ceil(offset * relaxed_ratio))
                        )
                        start = max(0, len(buf) - cover_len)
                        for j in range(start, len(buf)):
                            if buf[j] != '*':
                                buf[j] = '*'
                    else:
                        cover_len = offset if is_short_numeric else max(
                            1, min(offset, math.ceil(offset * relaxed_ratio))
                        )
                        end = min(len(buf), cover_len)
                        for j in range(0, end):
                            if buf[j] != '*':
                                buf[j] = '*'

    _cover("right")
    _cover("left")

    # Propagate masking to characters that are adjacent (within one position)
    # to masked entries in the immediate previous/next rows. This helps cover
    # near matches that differ by slight positional shifts while keeping the
    # overall logic intact.
    if len(buffers) > 1 and scan_limit > 0:
        snapshots = [buf.copy() for buf in buffers]

        def within_l_range(buf_len: int, pos: int) -> bool:
            """Check whether index `pos` falls within the first/last ℓ characters."""
            return (pos + 1) <= l or (buf_len - pos) <= l

        for row_idx, buf in enumerate(buffers):
            buf_len = len(buf)
            if buf_len == 0:
                continue
            neighbor_indices = []
            if row_idx > 0:
                neighbor_indices.append(row_idx - 1)
            if row_idx < len(buffers) - 1:
                neighbor_indices.append(row_idx + 1)
            if not neighbor_indices:
                continue

            for pos, ch in enumerate(buf):
                if ch == '*' or not ch.isalnum():
                    continue
                if not within_l_range(buf_len, pos):
                    continue

                right_offset = buf_len - pos - 1
                should_mask = False

                for neighbor_idx in neighbor_indices:
                    neighbor = snapshots[neighbor_idx]
                    neighbor_len = len(neighbor)
                    if neighbor_len == 0:
                        continue

                    # Left-aligned neighbors
                    for delta in (-1, 0, 1):
                        npos = pos + delta
                        if 0 <= npos < neighbor_len and neighbor[npos] == '*':
                            should_mask = True
                            break
                    if should_mask:
                        break

                    # Right-aligned neighbors (match by suffix position)
                    neighbor_pos = neighbor_len - right_offset - 1
                    for delta in (-1, 0, 1):
                        npos = neighbor_pos + delta
                        if 0 <= npos < neighbor_len and neighbor[npos] == '*':
                            should_mask = True
                            break
                    if should_mask:
                        break

                if should_mask:
                    buf[pos] = '*'

    out = [''.join(buf) for buf in buffers]
    return pd.Series(out, index=series.index)


def right_to_left_global_l_mask(df: pd.DataFrame, columns: list[str], l: int) -> pd.DataFrame:
    """
    Apply the requested behavior column-wise:
      - Always process from the edges inward over the last/first ℓ positions for strings.
      - For strings *and* numeric strings: for each right-aligned or left-aligned
        position n ≤ ℓ, if the **same alphanumeric character** appears in ≥ 2 rows at
        that position, mask the entire **suffix/prefix of length n** for those rows.
      - For numeric values additionally: mask the *entire* number if there exists any
        other number in the column within ±(l * 10^(max(k-2,0))) of it, where k is the
        digit count of the integer part of the absolute value.

    Mixed-type columns are handled per cell: integer-like entries follow the numeric
    rule; other entries follow the string rule.
    Modifies the provided DataFrame in-place and returns it.
    """
    # No copy: operate directly on the provided DataFrame.

    for col in columns:
        # First handle numbers (entire-value masking where range condition holds)
        after_numbers = mask_numbers_and_floats_by_scaled_range(df[col].astype(object).astype(str), l)
        # Then handle strings right-to-left (numbers pass through unchanged)
        df[col] = mask_values_by_rtl_positional_cover(after_numbers, l)

    freeze_dataframe_inplace(df)
    return df

# -------------------------
# Finalization helpers (no algorithm change)
# -------------------------

def finalize_l_diversity(df: pd.DataFrame, columns: Sequence[str], l: int) -> pd.DataFrame:
    """
    Produce a **frozen** masked DataFrame from a deep copy of *df*.
    The original *df* is untouched; assign the return value to replace it:
        df = finalize_l_diversity(df, ["col1", "col2"], l)
    """
    out = right_to_left_global_l_mask(df.copy(deep=True), list(columns), l)
    # Already frozen inside, but safe to ensure
    return freeze_dataframe_inplace(out)


def finalize_repo_l_diversity_inplace(repo, columns: Sequence[str], l: int):
    """
    Apply masking on a deep copy of repo.frame, then **replace** repo._frame with a
    frozen result. This drops references to the original table from the repository
    object so subsequent steps operate only on the masked, read‑only data.
    """
    df_out = right_to_left_global_l_mask(repo.frame.copy(deep=True), list(columns), l)
    freeze_dataframe_inplace(df_out)
    # Replace the repository's active frame with the frozen masked version
    if hasattr(repo, "_frame"):
        repo._frame = df_out
    # Remove any known backups related to previous anonymizations (best effort)
    for attr in ("_k_backup_cols", "_l_backup"):
        if hasattr(repo, attr):
            try:
                delattr(repo, attr)
            except Exception:
                pass
    # Mark for clarity
    setattr(repo, "_l_finalized", True)
    setattr(repo, "_l", l)
    setattr(repo, "_l_cols", list(columns))
    return repo
