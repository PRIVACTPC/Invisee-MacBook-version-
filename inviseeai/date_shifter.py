import pandas as pd
import numpy as np
from typing import Sequence

def shift_dates(df: pd.DataFrame, id_col: str, date_cols: Sequence[str], max_shift_days: int) -> pd.DataFrame:
    """
    Shifts date/datetime columns by a random number of days, consistent for each unique ID.

    This method is useful for anonymizing dates while preserving the intervals
    between events for each individual subject (identified by id_col).

    Args:
        df: The DataFrame to modify.
        id_col: The column name containing the unique identifier for each subject.
        date_cols: A list of date/datetime column names to be shifted.
        max_shift_days: The maximum number of days (positive or negative) to shift the dates.
                        The shift will be a random integer between -max_shift_days and +max_shift_days.

    Returns:
        A new DataFrame with the specified date columns shifted.
    """
    if not date_cols:
        return df.copy()

    if id_col not in df.columns:
        raise ValueError(f"Identifier column '{id_col}' not found in DataFrame.")

    df_shifted = df.copy()

    # Create a consistent random shift for each unique ID
    unique_ids = df_shifted[id_col].dropna().unique()
    shifts = pd.Series(
        np.random.randint(-max_shift_days, max_shift_days + 1, size=len(unique_ids)),
        index=unique_ids
    )

    # Map the shifts to each row based on the ID
    shift_map = df_shifted[id_col].map(shifts)
    time_delta = pd.to_timedelta(shift_map, unit='D')

    # Apply the shift to each specified date column
    for col in date_cols:
        if col not in df_shifted.columns:
            print(f"Warning: Date column '{col}' not found, skipping.")
            continue
        # Ensure column is in datetime format, coercing errors to NaT
        original_dtype = df_shifted[col].dtype
        df_shifted[col] = pd.to_datetime(df_shifted[col], errors='coerce')

        # Apply the shift (NaT + timedelta results in NaT, which is correct)
        df_shifted[col] += time_delta
        
        # Optional: attempt to convert back to original format if it was date only
        if 'date' in str(original_dtype):
             df_shifted[col] = df_shifted[col].dt.date


    return df_shifted