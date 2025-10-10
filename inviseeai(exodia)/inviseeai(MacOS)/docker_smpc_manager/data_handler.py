"""
Keep only numeric columns and (optionally) exploded Y-M-D parts
from any column that parses as a calendar date, and perform
feature engineering on dates to create ordinal and LOS features.
"""
import pandas as pd

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Retain numeric columns and explode any datetime columns into year/month/day parts
    keep = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            keep.append(col)
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            keep.extend([f"{col}_year", f"{col}_month", f"{col}_day"])

    df = df[keep].copy()

    # 2. Feature engineering for date columns: combine Y/M/D into ordinal and derive LOS
    date_parts = [
        ("Date of Admission_year", "Date of Admission_month", "Date of Admission_day", "admit_ord"),
        ("Discharge Date_year", "Discharge Date_month", "Discharge Date_day", "disch_ord"),
    ]
    for y_col, m_col, d_col, new_col in date_parts:
        if {y_col, m_col, d_col}.issubset(df.columns):
            dates = pd.to_datetime(
                dict(year=df[y_col], month=df[m_col], day=df[d_col]),
                errors="coerce",
            )
            df[new_col] = dates.map(lambda d: d.toordinal() if pd.notnull(d) else pd.NA)
            df.drop(columns=[y_col, m_col, d_col], inplace=True)

    # Length-of-stay in days
    if {"admit_ord", "disch_ord"}.issubset(df.columns):
        df["los_days"] = df["disch_ord"] - df["admit_ord"]

    # 3. Ensure numeric dtype and fill any NaNs introduced by parsing
    df = df.apply(pd.to_numeric, errors="ignore")
    if df.isna().any().any():
        df = df.fillna(df.mean(numeric_only=True))

    # Return only numeric columns ready for modelling
    return df.select_dtypes(include="number")
