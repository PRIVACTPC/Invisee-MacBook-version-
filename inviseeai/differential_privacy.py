import pandas as pd
import numpy as np

def mask_string_columns(df: pd.DataFrame):
    masked_df = df.copy()
    for col in masked_df.select_dtypes(include='object').columns:
        masked_df[col] = masked_df[col].apply(lambda x: "***" if pd.notna(x) else x)
    return masked_df

def apply_differential_privacy(df: pd.DataFrame, column_epsilons: dict):
    dp_df = df.copy()
    for col, epsilon in column_epsilons.items():
        if col in dp_df.columns and pd.api.types.is_numeric_dtype(dp_df[col]):
            sensitivity = dp_df[col].max() - dp_df[col].min()
            noise = np.random.laplace(loc=0.0, scale=sensitivity/epsilon, size=len(dp_df))
            dp_df[col] = dp_df[col] + noise
    return dp_df
