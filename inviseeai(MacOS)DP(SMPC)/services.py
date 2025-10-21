import pandas as pd
from io import StringIO

from scipy.optimize._lsap import linear_sum_assignment

from k_anonymizer import kanonymize
from data_repository import DataRepository
from l_diversity import l_diversity_mask
from summary_stats import simple_summary, format_summary_table


def load_repo_from_csv(csv_text, usecols=None):
    df = pd.read_csv(StringIO(csv_text), usecols=usecols)
    return DataRepository(df)


def apply_k_anonymity(csv_text, quasi_columns, k):
    repo = load_repo_from_csv(csv_text)
    result_repo = kanonymize(repo, quasi_columns, k)
    return result_repo.frame


def generate_summary(csv_text):
    repo = load_repo_from_csv(csv_text)
    summary = simple_summary(repo)
    return format_summary_table(summary)

# 25/7
# def apply_l_contextual_mask(csv_text, columns, l):
#     df = pd.read_csv(StringIO(csv_text))
#     new_df = vectorized_l_contextual_mask(df, columns=columns, l=l)
#     return new_df


def apply_l_contextual_mask(csv_text, columns, l):
    """
    Enforce ℓ‑diversity suppression.

    Behaviour
    ---------
    • If the caller provides *only* the sensitive column, suppression is decided
      **globally**: any sensitive value that appears ≥ ℓ times in the column is
      replaced by `'*'`.

    • If one or more additional columns are supplied, they define the
      equivalence class.  Every row whose class contains ≥ ℓ *distinct*
      sensitive values is suppressed.

    Parameters
    ----------
    csv_text : str
        Raw CSV data.
    columns : list[str]
        User‑selected columns. The first entry is interpreted as the sensitive
        attribute; any subsequent entries are quasi‑identifiers.
    l : int
        ℓ‑diversity threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame where the sensitive attribute is replaced by `'*'` either
        globally (frequency ≥ ℓ) or within each class (diversity ≥ ℓ).
    """
    df = pd.read_csv(StringIO(csv_text))

    if not columns:
        raise ValueError(
            "Select at least one column. "
            "The first will be used as the sensitive attribute."
        )

    sensitive_col = columns[0]
    # If the user selects no quasi‑identifiers, pass an empty list so that
    # l_diversity_mask operates in GLOBAL mode.
    quasi_cols = columns[1:]

    return l_diversity_mask(
        df,
        quasi_cols=quasi_cols,
        sensitive_col=sensitive_col,
        l=l,
        sentinel="*",
    )

from t_closeness import apply_t_closeness as t_closeness_core


def apply_t_closeness(data, quasi_columns, sensitive_column, t_value):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(StringIO(data))
    new_df = t_closeness_core(df, quasi_columns, sensitive_column, t_value)
    return new_df


# def run_privacy_advisor(csv_text):
#     df = pd.read_csv(StringIO(csv_text))
#     advisor = PrivacyAdvisor(use_ml=True, use_llm=True)
#     result = advisor.analyze(df)
#     return result



# In services.py...
import pandas as pd
from io import StringIO
from date_shifter import shift_dates # Import the new function

# ... (your other service functions like apply_k_anonymity)

def apply_date_shifting(csv_text: str, id_col: str, date_cols: list, max_days: int) -> pd.DataFrame:
    """
    Reads a CSV string, applies date shifting, and returns the resulting DataFrame.
    """
    df = pd.read_csv(StringIO(csv_text))
    shifted_df = shift_dates(df, id_col, date_cols, max_days)
    return shifted_df