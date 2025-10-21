from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


class DataRepository:
    def __init__(self, frame: pd.DataFrame, *, source: Path | None = None):
        self._frame = frame.copy()
        self._source = source
        self._inplace = False
    def set_inplace(self, enabled: bool = True) -> "DataRepository":
        """Enable or disable in-place updates for subsequent transformations and return self."""
        self._inplace = bool(enabled)
        return self

    # ---------------------------------------------------------------------
    # Constructors / loaders
    # ---------------------------------------------------------------------
    @classmethod
    def from_file(
            cls,
            file_path: str | Path,
            *,
            usecols: Iterable[str] | None = None,
            **read_kwargs,
    ) -> "DataRepository":
        path = Path(file_path).expanduser().resolve()
        suffix = path.suffix.lower()

        # auto-detect columns containing "date" to parse as datetime
        normalized_usecols = usecols
        if usecols is not None and not callable(usecols):
            normalized_usecols = tuple(usecols)

        if normalized_usecols is None or callable(normalized_usecols):
            date_cols = None
        else:
            date_cols = [c for c in normalized_usecols if isinstance(c, str) and "date" in c.lower()]
            date_cols = date_cols or None

        if suffix == ".csv":
            df = pd.read_csv(path, usecols=normalized_usecols, parse_dates=date_cols, **read_kwargs)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, usecols=normalized_usecols, engine="openpyxl", parse_dates=date_cols, **read_kwargs)
        else:
            raise ValueError(f"Unsupported file extension '{path.suffix}'.")

        return cls(df, source=path)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        """Return the DataFrame.
        - If in-place mode is enabled, return the internal frame (mutable reference).
        - Otherwise, return a defensive copy.
        """
        return self._frame if getattr(self, "_inplace", False) else self._frame.copy()

    @frame.setter
    def frame(self, value: pd.DataFrame) -> None:
        """Replace the internal DataFrame with a copy of `value`."""
        self._frame = value.copy()

    # ------------------------------------------------------------------
    # Transformations – all copy‑on‑write unless explicitly documented
    # ------------------------------------------------------------------
    def apply(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "DataRepository":
        """Apply a function that transforms the DataFrame."""
        new_df = func(self._frame.copy())
        if len(new_df) != len(self._frame):
            raise ValueError("Transform must preserve row count; use `filter_rows` to drop rows.")
        if self._inplace:
            self._frame = new_df
            return self
        return DataRepository(new_df, source=self._source)

    def rename_columns(self, mapping: dict[str, str]) -> "DataRepository":
        """Rename columns; updates in-place if enabled, else returns a new repository."""
        if self._inplace:
            self._frame.rename(columns=mapping, inplace=True)
            return self
        return DataRepository(self._frame.rename(columns=mapping), source=self._source)

    def filter_rows(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "DataRepository":
        """Keep rows where predicate is True; updates in-place if enabled, else returns a new repository."""
        mask = predicate(self._frame)
        if mask.dtype != bool or mask.size != self._frame.shape[0]:
            raise ValueError("Predicate must return a boolean Series aligned with the DataFrame.")
        if self._inplace:
            self._frame = self._frame.loc[mask].copy()
            return self
        return DataRepository(self._frame.loc[mask].copy(), source=self._source)

    def clone(self) -> "DataRepository":
        """Return a new DataRepository with a deep copy of the frame and same inplace flag."""
        clone = DataRepository(self._frame.copy(), source=self._source)
        clone._inplace = self._inplace
        return clone

    def sort_by_column(self, column: str, ascending: bool = True) -> "DataRepository":
        """Sort by column; updates in-place if enabled, else returns a new repository."""
        df_sorted = self._frame.sort_values(by=column, ascending=ascending).reset_index(drop=True)
        if self._inplace:
            self._frame = df_sorted
            return self
        clone = self.clone()
        clone._frame = df_sorted
        return clone

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_csv(self, file_path: str | Path, **kwargs) -> Path:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_csv(path, index=False, **kwargs)
        return path

    def save_parquet(self, file_path: str | Path, **kwargs) -> Path:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(path, index=False, **kwargs)
        return path

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"DataRepository(rows={len(self)}, cols={self._frame.shape[1]})"
