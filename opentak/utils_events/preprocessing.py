import pandas as pd


def stable_sort(base: pd.DataFrame) -> pd.DataFrame:
    """Ordonne la base selon ID_PATIENT puis TIMESTAMP, par un mergesort (donc algo de tri stable)."""
    base = base.sort_values(["ID_PATIENT", "TIMESTAMP"], kind="mergesort").reset_index(drop=True)
    return base
