import pandas as pd


def stable_sort(base: pd.DataFrame) -> pd.DataFrame:
    """Sort the DataFrame by ID_PATIENT then by TIMESTAMP using mergesort"""
    base = base.sort_values(["ID_PATIENT", "TIMESTAMP"], kind="mergesort").reset_index(
        drop=True
    )
    return base
