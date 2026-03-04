"""Data transformation functions."""
import pandas as pd
from typing import Optional

def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

def filter_valid_records(df: pd.DataFrame, min_value: Optional[float] = None) -> pd.DataFrame:
    mask = df['value'].notna()
    if min_value is not None:
        mask &= df['value'] >= min_value
    else:
        mask &= df['value'] > 0
    return df[mask]

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_timestamps(df)
    df = filter_valid_records(df)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    return df
