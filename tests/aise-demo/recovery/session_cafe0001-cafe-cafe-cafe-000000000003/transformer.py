"""Data transformation functions."""
import pandas as pd

def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

def filter_valid_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['value'].notna() & (df['value'] > 0)]

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_timestamps(df)
    df = filter_valid_records(df)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    return df
