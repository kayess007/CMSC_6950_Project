import pandas as pd
import numpy as np
from pathlib import Path


def find_column(df: pd.DataFrame, contains: str) -> str:
    matches = [c for c in df.columns if contains.lower() in c.lower()]
    if not matches:
        raise KeyError(f"No column containing '{contains}' found. Columns: {list(df.columns)}")
    return matches[0]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_dates(df: pd.DataFrame, date_like: str | None = None):
    df = df.copy()
    date_col = date_like or find_column(df, "date")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df, date_col


def coerce_temperature(df: pd.DataFrame, temp_like: str | None = None):
    df = df.copy()
    temp_col = temp_like or find_column(df, "mean temp")
    df[temp_col] = pd.to_numeric(df[temp_col].replace({"M": np.nan}), errors="coerce")
    return df, temp_col


def filter_temp_range(df: pd.DataFrame, temp_col: str,
                      min_val: float = -50, max_val: float = 40):
    return df[df[temp_col].between(min_val, max_val)].dropna(subset=[temp_col])


def add_rolling_mean(df: pd.DataFrame, date_col: str, temp_col: str,
                     window: int = 7, out_col: str = "Rolling_7d"):
    df = df.sort_values(date_col).copy()
    df[out_col] = df[temp_col].rolling(window=window, min_periods=1).mean()
    return df


def load_and_process(raw_path: str):

    df = pd.read_csv(raw_path)

    df = standardize_columns(df)
    df, date_col = parse_dates(df)
    df, temp_col = coerce_temperature(df)
    df = filter_temp_range(df, temp_col)
    df = add_rolling_mean(df, date_col, temp_col)

   
    project_root = Path(__file__).resolve().parents[1]  
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    raw_file = Path(raw_path).name
    processed_name = f"processed_{raw_file}"
    processed_path = data_dir / processed_name

    df.to_csv(processed_path, index=False)
    print(f"Processed file saved to: {processed_path}")

    return df, date_col, temp_col, processed_path


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]
    raw = project_root / "data" / "en_climate_daily_NL_8403603_2024_P1D.csv"

    df, date_col, temp_col, outpath = load_and_process(str(raw))

    print(df.head())
    print(df.describe())
