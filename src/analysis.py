import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def compute_daily_statistics(df: pd.DataFrame, temp_col: str):
    s = df[temp_col].dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return mean_val, median_val, std_val


def identify_extremes(df: pd.DataFrame, temp_col: str, z_threshold: float = 2.0):
    s = df[temp_col].dropna()
    if len(s) == 0:
        return df.iloc[0:0]

    mean_val = s.mean()
    std_val = s.std(ddof=1)

    if std_val == 0 or np.isnan(std_val):
        return df.iloc[0:0]

    z = (s - mean_val) / std_val
    return df[(z.abs() >= z_threshold)]


def monthly_summary(df: pd.DataFrame, date_col: str, temp_col: str) -> pd.DataFrame:

    g = df.set_index(date_col)[temp_col].groupby(pd.Grouper(freq="ME"))

    out = pd.DataFrame({
        "mean": g.mean(),
        "min": g.min(),
        "max": g.max(),
        "count": g.count(),
    }).reset_index().rename(columns={date_col: "month"})
    return out


if __name__ == "__main__":
    data_dir = project_root / "data"
    processed_file = data_dir / "processed_en_climate_daily_NL_8403603_2024_P1D.csv"

    df = pd.read_csv(processed_file)

    date_col = next((c for c in df.columns if "Date" in c or "date" in c), None)
    temp_col = next((c for c in df.columns if "Mean Temp" in c or "mean temp" in c), None)

    if date_col is None or temp_col is None:
        print("Could not identify date or temperature column. Please check your CSV columns.")
        sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    mean_val, median_val, std_val = compute_daily_statistics(df, temp_col)
    print("\n=== BASIC STATISTICS ===")
    print(f"Mean:   {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"StdDev: {std_val:.2f}")

    extremes = identify_extremes(df, temp_col)
    print(f"\nNumber of extreme days: {len(extremes)}")

    monthly = monthly_summary(df, date_col, temp_col)
    print("\nMonthly summary (first few rows):")
    print(monthly.head())

