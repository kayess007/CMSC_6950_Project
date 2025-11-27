from pathlib import Path
import pandas as pd

from analysis import (
    compute_daily_statistics,
    identify_extremes,
    monthly_summary,
 )


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    processed_file = data_dir / "processed_en_climate_daily_NL_8403603_2024_P1D.csv"
    df = pd.read_csv(processed_file)

    date_col = [c for c in df.columns if "Date" in c or "date" in c][0]
    temp_col = [c for c in df.columns if "Mean Temp" in c or "mean temp" in c][0]

   
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    mean_val, median_val, std_val = compute_daily_statistics(df, temp_col)
    print(f"Mean={mean_val:.2f}  Median={median_val:.2f}  Std={std_val:.2f}")

    extremes = identify_extremes(df, temp_col)
    print(f"Extreme days: {len(extremes)}")

    monthly = monthly_summary(df, date_col, temp_col)
    print(monthly.head())

    print(f"\nUsing processed file: {processed_file}")


if __name__ == "__main__":
    main()


