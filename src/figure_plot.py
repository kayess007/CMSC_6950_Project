from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROCESSED_FILE = "processed_en_climate_daily_NL_8403603_2024_P1D.csv"

def my_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "lines.linewidth": 1.4
    })

def load_processed_df():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / PROCESSED_FILE)
    date_col = next(c for c in df.columns if "Date" in c or "date" in c)
    temp_col = next(c for c in df.columns if "Mean Temp" in c or "mean temp" in c)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, temp_col]).sort_values(date_col)
    return df, date_col, temp_col, project_root

def format_months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

def fig1(df, date_col, temp_col, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[date_col], df[temp_col])
    if "Rolling_7d" in df.columns:
        ax.plot(df[date_col], df["Rolling_7d"])
    ax.set_title("Daily Temperature (2024)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temp (°C)")
    format_months(ax)
    fig.tight_layout()
    fig.savefig(out / "daily_temperature.png")
    plt.close(fig)

def fig2(df, date_col, temp_col, out):
    s = df[temp_col]
    mu, sd = s.mean(), s.std(ddof=1)
    z = (s - mu) / sd
    extremes = df[z.abs() >= 2.5]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[date_col], df[temp_col])
    ax.scatter(extremes[date_col], extremes[temp_col])
    ax.set_title("Extreme Days (|z|>=2.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temp (°C)")
    format_months(ax)
    fig.tight_layout()
    fig.savefig(out / "extreme_days.png")
    plt.close(fig)

def fig3(df, date_col, temp_col, out):
    monthly = df.set_index(date_col)[temp_col].resample("M").agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["mean"])
    ax.plot(monthly.index, monthly["min"])
    ax.plot(monthly.index, monthly["max"])
    ax.set_title("Monthly Summary")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temp (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    fig.savefig(out / "monthly_summary.png")
    plt.close(fig)

def fig4(df, date_col, temp_col, out):
    y = df[temp_col].values
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    trend = a * x + b
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[date_col], df[temp_col])
    ax.plot(df[date_col], trend, linestyle="--")
    ax.set_title("Trend Line")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temp (°C)")
    format_months(ax)
    fig.tight_layout()
    fig.savefig(out / "trendline.png")
    plt.close(fig)

def fig5_season_scatter(df, date_col, temp_col, out):
    df = df.copy()
    df["DayOfYear"] = df[date_col].dt.dayofyear
    def season(m):
        if m in [12, 1, 2]: return "Winter"
        if m in [3, 4, 5]: return "Spring"
        if m in [6, 7, 8]: return "Summer"
        return "Fall"
    df["Season"] = df[date_col].dt.month.apply(season)
    season_colors = {"Winter": "blue", "Spring": "green", "Summer": "red", "Fall": "orange"}
    fig, ax = plt.subplots(figsize=(12, 6))
    for s, color in season_colors.items():
        d = df[df["Season"] == s]
        ax.scatter(d["DayOfYear"], d[temp_col], s=25, color=color, alpha=0.8, label=s)
    ax.set_xlabel("Day of Year (1-365)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature vs. Day of Year Grouped by Season (2024)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "season_scatter.png")
    plt.close(fig)

def fig6_heatmap_day_vs_month(df, date_col, temp_col, out):
    df = df.copy()
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    heatmap = df.pivot_table(index="Day", columns="Month", values=temp_col)
    fig, ax = plt.subplots(figsize=(12, 6))
    c = ax.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    fig.colorbar(c, ax=ax, label="Temperature (°C)")
    ax.set_title("2D Temperature Heatmap (Day vs Month, 2024)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Month")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "Sept", "Oct", "Nov", "Dec"
    ])
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_yticklabels(range(1, 32, 5))
    fig.tight_layout()
    fig.savefig(out / "heatmap_day_vs_month.png")
    plt.close(fig)

def fig7_histogram(df, temp_col, out):
    temps = df[temp_col]
    plt.figure(figsize=(10, 6))
    plt.hist(temps, bins=30, color="orange", edgecolor="black", alpha=0.85)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Temperatures (2024)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / "histogram_temperature.png")
    plt.close()

def fig8_monthly_mean_bar(df, date_col, temp_col, out):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Month"] = df[date_col].dt.month
    monthly_mean = df.groupby("Month")[temp_col].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_mean.index, monthly_mean.values, color="skyblue", edgecolor="black")
    plt.xlabel("Month")
    plt.ylabel("Mean Temperature (°C)")
    plt.title("Monthly Average Temperatures (2024)")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(out / "monthly_mean_bar.png")
    plt.close()

def main():
    my_style()
    df, date_col, temp_col, project_root = load_processed_df()
    out = project_root / "figures"
    out.mkdir(exist_ok=True)
    fig1(df, date_col, temp_col, out)
    fig2(df, date_col, temp_col, out)
    fig3(df, date_col, temp_col, out)
    fig4(df, date_col, temp_col, out)
    fig5_season_scatter(df, date_col, temp_col, out)
    fig6_heatmap_day_vs_month(df, date_col, temp_col, out)
    fig7_histogram(df, temp_col, out)
    fig8_monthly_mean_bar(df, date_col, temp_col, out)
    print("Generated all 8 figures in /figures")

if __name__ == "__main__":
    main()
