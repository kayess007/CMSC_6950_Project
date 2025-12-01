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
    wind_col = next(c for c in df.columns if "gust" in c.lower())
    df[date_col] = pd.to_datetime(df[date_col])
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    df = df.dropna(subset=[date_col, temp_col]).sort_values(date_col)
    return df, date_col, temp_col, wind_col, project_root

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
    fig.savefig(out / "fig1_daily_temperature.png")
    plt.close(fig)

def fig2(df, date_col, temp_col, out):
    s = df[temp_col]
    mu, sd = s.mean(), s.std(ddof=1)
    z = (s - mu) / sd
    extremes = df[z.abs() >= 2.0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[date_col], df[temp_col])
    ax.scatter(extremes[date_col], extremes[temp_col], color="red")
    ax.set_title("Extreme Temperature Days (|z|≥2.0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temp (°C)")
    format_months(ax)
    fig.tight_layout()
    fig.savefig(out / "fig2_extreme_days.png")
    plt.close(fig)

def fig3(df, date_col, temp_col, out):
    monthly = df.set_index(date_col)[temp_col].resample("M").agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly["mean"])
    ax.plot(monthly.index, monthly["min"])
    ax.plot(monthly.index, monthly["max"])
    ax.set_title("Monthly Temperature Summary")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temp (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    fig.savefig(out / "fig3_monthly_summary.png")
    plt.close(fig)

def fig4(df, date_col, temp_col, out):
    df = df.copy()
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    heatmap = df.pivot_table(index="Day", columns="Month", values=temp_col)
    fig, ax = plt.subplots(figsize=(12, 6))
    c = ax.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    fig.colorbar(c, ax=ax, label="Temperature (°C)")
    ax.set_title("2D Temperature Heatmap (Day vs Month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "Sept", "Oct", "Nov", "Dec"
    ])
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_yticklabels(range(1, 32, 5))
    fig.tight_layout()
    fig.savefig(out / "fig4_heatmap.png")
    plt.close(fig)

def fig5(df, temp_col, out):
    temps = df[temp_col]
    plt.figure(figsize=(10, 6))
    plt.hist(temps, bins=30, color="orange", edgecolor="black")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Temperatures (2024)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / "fig5_temperature_histogram.png")
    plt.close()


def fig6(df, date_col, wind_col, out):
    df = df.copy()
    df["Month"] = df[date_col].dt.month

    fig, ax = plt.subplots(figsize=(12, 5))

    for m in range(1, 13):
        d = df[df["Month"] == m]

        x = np.random.normal(m, 0.12, size=len(d))
        ax.scatter(x, d[wind_col], s=25, alpha=0.7)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Daily Max Wind Gusts Grouped by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind Gust (km/h)")

    fig.tight_layout()
    fig.savefig(out / "fig6_wind_gust_jitter.png")
    plt.close(fig)


def fig7(df, wind_col, out):
    plt.figure(figsize=(10, 5))
    plt.hist(df[wind_col].dropna(), bins=20, color="orange", edgecolor="black")
    plt.xlabel("Wind Gust (km/h)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Max Wind Gusts")
    plt.tight_layout()
    plt.savefig(out / "fig7_wind_gust_histogram.png")
    plt.close()

def fig8(df, date_col, temp_col, wind_col, out):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df[date_col], df[temp_col], color="orange")
    ax1.set_ylabel("Temperature (°C)")
    ax2 = ax1.twinx()
    ax2.scatter(df[date_col], df[wind_col], s=14, color="blue", alpha=0.6)
    ax2.set_ylabel("Wind Gust (km/h)")
    ax1.set_title("Temperature vs Wind Gust (Dual Axis)")
    format_months(ax1)
    fig.tight_layout()
    fig.savefig(out / "fig8_temp_vs_wind.png")
    plt.close(fig)

def main():
    my_style()
    df, date_col, temp_col, wind_col, project_root = load_processed_df()
    out = project_root / "figures"
    out.mkdir(exist_ok=True)

    fig1(df, date_col, temp_col, out)
    fig2(df, date_col, temp_col, out)
    fig3(df, date_col, temp_col, out)
    fig4(df, date_col, temp_col, out)
    fig5(df, temp_col, out)   
    fig6(df, date_col, wind_col, out)
    fig7(df, wind_col, out)
    fig8(df, date_col, temp_col, wind_col, out)

    print("Generated all 8 figures in /figures")

if __name__ == "__main__":
    main()
