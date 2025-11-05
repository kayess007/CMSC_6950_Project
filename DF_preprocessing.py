import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

csv_path = r"C:\Users\kbaah\Desktop\project proposal\en_climate_daily_NL_8403603_2024_P1D.csv"
out_dir  = Path(r"C:\Users\kbaah\Desktop\CMSC_6950_Project")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "mean_temp_2024.png"

df = pd.read_csv(csv_path)

df.columns = [c.strip() for c in df.columns]

date_col  = [c for c in df.columns if "Date" in c][0]
tmean_col = [c for c in df.columns if "Mean Temp" in c][0]  

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


missing_flags = {"M": None, "E": None, "": None, "NA": None, "—": None, "-": None}
df[tmean_col] = pd.to_numeric(df[tmean_col].replace(missing_flags), errors="coerce")


df = df.dropna(subset=[date_col, tmean_col]).sort_values(date_col)
df = df[~df[date_col].duplicated(keep="first")]


df = df[df[tmean_col].between(-50, 40)]


df = df.set_index(date_col)

df["Rolling_7d"] = df[tmean_col].rolling(window=7, min_periods=1).mean()


plt.figure(figsize=(12, 5.5))
plt.plot(df.index, df[tmean_col], label="Daily Mean Temp (°C)", linewidth=1)
plt.plot(df.index, df["Rolling_7d"], label="7-Day Rolling Mean (°C)", linewidth=2)

plt.title("Daily Mean Temperature — 2024\nSt. John's West Climate Station (8403603)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.savefig(out_file, dpi=180, bbox_inches="tight")
plt.show() 
plt.close()

print(f"Plot saved to: {out_file}")
