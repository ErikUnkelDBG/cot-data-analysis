from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent

# 1. Check historical Data
df_hist = pd.read_parquet(base_dir / "historical_data.parquet")
print("=== HISTORICAL DATA ===")
print("Zeilen & Spalten:", df_hist.shape)
print("Datumsbereich:", df_hist["Date"].min(), "bis", df_hist["Date"].max())
print("Vorhandene Märkte:\n", df_hist["Market Name"].value_counts())
print("\nBeispiel-Zeilen (Neueste Werte):")
print(
    df_hist.sort_values("Date")
    .groupby("Market Name")
    .last()[
        [
            "Date",
            "Open Positions",
            "Long Positions",
            "Short Positions",
            "Net Leverage",
        ]
    ]
)

# 2. Check HP Filter
df_hp = pd.read_parquet(base_dir / "hp_filter_data.parquet")
print("\n=== HP FILTER DATA ===")
print("Zeilen:", len(df_hp))
print("Null-Werte in Trend/Cycle?:", df_hp[["Trend", "Cycle"]].isna().sum())
print("\nBeispiel HP-Filter:")
print(df_hp.head(5))
