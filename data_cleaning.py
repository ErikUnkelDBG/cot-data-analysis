import datetime
import io
import os
from pathlib import Path
import zipfile
import pandas as pd
import requests
import statsmodels.api as sm

# ==========================================
# CONFIG & MAPPINGS
# ==========================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

MARKET_CODE_MAPPING = {
    "020601": "UST Bond",
    "020604": "Ultra UST Bond",
    "042601": "UST 2Y Note",
    "043602": "UST 10Y Note",
    "043607": "Ultra UST 10Y",
    "044601": "UST 5Y Note",
}

COLS_TO_USE = [
    "As_of_Date_In_Form_YYMMDD",
    "CFTC_Contract_Market_Code",
    "Open_Interest_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
]

HP_VARIABLES = [
    "Net Leverage",
    "Net Leverage Ratio",
    "Gross Short Ratio",
    "Open Positions",
    "Long Positions",
    "Short Positions",
]


# ==========================================
# 1. CLEANING & DOWNLOADS
# ==========================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names, filters markets, and calculates basic metrics."""
    df.columns = df.columns.str.strip()

    available_cols = [c for c in COLS_TO_USE if c in df.columns]
    df = df[available_cols].copy()

    df["CFTC_Contract_Market_Code"] = (
        df["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    )
    df = df[
        df["CFTC_Contract_Market_Code"].isin(MARKET_CODE_MAPPING.keys())
    ].copy()

    df.rename(
        columns={
            "As_of_Date_In_Form_YYMMDD": "Date",
            "CFTC_Contract_Market_Code": "Market Name",
            "Open_Interest_All": "Open Positions",
            "Lev_Money_Positions_Long_All": "Long Positions",
            "Lev_Money_Positions_Short_All": "Short Positions",
        },
        inplace=True,
    )

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str).str.zfill(6), format="%y%m%d", errors="coerce"
    )

    df["Market Name"] = df["Market Name"].map(MARKET_CODE_MAPPING)

    num_cols = ["Open Positions", "Long Positions", "Short Positions"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df["Net Leverage"] = df["Long Positions"] - df["Short Positions"]
    df["Net Leverage Ratio"] = 100 * (
        df["Net Leverage"] / df["Open Positions"]
    )
    df["Gross Short Ratio"] = 100 * (
        df["Short Positions"] / df["Open Positions"]
    )

    return df


def download_cftc_zip(url: str, label: str) -> pd.DataFrame:
    """Downloads any CFTC ZIP archive and cleans the text file."""
    print(f"Lade {label}: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            txt_files = [f for f in z.namelist() if f.endswith(".txt")]
            if not txt_files:
                return pd.DataFrame()
            with z.open(txt_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
                return clean_dataframe(df)
    except Exception as e:
        print(f"Fehler beim Laden von {label}: {e}")
        return pd.DataFrame()


def build_full_history_2006_to_past_year(
    base_dir: Path, current_year: int
) -> pd.DataFrame:
    """Loads the entire history from 2006 to the previous year once and saves it locally."""
    cache_file = base_dir / f"history_2006_{current_year - 1}.parquet"

    if cache_file.exists():
        print(f"Lese gecachte Historie 2006–{current_year - 1} aus Parquet...")
        return pd.read_parquet(cache_file)

    print(
        f"Creating full history 2006–{current_year - 1} fresh from the web..."
    )
    all_dfs = []

    # 1. 2006-2016 load Backfill archive (only use up to 2010-07-20)
    url_0616 = "https://www.cftc.gov/files/dea/history/fin_com_txt_2006_2016.zip"
    df_0616 = download_cftc_zip(url_0616, "2006-2016 Backfill")
    if not df_0616.empty:
        df_0616 = df_0616[df_0616["Date"] < pd.to_datetime("2010-07-20")].copy()
        all_dfs.append(df_0616)

    # 2. Year 2010 from July 20
    url_2010 = "https://www.cftc.gov/files/dea/history/com_fin_txt_2010.zip"
    df_2010 = download_cftc_zip(url_2010, "Jahr 2010")
    if not df_2010.empty:
        all_dfs.append(df_2010)

    # 3. All subsequent years up to the previous year (2011 to current_year - 1)
    for y in range(2011, current_year):
        url_year = f"https://www.cftc.gov/files/dea/history/com_fin_txt_{y}.zip"
        df_year = download_cftc_zip(url_year, f"Jahr {y}")
        if not df_year.empty:
            all_dfs.append(df_year)

    if not all_dfs:
        return pd.DataFrame()

    full_hist = pd.concat(all_dfs, ignore_index=True)
    full_hist.drop_duplicates(
        subset=["Date", "Market Name"], keep="last", inplace=True
    )
    full_hist.sort_values(
        by=["Market Name", "Date"], ascending=[True, True], inplace=True
    )
    full_hist.to_parquet(cache_file, index=False)
    print(f"History 2006–{current_year - 1} successfully saved locally.")
    return full_hist


# ==========================================
# 2. STATISTICS: DIFFERENCES & HP-FILTER
# ==========================================
def make_percent_difference(historical_df: pd.DataFrame) -> pd.DataFrame:
    differences = historical_df[["Date", "Market Name"]].copy()
    differences["Net Leverage"] = historical_df.groupby("Market Name")[
        "Net Leverage"
    ].diff()
    differences["Net Leverage Ratio"] = historical_df.groupby("Market Name")[
        "Net Leverage Ratio"
    ].diff()
    differences["Gross Short Ratio"] = historical_df.groupby("Market Name")[
        "Gross Short Ratio"
    ].diff()
    differences["Open Positions"] = historical_df.groupby("Market Name")[
        "Open Positions"
    ].pct_change()
    differences["Long Positions"] = historical_df.groupby("Market Name")[
        "Long Positions"
    ].pct_change()
    differences["Short Positions"] = historical_df.groupby("Market Name")[
        "Short Positions"
    ].pct_change()
    return differences


def make_hp_filtering(
    historical_df: pd.DataFrame, columns: list, lamb: float = 270400
) -> pd.DataFrame:
    hp_records = []
    market_names = historical_df["Market Name"].unique()

    for market_name in market_names:
        market_data = (
            historical_df[historical_df["Market Name"] == market_name]
            .sort_values("Date")
            .copy()
        )

        for variable in columns:
            series = market_data[variable].dropna()
            if len(series) < 10:
                continue

            cycle, trend = sm.tsa.filters.hpfilter(series, lamb=lamb)

            temp_res = pd.DataFrame(
                {
                    "Date": market_data.loc[series.index, "Date"],
                    "Market Name": market_name,
                    "Variable": variable,
                    "Value": series,
                    "Cycle": cycle,
                    "Trend": trend,
                }
            )
            hp_records.append(temp_res)

    if hp_records:
        return pd.concat(hp_records, ignore_index=True)
    return pd.DataFrame()


# ==========================================
# 3. MAIN PROCESS
# ==========================================
def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    current_year = datetime.datetime.now().year

    # 1. Load history from 2006 to the previous year (first time via web download, afterwards from cache)
    historical_df = build_full_history_2006_to_past_year(data_dir, current_year)

    # 2. Always download the current year fresh from the CFTC
    url_current = (
        f"https://www.cftc.gov/files/dea/history/com_fin_txt_{current_year}.zip"
    )
    current_df = download_cftc_zip(url_current, f"Current Year {current_year}")

    # Merge
    combined_df = pd.concat([historical_df, current_df], ignore_index=True)
    combined_df.drop_duplicates(
        subset=["Date", "Market Name"], keep="last", inplace=True
    )
    combined_df.sort_values(
        by=["Market Name", "Date"], ascending=[True, True], inplace=True
    )
    combined_df.reset_index(drop=True, inplace=True)

    print(
        f"Total dataset ready: {len(combined_df)} rows from "
        f"{combined_df['Date'].min().strftime('%Y-%m-%d')} bis {combined_df['Date'].max().strftime('%Y-%m-%d')}."
    )

    # 3. Calculate differences & HP filter over the entire 20-year history
    print("Berechne Veränderungen und HP-Filter über 20 Jahre Historie...")
    differences_df = make_percent_difference(combined_df)
    hp_filter_df = make_hp_filtering(combined_df, HP_VARIABLES, lamb=270400)

    # 4. Save as Parquet for Power BI
    combined_df.to_parquet(data_dir / "historical_data.parquet", index=False)
    differences_df.to_parquet(data_dir / "differences_data.parquet", index=False)
    hp_filter_df.to_parquet(data_dir / "hp_filter_data.parquet", index=False)

    print(
        f"\Successfully finished! Updated 3 Parquet files in:\n{data_dir}"
    )


if __name__ == "__main__":
    main()
