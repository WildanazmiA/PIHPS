from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "hasil" / "csv"


def analyze_returns(df):
    close = df.sort_index()
    daily = close.pct_change(fill_method=None)
    monthly = close.resample("ME").last().pct_change(fill_method=None)
    yearly = close.resample("YE").last().pct_change(fill_method=None)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(CSV_DIR / "return_harian_komoditas.csv", index_label="Date")
    monthly.to_csv(CSV_DIR / "return_bulanan_komoditas.csv", index_label="Date")
    yearly.to_csv(CSV_DIR / "return_tahunan_komoditas.csv", index_label="Date")
    print("Return harga harian, bulanan, dan tahunan per komoditas tersimpan di hasil/csv.")
    print("Return harian terbaru (%):\n", (daily.tail(5) * 100).round(3).to_string())


def analyze_volatility(df):
    returns = df.sort_index().pct_change(fill_method=None)
    daily_vol = returns.rolling(20, min_periods=10).std()
    annualized = daily_vol * (252 ** 0.5)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    daily_vol.to_csv(CSV_DIR / "volatilitas_20hari_komoditas.csv", index_label="Date")
    annualized.to_csv(CSV_DIR / "volatilitas_tahunan_komoditas.csv", index_label="Date")
    latest = annualized.dropna(how="all").tail(1).T.dropna()
    if not latest.empty:
        latest.columns = ["volatilitas_tahunan_terakhir"]
        print("Volatilitas tahunan terakhir (return, %):\n", (latest * 100).sort_values(latest.columns[0], ascending=False).round(2).to_string())
    print("Seri volatilitas tersimpan di hasil/csv.")
