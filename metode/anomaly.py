from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    from sklearn.ensemble import IsolationForest

    returns = df.sort_index().pct_change(fill_method=None)
    returns = returns.dropna(how="all")
    if len(returns) < 30:
        print("Observasi harian valid kurang dari 30; anomaly detection tidak dijalankan.")
        return
    # Nilai yang kosong diisi median masing-masing komoditas; hal ini menjaga
    # hari dengan beberapa laporan kosong tanpa membuat harga sintetis.
    model_data = returns.fillna(returns.median()).fillna(0)
    anomaly_flags = IsolationForest(contamination=.02, random_state=42).fit_predict(model_data) == -1
    result = returns.copy()
    result["anomaly"] = anomaly_flags
    out = ROOT / "hasil" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "anomaly_harga_pangan.csv", index_label="Date")
    anomaly_dates = result.index[result["anomaly"]]
    print(f"Hari dengan pola perubahan harga tidak biasa: {len(anomaly_dates)}.")
    print("Anomali statistik bukan bukti kesalahan atau manipulasi harga.")

    normalized = df.sort_index().div(df.sort_index().iloc[0]).mul(100)
    market_index = normalized.mean(axis=1, skipna=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(market_index.index, market_index, color="#5277b8", label="Indeks rata-rata harga (awal = 100)")
    ax.scatter(anomaly_dates, market_index.reindex(anomaly_dates), color="#d62728", s=30, label="Hari anomali", zorder=3)
    ax.set_title("Anomaly Detection perubahan harga pangan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Indeks harga")
    ax.legend()
    tampilkan_grafik(fig, "anomaly_harga_pangan.png")
