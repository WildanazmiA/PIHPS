from pathlib import Path
import matplotlib.pyplot as plt
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    commodities = list(df.columns)
    print("Komoditas:")
    for i, name in enumerate(commodities, 1):
        print(f"{i}. {name}")
    answer = input("Pilih nomor komoditas untuk grafik [1]: ").strip() or "1"
    try:
        commodity = commodities[int(answer) - 1]
    except (ValueError, IndexError):
        print("Pilihan tidak valid.")
        return

    series = df[commodity].dropna()
    if series.empty:
        print(f"Tidak ada data harga untuk {commodity}.")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="#247ba0", linewidth=1.7)
    ax.set_title(f"Tren harga {commodity} ({series.index.min():%Y-%m-%d} s.d. {series.index.max():%Y-%m-%d})")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp/{})".format("liter" if commodity.lower().startswith("minyak goreng") else "kg"))
    ax.set_xlim(series.index.min(), series.index.max())
    tampilkan_grafik(fig, "tren_harga_pangan.png")
