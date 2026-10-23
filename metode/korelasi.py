from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    returns = df.sort_index().pct_change(fill_method=None).dropna(how="all")
    if returns.shape[1] < 2:
        print("Korelasi membutuhkan data minimal dua komoditas.")
        return
    corr = returns.corr(min_periods=20)
    out = ROOT / "hasil" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out / "korelasi_komoditas.csv")
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * .55), max(6, len(corr) * .45)))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Korelasi return harian antar komoditas PIHPS")
    tampilkan_grafik(fig, "korelasi_komoditas.png")
    print(f"Matriks korelasi tersimpan: {out / 'korelasi_komoditas.csv'}")
