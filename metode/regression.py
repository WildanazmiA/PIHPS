from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    returns = df.sort_index().pct_change(fill_method=None)
    target = returns.mean(axis=1, skipna=True).shift(-1)
    data = returns.copy()
    data["target"] = target
    data = data.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=["target"])
    if len(data) < 100:
        print("Observasi valid kurang dari 100; evaluasi regression tidak dijalankan.")
        return
    X = data.drop(columns="target").fillna(0)
    y = data["target"]
    cut = int(len(data) * .8)
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=5)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    mae = mean_absolute_error(y.iloc[cut:], pred)
    print(f"Prediksi rata-rata return keranjang komoditas hari berikutnya — MAE kronologis 80/20: {mae:.6f}")
    print("Eksperimen historis; bukan prakiraan resmi PIHPS.")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y.index[cut:], y.iloc[cut:] * 100, label="Aktual", linewidth=1)
    ax.plot(y.index[cut:], pred * 100, label="Prediksi", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Regression perubahan harga pangan hari berikutnya (MAE {mae * 100:.3f}%)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Rata-rata perubahan harian (%)")
    ax.legend()
    tampilkan_grafik(fig, "regression_harga_pangan.png")
