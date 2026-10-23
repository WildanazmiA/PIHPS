from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    returns = df.sort_index().pct_change(fill_method=None)
    # Indeks keranjang setara memberi sinyal lebih stabil daripada median,
    # yang kerap nol ketika mayoritas harga komoditas tidak berubah hari itu.
    target = returns.mean(axis=1, skipna=True).shift(-1)
    data = returns.copy()
    data["target"] = (target > 0).astype("float")
    data.loc[target.isna(), "target"] = float("nan")
    data = data.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=["target"])
    if len(data) < 100:
        print("Observasi valid kurang dari 100; evaluasi classification tidak dijalankan.")
        return
    X = data.drop(columns="target").fillna(0)
    y = data["target"].astype(int)
    cut = int(len(data) * .8)
    if y.iloc[:cut].nunique() < 2:
        print("Data latih hanya memiliki satu kategori arah harga; model tidak dapat dilatih.")
        return
    model = RandomForestClassifier(n_estimators=250, random_state=42, min_samples_leaf=5, class_weight="balanced")
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    accuracy = accuracy_score(y.iloc[cut:], pred)
    balanced_accuracy = balanced_accuracy_score(y.iloc[cut:], pred)
    baseline = max(y.iloc[cut:].mean(), 1 - y.iloc[cut:].mean())
    print(f"Prediksi arah rata-rata return keranjang komoditas hari berikutnya — akurasi: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f} | Baseline kelas mayoritas: {baseline:.3f}")
    print("Kategori: tidak naik atau naik. Evaluasi kronologis 80/20; eksperimen historis, bukan rekomendasi perdagangan.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y.iloc[cut:], pred, labels=[0, 1]),
        display_labels=["Turun/tetap", "Naik"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Classification arah perubahan harga pangan")
    tampilkan_grafik(fig, "classification_harga_pangan.png")
