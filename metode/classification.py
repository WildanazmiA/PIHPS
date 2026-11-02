from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _normalize_name(value):
    return " ".join(str(value).strip().split()).casefold()


def _resolve_single_series(df):
    """Pilih sprint fokus ke satu komoditas agar variabel lebih spesifik."""
    if df is None or df.empty:
        return None
    if isinstance(df, pd.Series):
        return df.sort_index()
    if len(df.columns) == 1:
        return df.iloc[:, 0].sort_index()

    candidates = []
    for commodity in df.columns:
        name = _normalize_name(commodity)
        if "beras" in name and "bawah" in name and ("1" in name or "i" in name):
            candidates.append(commodity)
    if candidates:
        return df[candidates[0]].sort_index()
    return df.iloc[:, 0].sort_index()


def run(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    series = _resolve_single_series(df)
    if series is None or series.empty:
        print("Dataset harga tidak memiliki seri komoditas yang valid untuk analisis.")
        return

    price = series.sort_index()
    returns = price.pct_change(fill_method=None)
    frame = pd.DataFrame(index=returns.index)
    for lag in [1, 2, 3, 7, 14]:
        frame[f"return_lag_{lag}"] = returns.shift(lag)
    frame["rolling_mean_3"] = returns.rolling(3, min_periods=1).mean()
    frame["rolling_mean_7"] = returns.rolling(7, min_periods=1).mean()
    frame["rolling_std_7"] = returns.rolling(7, min_periods=2).std()
    frame["price_lag_1"] = price.shift(1)
    frame["price_lag_7"] = price.shift(7)
    frame["momentum_7"] = (price / price.shift(7) - 1)
    frame["target"] = returns.shift(-1)
    frame = frame.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=["target"])

    if len(frame) < 100:
        print("Observasi valid kurang dari 100; evaluasi classification tidak dijalankan.")
        return

    X = frame.drop(columns="target").fillna(0)
    y = (frame["target"] > 0).astype(int)
    cut = int(len(frame) * 0.8)
    if y.iloc[:cut].nunique() < 2:
        print("Data latih hanya memiliki satu kategori arah harga; model tidak dapat dilatih.")
        return

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=5,
        class_weight="balanced",
    )
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    accuracy = accuracy_score(y.iloc[cut:], pred)
    balanced_accuracy = balanced_accuracy_score(y.iloc[cut:], pred)
    baseline = max(y.iloc[cut:].mean(), 1 - y.iloc[cut:].mean())
    print(f"Prediksi arah harga {price.name or 'komoditas'} berikutnya — akurasi: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f} | Baseline kelas mayoritas: {baseline:.3f}")
    print("Variabel fokus: return lag, rolling mean, rolling std, momentum 7 hari, dan level harga sebelumnya.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y.iloc[cut:], pred, labels=[0, 1]),
        display_labels=["Turun/tetap", "Naik"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Classification arah perubahan harga {price.name or 'komoditas'}")
    plt.tight_layout()
    fig.savefig(ROOT / "hasil" / "classification_harga_komoditas_fokus.png")
    plt.close(fig)
