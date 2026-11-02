from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _normalize_name(value):
    return " ".join(str(value).strip().split()).casefold()


def _resolve_single_series(df):
    """Fokus pada satu komoditas utama agar variabel pendukung lebih sempit."""
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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
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
        print("Observasi valid kurang dari 100; evaluasi regression tidak dijalankan.")
        return

    X = frame.drop(columns="target").fillna(0)
    y = frame["target"]
    cut = int(len(frame) * 0.8)
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=5)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    mae = mean_absolute_error(y.iloc[cut:], pred)
    print(f"Prediksi return harian berikutnya untuk {price.name or 'komoditas'} — MAE kronologis 80/20: {mae:.6f}")
    print("Variabel fokus: lag return, rolling mean, rolling std, momentum 7 hari, dan level harga sebelumnya.")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y.index[cut:], y.iloc[cut:] * 100, label="Aktual", linewidth=1)
    ax.plot(y.index[cut:], pred * 100, label="Prediksi", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Regression perubahan harga {price.name or 'komoditas'} hari berikutnya (MAE {mae * 100:.3f}%)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Perubahan harian (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ROOT / "hasil" / "regression_harga_komoditas_fokus.png")
    plt.close(fig)
