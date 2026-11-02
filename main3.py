"""Forecasting harga pangan PIHPS dengan pilihan wilayah yang dinamis.

Jalankan: python main3.py
Data yang digunakan harus sudah tersedia di data/raw (gunakan main.py menu D
untuk mengunduh atau memperbarui data PIHPS).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ambil_data import MARKET_TYPES, RAW_DIR, load_provinces, raw_file


HORIZONS = {"1": 1, "7": 7, "30": 30}
MIN_OBSERVATIONS = 45
FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    "mean_7", "mean_14", "mean_30", "day_of_week", "day_of_year", "month",
]


def _available_regions():
    """Cari wilayah berdasarkan file PIHPS yang tersedia secara lokal."""
    regions = {"Nasional"}
    for path in RAW_DIR.glob("*_data.csv"):
        try:
            frame = pd.read_csv(path, usecols=["Region"])
            regions.update(frame["Region"].dropna().astype(str).str.strip())
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    return sorted(regions, key=lambda region: (region != "Nasional", region.casefold()))


def _select_region():
    regions = _available_regions()
    try:
        official = load_provinces()
        # Tambahkan provinsi PIHPS resmi; data yang belum diunduh akan
        # dijelaskan setelah pengguna memilih komoditas/pasar.
        regions = sorted(set(regions).union(
            str(item.get("name", "")).strip() for item in official
            if item.get("name")
        ), key=lambda region: (region != "Nasional", region.casefold()))
    except Exception as exc:
        print(f"Daftar provinsi online tidak tersedia ({exc}); memakai wilayah lokal.")

    print("\nPilih wilayah:")
    for index, region in enumerate(regions, 1):
        print(f"{index}. {region}")
    answer = input("Nomor wilayah: ").strip()
    try:
        return regions[int(answer) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Nomor wilayah tidak valid.") from exc


def _read_series(region, market_type, commodity):
    path = raw_file(market_type, region)
    if not path.exists():
        raise FileNotFoundError(
            f"Data untuk {region} belum ada ({path}). Unduh dahulu lewat menu D di main.py."
        )
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"File tidak dapat dibaca: {path}") from exc

    required = {"Date", "Commodity", "Price", "MarketType", "Region"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Kolom wajib tidak lengkap pada {path}: {sorted(required - set(raw.columns))}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    selected = raw[
        raw["Region"].astype(str).str.strip().str.casefold().eq(region.casefold())
        & raw["MarketType"].astype(str).str.strip().str.casefold().eq(
            MARKET_TYPES[market_type].casefold()
        )
        & raw["Commodity"].astype(str).str.strip().eq(commodity)
    ].dropna(subset=["Date", "Price"])
    if selected.empty:
        raise ValueError(f"Tidak ada data {commodity} untuk {region} / {MARKET_TYPES[market_type]}.")

    # Satu harga per hari; indeks kalender harian membuat horizon benar-benar
    # berarti 1/7/30 hari, meskipun PIHPS tidak melaporkan harga setiap hari.
    series = selected.groupby("Date")["Price"].mean().sort_index()
    series = series.asfreq("D")
    # Harga di antara dua laporan PIHPS diisi interpolasi waktu agar lag 7
    # selalu berarti tujuh hari kalender, bukan tujuh baris laporan.
    series = series.interpolate(method="time", limit_area="inside")
    return series.dropna(), path


def _make_features(series):
    frame = pd.DataFrame({"price": series})
    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"lag_{lag}"] = frame["price"].shift(lag)
    for window in (7, 14, 30):
        frame[f"mean_{window}"] = frame["price"].shift(1).rolling(window).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["day_of_year"] = frame.index.dayofyear
    frame["month"] = frame.index.month
    return frame.dropna()


def _features_for_date(history, date):
    row = {f"lag_{lag}": history.iloc[-lag] for lag in (1, 2, 3, 7, 14, 30)}
    for window in (7, 14, 30):
        row[f"mean_{window}"] = history.iloc[-window:].mean()
    row.update(day_of_week=date.dayofweek, day_of_year=date.dayofyear, month=date.month)
    return pd.DataFrame([row], columns=FEATURES)


def _select_number(prompt, options):
    for key, label in options.items():
        print(f"{key}. {label}")
    value = input(prompt).strip()
    if value not in options:
        raise ValueError("Pilihan tidak tersedia.")
    return value


def main():
    print("=== Forecasting Harga Pangan PIHPS (Random Forest) ===")
    region = _select_region()
    market_key = _select_number("Pilih jenis pasar: ", {
        str(key): value for key, value in MARKET_TYPES.items()
    })
    market_type = int(market_key)

    path = raw_file(market_type, region)
    if not path.exists():
        raise FileNotFoundError(
            f"Data untuk {region} belum diunduh ({path}). Jalankan main.py, pilih wilayah, lalu menu D."
        )
    raw = pd.read_csv(path, usecols=["Date", "Commodity", "MarketType", "Region"])
    commodities = sorted(raw.loc[
        raw["Region"].astype(str).str.strip().str.casefold().eq(region.casefold())
        & raw["MarketType"].astype(str).str.strip().str.casefold().eq(MARKET_TYPES[market_type].casefold()),
        "Commodity",
    ].dropna().astype(str).str.strip().unique())
    if not commodities:
        raise ValueError(f"Tidak ada komoditas untuk {region} / {MARKET_TYPES[market_type]}.")
    print("\nPilih komoditas:")
    for index, name in enumerate(commodities, 1):
        print(f"{index}. {name}")
    try:
        commodity = commodities[int(input("Nomor komoditas: ").strip()) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Nomor komoditas tidak valid.") from exc

    horizon = HORIZONS[_select_number("\nPilih horizon prediksi (hari): ", {
        "1": "1 hari", "7": "7 hari", "30": "30 hari",
    })]
    series, data_path = _read_series(region, market_type, commodity)
    supervised = _make_features(series)
    if len(supervised) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Data efektif hanya {len(supervised)} baris; minimal {MIN_OBSERVATIONS} diperlukan. "
            "Unduh rentang historis yang lebih panjang atau pilih komoditas lain."
        )

    # Evaluasi rolling-origin: setiap prediksi uji hanya memakai observasi
    # yang tersedia sebelum tanggal targetnya.
    test_size = max(horizon, int(len(supervised) * 0.2))
    test_size = min(test_size, len(supervised) - MIN_OBSERVATIONS)
    split = len(supervised) - test_size
    train = supervised.iloc[:split]
    test = supervised.iloc[split:]
    evaluation_model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.9,
        random_state=42, n_jobs=-1,
    )
    evaluation_model.fit(train[FEATURES], train["price"])
    evaluated = []
    for date, row in test.iterrows():
        available = series.loc[:date - pd.Timedelta(days=1)]
        feature = _features_for_date(available, date)
        evaluated.append(float(evaluation_model.predict(feature)[0]))
    actual = test["price"].to_numpy()
    mae = mean_absolute_error(actual, evaluated)
    rmse = mean_squared_error(actual, evaluated) ** 0.5

    # Model akhir memakai seluruh histori dan meramal secara rekursif:
    # hasil satu hari menjadi fitur lag untuk hari selanjutnya.
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.9,
        random_state=42, n_jobs=-1,
    )
    model.fit(supervised[FEATURES], supervised["price"])
    history = series.copy()
    future_dates, forecasts = [], []
    for _ in range(horizon):
        date = history.index[-1] + pd.Timedelta(days=1)
        prediction = max(0.0, float(model.predict(_features_for_date(history, date))[0]))
        future_dates.append(date)
        forecasts.append(prediction)
        history.loc[date] = prediction

    output = pd.DataFrame({"Date": future_dates, "PredictedPrice": forecasts})
    output_dir = Path(__file__).resolve().parent / "hasil" / "forecasting"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_region = "_".join(region.lower().split())
    safe_commodity = "_".join(commodity.lower().split())
    stem = f"{safe_region}_{safe_commodity}_{market_type}_{horizon}hari"
    csv_path = output_dir / f"{stem}.csv"
    image_path = output_dir / f"{stem}.png"
    output.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    recent = series.tail(60)
    plt.figure(figsize=(11, 5))
    plt.plot(recent.index, recent.values, label="Harga aktual", color="#2563eb")
    plt.plot(output["Date"], output["PredictedPrice"], label="Forecast Random Forest",
            color="#dc2626", marker="o")
    plt.title(f"Forecast harga {commodity} — {region}\n{MARKET_TYPES[market_type]}, {horizon} hari")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp per unit PIHPS)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    print("\n=== Ringkasan hasil ===")
    print(f"Wilayah: {region} | Pasar: {MARKET_TYPES[market_type]} | Komoditas: {commodity}")
    print(f"Data: {data_path} | Rentang: {series.index.min().date()} — {series.index.max().date()}")
    print(f"Observasi harian efektif: {len(series)} | Baris model: {len(supervised)}")
    print(f"Evaluasi kronologis (MAE): Rp{mae:,.2f} | RMSE: Rp{rmse:,.2f}")
    print("Harga prediksi:")
    print(output.to_string(index=False, formatters={"PredictedPrice": "{:,.2f}".format}))
    print(f"\nCSV: {csv_path}")
    print(f"Grafik: {image_path}")
    print("Catatan: ini estimasi model dari data historis, bukan harga resmi atau jaminan PIHPS.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"\nTidak dapat membuat forecast: {exc}")
