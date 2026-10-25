"""Latih Random Forest untuk satu komoditas dari dataset PIHPS proyek."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ambil_data import MARKET_TYPES, raw_file


DEFAULT_REGION = "Kalimantan Timur"
DEFAULT_MARKET_TYPE = 1


def _pilih_pasar():
    print("Jenis pasar:")
    for key, name in MARKET_TYPES.items():
        print(f"{key}. {name}")
    answer = input(f"Pilih jenis pasar [{DEFAULT_MARKET_TYPE}]: ").strip()
    market_type = int(answer) if answer else DEFAULT_MARKET_TYPE
    if market_type not in MARKET_TYPES:
        raise ValueError("Pilihan jenis pasar harus angka 1 sampai 4.")
    return market_type


def _baca_seri(region, market_type):
    path = raw_file(market_type, region)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset belum ditemukan: {path}. Pilih provinsi lewat menu 11 di main.py, "
            "lalu unduh data lewat menu D."
        )

    df = pd.read_csv(path)
    required = {"Date", "Commodity", "Price", "MarketType", "Region"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Kolom dataset tidak lengkap: {', '.join(sorted(missing))}. "
            "Gunakan CSV PIHPS versi terbaru."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    region_key = region.strip().casefold()
    market_name = MARKET_TYPES[market_type]
    df = df[
        df["Region"].astype(str).str.strip().str.casefold().eq(region_key)
        & df["MarketType"].astype(str).str.strip().str.casefold().eq(market_name.casefold())
    ].dropna(subset=["Date", "Price", "Commodity"])

    if df.empty:
        raise ValueError(
            f"Tidak ada data untuk {region} / {market_name} di {path}. "
            "Periksa isi CSV atau unduh jenis pasar tersebut lewat menu D."
        )

    commodities = sorted(df["Commodity"].astype(str).str.strip().unique())
    print(f"\nKomoditas tersedia untuk {region} — {market_name}:")
    for number, commodity in enumerate(commodities, 1):
        print(f"{number}. {commodity}")
    selection = input("Pilih nomor komoditas: ").strip()
    try:
        commodity = commodities[int(selection) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Nomor komoditas tidak valid.") from exc

    series = (
        df[df["Commodity"].astype(str).str.strip().eq(commodity)]
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )
    return path, commodity, series


def main():
    region = input(f"Nama provinsi [{DEFAULT_REGION}]: ").strip() or DEFAULT_REGION
    market_type = _pilih_pasar()
    path, commodity, df = _baca_seri(region, market_type)

    # Lag dihitung berdasarkan catatan yang tersedia; target adalah catatan
    # harga berikutnya, sehingga jaraknya bisa lebih dari satu hari kalender.
    df["price_lag_1"] = df["Price"].shift(1)
    df["price_lag_7"] = df["Price"].shift(7)
    df["price_lag_14"] = df["Price"].shift(14)
    df["target_price_next_observation"] = df["Price"].shift(-1)
    df["target_date"] = df["Date"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Data terlalu sedikit setelah membuat fitur lag. "
            "Unduh rentang tanggal yang lebih panjang."
        )

    features = ["price_lag_1", "price_lag_7", "price_lag_14"]
    X = df[features]
    y = df["target_price_next_observation"]

    # Split kronologis agar data masa depan tidak bocor ke data latihan.
    split = int(len(df) * 0.8)
    if split == 0 or split == len(df):
        raise ValueError("Jumlah observasi tidak cukup untuk split latihan dan pengujian.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"\nDataset: {Path(path).resolve()}")
    print(f"Wilayah: {region} | Jenis pasar: {MARKET_TYPES[market_type]}")
    print(f"Komoditas: {commodity}")
    print(f"Jumlah observasi: {len(df)} | Latih: {len(X_train)} | Uji: {len(X_test)}")
    print(f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f}")
    print("\nContoh prediksi pada data uji:")
    print(
        pd.DataFrame(
            {
                "target_date": df["target_date"].iloc[split:].to_numpy(),
                "actual_price": y_test.to_numpy(),
                "predicted_price": predictions,
            }
        ).head(10).to_string(index=False)
    )
    print("\nEvaluasi historis kronologis; hasil model bukan prakiraan resmi PIHPS.")


if __name__ == "__main__":
    main()
