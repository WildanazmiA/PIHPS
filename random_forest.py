"""Contoh terpisah melatih Random Forest untuk harga satu komoditas PIHPS.

Script ini membaca data PIHPS yang sudah ada. Script tidak mengunduh,
mengubah, atau menyimpan ulang file CSV.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ambil_data import MARKET_TYPES, load_data


def main():
    print("=== Contoh Random Forest: prediksi harga komoditas berikutnya ===")
    print("Pilih jenis pasar:")
    for market_id, market_name in MARKET_TYPES.items():
        print(f"{market_id}. {market_name}")

    try:
        market_type = int(input("Nomor jenis pasar [1]: ").strip() or "1")
    except ValueError:
        print("Pilihan pasar harus berupa angka 1 sampai 4.")
        return
    if market_type not in MARKET_TYPES:
        print("Pilihan pasar harus berupa angka 1 sampai 4.")
        return

    region = input("Wilayah data [Nasional]: ").strip() or "Nasional"
    df = load_data(market_type, region=region)
    if df is None or df.empty:
        print("Data tidak ditemukan atau kosong. Unduh data PIHPS terlebih dahulu.")
        return

    commodities = list(df.columns)
    print("\nPilih komoditas:")
    for number, name in enumerate(commodities, 1):
        print(f"{number}. {name}")

    try:
        commodity_number = int(input("Nomor komoditas: ").strip())
    except ValueError:
        print("Pilihan komoditas harus berupa nomor.")
        return
    if not 1 <= commodity_number <= len(commodities):
        print("Nomor komoditas tidak tersedia.")
        return
    commodity = commodities[commodity_number - 1]

    # Satu baris adalah satu tanggal pengamatan. Drop nilai kosong untuk
    # komoditas terpilih, lalu urutkan dari tanggal lama ke terbaru.
    prices = pd.to_numeric(df[commodity], errors="coerce").dropna().sort_index()

    # Fitur hanya memakai harga yang sudah diketahui. Target adalah harga
    # pada catatan berikutnya (belum tentu hari kalender berikutnya).
    data = pd.DataFrame(index=prices.index)
    data["harga_1_catatan_lalu"] = prices.shift(1)
    data["harga_2_catatan_lalu"] = prices.shift(2)
    data["harga_7_catatan_lalu"] = prices.shift(7)
    data["target_harga_catatan_berikutnya"] = prices.shift(-1)
    data = data.dropna()

    if len(data) < 20:
        print(
            f"Hanya ada {len(data)} baris latihan/pengujian setelah menyiapkan fitur. "
            "Gunakan data historis yang lebih panjang."
        )
        return

    feature_columns = [
        "harga_1_catatan_lalu",
        "harga_2_catatan_lalu",
        "harga_7_catatan_lalu",
    ]
    X = data[feature_columns]
    y = data["target_harga_catatan_berikutnya"]

    # Pemisahan kronologis: data lama untuk melatih, data terbaru untuk menguji.
    split_at = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    # Inilah bagian model Random Forest dilatih dari fitur (X) dan target (y).
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    # Uji pada bagian data terbaru yang tidak ikut digunakan saat pelatihan.
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"\nPasar: {MARKET_TYPES[market_type]}")
    print(f"Wilayah: {region}")
    print(f"Komoditas: {commodity}")
    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("\nPerbandingan harga aktual dan prediksi:")
    comparison = pd.DataFrame(
        {
            "tanggal_fitur": X_test.index,
            "harga_aktual_catatan_berikutnya": y_test.to_numpy(),
            "harga_prediksi": predictions,
        }
    )
    print(comparison.head(10).to_string(index=False))
    print("\nMAE/RMSE lebih kecil berarti kesalahan prediksi pada data uji lebih kecil.")
    print("Ini contoh pembelajaran historis, bukan jaminan harga berikutnya.")


if __name__ == "__main__":
    main()
