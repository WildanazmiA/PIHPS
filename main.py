"""Menu utama sistem data mining harga pangan PIHPS."""
from ambil_data import (
    MARKET_TYPES, default_start_date, download_data,
    load_data, process_data, raw_file, processed_file,
)
from metode import (tren, return_volatilitas, korelasi, clustering,
                    association_rule, anomaly, classification, regression)


def status_dataset(_=None):
    print("Sumber: Bank Indonesia — PIHPS, harga rata-rata nasional")
    for market_type, market_name in MARKET_TYPES.items():
        raw_path = raw_file(market_type)
        processed_path = processed_file(market_type)
        print(f"\n{market_name}")
        if not raw_path.exists():
            print(f"  Belum diunduh: {raw_path}")
            continue
        df = load_data(market_type)
        if df is None or df.empty:
            print("  File tersedia, tetapi tidak memiliki observasi yang dapat dibaca.")
            continue
        print(f"  Mentah: {raw_path}")
        print(f"  Processed: {processed_path} ({'tersedia' if processed_path.exists() else 'belum dibuat'})")
        print(f"  Rentang: {df.index.min().date()} — {df.index.max().date()}")
        print(f"  Tanggal observasi: {len(df)} | Komoditas: {len(df.columns)}")


def pilih_jenis_pasar(prompt="Pilih jenis pasar [1]: "):
    print("Jenis pasar PIHPS:")
    for key, name in MARKET_TYPES.items():
        print(f"{key}. {name}")
    choice = input(prompt).strip() or "1"
    try:
        market_type = int(choice)
    except ValueError as exc:
        raise ValueError("Pilihan jenis pasar harus angka 1 sampai 4.") from exc
    if market_type not in MARKET_TYPES:
        raise ValueError("Pilihan jenis pasar harus angka 1 sampai 4.")
    return market_type


def unduh_pasar(market_type, start=None, end=None):
    start = start or default_start_date(market_type)
    downloaded = download_data(start=start, end=end, market_type=market_type)
    if downloaded is not None:
        df = load_data(market_type)
        if df is not None:
            process_data(df, market_type)
    return downloaded


def main():
    selected_market_type = 1
    menu = {
        "1": ("Grafik tren harga komoditas", tren.run),
        "2": ("Analisis return harga", return_volatilitas.analyze_returns),
        "3": ("Analisis volatilitas harga", return_volatilitas.analyze_volatility),
        "4": ("Korelasi perubahan harga komoditas", korelasi.run),
        "5": ("Clustering komoditas", clustering.run),
        "6": ("Association Rule Mining", association_rule.run),
        "7": ("Anomaly Detection", anomaly.run),
        "8": ("Classification perubahan harga", classification.run),
        "9": ("Regression perubahan harga", regression.run),
        "10": ("Lihat status dataset semua pasar", status_dataset),
    }
    while True:
        print(f"\n=== DATA MINING HARGA PANGAN PIHPS | {MARKET_TYPES[selected_market_type]} ===")
        print("D. Unduh/perbarui data PIHPS (semua pasar atau satu pasar)")
        print("P. Pilih jenis pasar untuk analisis")
        for key, (label, _) in menu.items():
            print(f"{key}. {label}")
        print("0. Keluar")
        choice = input("Pilih menu: ").strip().lower()
        if choice == "0":
            break
        if choice == "p":
            try:
                selected_market_type = pilih_jenis_pasar()
            except ValueError as exc:
                print(exc)
            continue
        if choice == "d":
            try:
                market_choice = input("Unduh semua pasar [A] atau satu pasar [1-4] [A]: ").strip().lower() or "a"
                if market_choice == "a":
                    markets = list(MARKET_TYPES)
                else:
                    try:
                        markets = [int(market_choice)]
                    except ValueError as exc:
                        raise ValueError("Masukkan A atau angka 1 sampai 4.") from exc
                    if markets[0] not in MARKET_TYPES:
                        raise ValueError("Pilihan jenis pasar harus angka 1 sampai 4.")
                start_override = input("Tanggal mulai untuk semua pasar [kosong = otomatis per file]: ").strip() or None
                end = input("Tanggal akhir inklusif [kosong = hari ini]: ").strip() or None
                for market_type in markets:
                    start = start_override or default_start_date(market_type)
                    print(f"\n--- {MARKET_TYPES[market_type]} | mulai {start} ---")
                    unduh_pasar(market_type, start, end)
                print("\nUnduhan selesai. Data mentah dan processed tersimpan per jenis pasar.")
            except Exception as exc:
                print(f"Unduhan PIHPS gagal: {exc}")
            continue
        if choice not in menu:
            print("Pilihan tidak dikenal.")
            continue
        if choice == "10":
            status_dataset()
            continue
        df = load_data(selected_market_type)
        if df is None:
            continue
        df = process_data(df, selected_market_type)
        try:
            menu[choice][1](df)
        except ImportError as exc:
            print(f"Dependensi belum tersedia: {exc}. Pasang library dari requirements.txt.")
        except Exception as exc:
            print(f"Analisis gagal: {exc}")


if __name__ == "__main__":
    main()
