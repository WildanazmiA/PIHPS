"""Menu utama sistem data mining harga pangan PIHPS."""
from ambil_data import (
    MARKET_TYPES, default_start_date, download_data,
    load_data, load_provinces, process_data, raw_file, processed_file,
)
from metode import (tren, return_volatilitas, korelasi, clustering,
                    association_rule, anomaly, classification, regression)


def status_dataset(region="Nasional"):
    print(f"Sumber: Bank Indonesia — PIHPS | Wilayah: {region}")
    for market_type, market_name in MARKET_TYPES.items():
        raw_path = raw_file(market_type, region)
        processed_path = processed_file(market_type, region)
        print(f"\n{market_name}")
        if not raw_path.exists():
            print(f"  Belum diunduh: {raw_path}")
            continue
        df = load_data(market_type, region=region)
        if df is None or df.empty:
            print("  File tersedia, tetapi tidak memiliki observasi yang dapat dibaca.")
            continue
        print(f"  Mentah: {raw_path}")
        if region == "Nasional":
            print(f"  Processed: {processed_path} ({'tersedia' if processed_path.exists() else 'belum dibuat'})")
        else:
            print("  Dataset provinsi: satu CSV gabungan; analisis memfilter jenis pasar yang dipilih.")
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


def pilih_komoditas(df):
    """Pilih satu kolom komoditas, atau None untuk memakai semua komoditas."""
    commodities = list(df.columns)
    if not commodities:
        print("Dataset tidak memiliki kolom komoditas.")
        return None

    print("Komoditas yang tersedia untuk wilayah dan jenis pasar ini:")
    print("0. Semua komoditas")
    for number, commodity in enumerate(commodities, 1):
        print(f"{number}. {commodity}")

    answer = input("Pilih nomor komoditas [0 = semua]: ").strip() or "0"
    try:
        selection = int(answer)
    except ValueError:
        print("Pilihan harus berupa nomor.")
        return False

    if selection == 0:
        return None
    if 1 <= selection <= len(commodities):
        return commodities[selection - 1]
    print("Nomor komoditas tidak tersedia.")
    return False


def unduh_pasar(market_type, start=None, end=None, province_id="", region="Nasional"):
    start = start or default_start_date(market_type, region)
    downloaded = download_data(start=start, end=end, market_type=market_type,
                               province_id=province_id, region=region)
    if downloaded is not None:
        df = load_data(market_type, region=region)
        if df is not None:
            process_data(df, market_type, region=region)
    return downloaded


def main():
    selected_market_type = 1
    selected_province_id = ""
    selected_region = "Nasional"
    selected_commodity = None
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
        "11": ("Pilih provinsi untuk analisis", None),
        "12": ("Pilih komoditas untuk analisis", None),
    }
    while True:
        print(f"\n=== DATA MINING HARGA PANGAN PIHPS | {MARKET_TYPES[selected_market_type]} ===")
        print("D. Unduh/perbarui data PIHPS (semua pasar atau satu pasar)")
        print("P. Pilih jenis pasar untuk analisis")
        print(f"Provinsi aktif: {selected_region} (ID PIHPS: {selected_province_id or 'nasional'})")
        print(f"Komoditas aktif: {selected_commodity or 'Semua komoditas'}")
        for key, (label, _) in menu.items():
            print(f"{key}. {label}")
        print("0. Keluar")
        choice = input("Pilih menu: ").strip().lower()
        if choice == "0":
            break
        if choice == "p":
            try:
                selected_market_type = pilih_jenis_pasar()
                selected_commodity = None
            except ValueError as exc:
                print(exc)
            continue
        if choice == "11":
            try:
                provinces = load_provinces()
                print("Pilih wilayah:")
                print("0. Nasional")
                for number, province in enumerate(provinces, 1):
                    print(f"{number}. {province['name']} (ID PIHPS: {province['id']})")
                answer = input("Nomor provinsi [0 = Nasional]: ").strip() or "0"
                selection = int(answer)
                if selection == 0:
                    selected_region, selected_province_id = "Nasional", ""
                elif 1 <= selection <= len(provinces):
                    province = provinces[selection - 1]
                    selected_region = province["name"]
                    selected_province_id = str(province["id"])
                else:
                    print("Nomor provinsi tidak tersedia.")
                    continue
            except (ValueError, TypeError, KeyError) as exc:
                print(f"Pilihan provinsi tidak valid: {exc}")
                continue
            except Exception as exc:
                print(f"Gagal mengambil daftar provinsi PIHPS: {exc}")
                continue
            selected_commodity = None
            print(f"Wilayah analisis: {selected_region} (ID PIHPS: {selected_province_id or 'nasional'})")
            continue
        if choice == "12":
            df = load_data(selected_market_type, region=selected_region)
            if df is None or df.empty:
                continue
            selection = pilih_komoditas(df)
            if selection is False:
                continue
            selected_commodity = selection
            print(f"Komoditas analisis: {selected_commodity or 'Semua komoditas'}")
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
                start_override = input("Tanggal mulai [kosong = otomatis per file]: ").strip() or None
                end = input("Tanggal akhir inklusif [kosong = hari ini]: ").strip() or None
                downloaded_markets = []
                empty_markets = []
                for market_type in markets:
                    start = start_override or default_start_date(market_type, selected_region)
                    print(f"\n--- {MARKET_TYPES[market_type]} | {selected_region} | mulai {start} ---")
                    result = unduh_pasar(market_type, start, end, selected_province_id, selected_region)
                    (downloaded_markets if result is not None else empty_markets).append(MARKET_TYPES[market_type])
                if downloaded_markets:
                    print("\nData berhasil disimpan untuk: " + ", ".join(downloaded_markets) + ".")
                if empty_markets:
                    print("Tidak ada file baru untuk: " + ", ".join(empty_markets) + ".")
                if not downloaded_markets:
                    print("Unduhan tidak menghasilkan dataset. Pastikan ID provinsi benar dan rentang tanggal memiliki data PIHPS.")
                elif selected_region != "Nasional":
                    print(f"Dataset provinsi gabungan tersimpan di: {raw_file(markets[0], selected_region)}")
            except Exception as exc:
                print(f"Unduhan PIHPS gagal: {exc}")
            continue
        if choice not in menu:
            print("Pilihan tidak dikenal.")
            continue
        if choice == "10":
            status_dataset(selected_region)
            continue
        df = load_data(selected_market_type, region=selected_region)
        if df is None or df.empty:
            continue
        df = process_data(df, selected_market_type, region=selected_region)
        if selected_commodity is not None:
            if selected_commodity not in df.columns:
                print("Komoditas aktif tidak tersedia pada dataset ini. Pilih ulang menu 12.")
                selected_commodity = None
                continue
            if choice in {"4", "5", "6"}:
                requirements = {
                    "4": "Korelasi membutuhkan minimal dua komoditas.",
                    "5": "Clustering membutuhkan minimal tiga komoditas.",
                    "6": "Association Rule membandingkan kategori beberapa komoditas.",
                }
                print(requirements[choice] + " Pilih 'Semua komoditas' melalui menu 12.")
                continue
            df = df[[selected_commodity]]
        try:
            menu[choice][1](df)
        except ImportError as exc:
            print(f"Dependensi belum tersedia: {exc}. Pasang library dari requirements.txt.")
        except Exception as exc:
            print(f"Analisis gagal: {exc}")


if __name__ == "__main__":
    main()
