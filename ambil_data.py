"""Pengambilan data harga pangan PIHPS dari endpoint publik Bank Indonesia."""
from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BASE_URL = "https://www.bi.go.id/hargapangan"
REPORT_URL = f"{BASE_URL}/WebSite/TabelHarga/GetGridDataDaerah"
MARKET_TYPES = {
    1: "Pasar Tradisional",
    2: "Pasar Modern",
    3: "Pedagang Besar",
    4: "Produsen",
}
MARKET_FILES = {
    1: "pasar_tradisional.csv",
    2: "pasar_modern.csv",
    3: "pedagang_besar.csv",
    4: "produsen.csv",
}


def raw_file(market_type):
    return RAW_DIR / MARKET_FILES[market_type]


def processed_file(market_type):
    return PROCESSED_DIR / MARKET_FILES[market_type]


def default_start_date(market_type=1):
    """Tanggal usulan untuk pembaruan; ambil ulang 7 hari agar revisi tercakup."""
    path = raw_file(market_type)
    if path.exists():
        try:
            old = pd.read_csv(path, usecols=["Date", "MarketType"], parse_dates=["Date"])
            market_names = set(old["MarketType"].dropna().astype(str))
            if market_names == {MARKET_TYPES[market_type]} and not old.empty:
                return max(date(2020, 1, 1), old["Date"].max().date() - timedelta(days=7)).isoformat()
        except (ValueError, OSError, pd.errors.ParserError):
            pass
    return "2020-01-01"


def _session():
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/TabelHarga/PasarTradisionalDaerah",
    })
    return session


def _rows_from_report(payload, market_name):
    rows = []
    date_key = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    for item in payload.get("data", []):
        # level 1 adalah judul kategori (mis. Beras), bukan komoditas harga.
        if int(item.get("level", 2)) != 2:
            continue
        commodity = str(item.get("name", "")).strip()
        for key, value in item.items():
            if not date_key.match(str(key)) or value in (None, "", "-"):
                continue
            try:
                price = float(str(value).replace(",", "").strip())
                observed = pd.to_datetime(key, format="%d/%m/%Y")
            except (TypeError, ValueError):
                continue
            if commodity and pd.notna(price):
                unit = "liter" if commodity.lower().startswith("minyak goreng") else "kg"
                rows.append({
                    "Date": observed,
                    "Commodity": commodity,
                    "Price": price,
                    "Unit": unit,
                    "MarketType": market_name,
                    "Region": "Nasional",
                })
    return rows


def download_data(start="2020-01-01", end=None, market_type=1):
    """Unduh laporan harga harian per tahun; end bersifat inklusif."""
    if market_type not in MARKET_TYPES:
        raise ValueError("Jenis pasar harus 1, 2, 3, atau 4.")
    start_date = date.fromisoformat(str(start))
    end_date = date.fromisoformat(str(end)) if end else date.today()
    if start_date > end_date:
        raise ValueError("Tanggal mulai tidak boleh sesudah tanggal akhir.")

    session = _session()
    records = []
    cursor = start_date
    while cursor <= end_date:
        # Laporan harian setahun berukuran kecil tetapi endpoint PIHPS sangat
        # lambat untuk rentang penuh; kuartal memberi ukuran respons sedang.
        chunk_end = min(cursor + timedelta(days=89), end_date)
        params = {
            "price_type_id": market_type,
            "comcat_id": "",
            "province_id": "",
            "regency_id": "",
            "market_id": "",
            "tipe_laporan": 1,
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
            "skip": 0,
            "take": 100,
            "requireTotalCount": "true",
        }
        response = session.get(REPORT_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"PIHPS mengembalikan kesalahan: {payload['errors']}")
        records.extend(_rows_from_report(payload, MARKET_TYPES[market_type]))
        print(f"Mengambil data {cursor} s.d. {chunk_end} ...")
        cursor = chunk_end + timedelta(days=1)

    if not records:
        print("PIHPS tidak mengembalikan observasi pada rentang tersebut; file lama tidak diubah.")
        return None
    data = pd.DataFrame.from_records(records)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop_duplicates(["Date", "Commodity", "MarketType", "Region"], keep="last")
    data = data.sort_values(["Date", "Commodity"]).reset_index(drop=True)

    # Untuk pasar yang sama, gabungkan unduhan baru dengan histori lama.
    # Jika pengguna memilih jenis pasar berbeda, mulai dataset baru agar data
    # dua segmen pasar tidak tercampur saat dianalisis.
    path = raw_file(market_type)
    if path.exists():
        try:
            old = pd.read_csv(path, parse_dates=["Date"])
            old_market = set(old.get("MarketType", pd.Series(dtype=str)).dropna().astype(str))
            if old_market == {MARKET_TYPES[market_type]}:
                data = pd.concat([old, data], ignore_index=True)
                data = data.drop_duplicates(["Date", "Commodity", "MarketType", "Region"], keep="last")
                data = data.sort_values(["Date", "Commodity"]).reset_index(drop=True)
        except (ValueError, OSError, pd.errors.ParserError):
            pass

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(
        f"Data PIHPS ({MARKET_TYPES[market_type]}, rata-rata nasional) tersimpan: {path}\n"
        f"Observasi: {len(data)} | Komoditas: {data['Commodity'].nunique()} | "
        f"Rentang aktual: {data['Date'].min().date()} s.d. {data['Date'].max().date()}"
    )
    return data


def load_data(market_type=1, path=None):
    """Baca data mentah PIHPS dan susun menjadi tanggal × komoditas."""
    path = Path(path) if path else raw_file(market_type)
    if not path.exists():
        print(f"Dataset PIHPS belum ditemukan: {path}. Gunakan menu D untuk mengunduh.")
        return None
    raw = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Commodity", "Price"}
    if not required.issubset(raw.columns):
        print(f"Format file bukan data PIHPS yang dikenali: {path}")
        return None
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Price", "Commodity"])
    wide = raw.pivot_table(index="Date", columns="Commodity", values="Price", aggfunc="mean")
    wide.columns.name = None
    wide.index.name = "Date"
    return wide.sort_index()


def process_data(df, market_type=1):
    """Simpan salinan rapi tanggal × komoditas ke folder processed."""
    path = processed_file(market_type)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean = df.copy().sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    clean.to_csv(path, index_label="Date")
    print(f"Data bersih PIHPS diperbarui: {path} ({len(clean)} tanggal)")
    return clean
