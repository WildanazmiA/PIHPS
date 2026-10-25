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
PROVINCE_URL = f"{BASE_URL}/WebSite/TabelHarga/GetRefProvince"
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


def _region_slug(region):
    slug = re.sub(r"[^a-z0-9]+", "_", str(region).strip().lower()).strip("_")
    if not slug:
        raise ValueError("Nama provinsi tidak boleh kosong.")
    return slug


def raw_file(market_type, region="Nasional"):
    filename = MARKET_FILES[market_type]
    if region == "Nasional":
        return RAW_DIR / filename
    return RAW_DIR / f"{_region_slug(region)}_data.csv"


def processed_file(market_type, region="Nasional"):
    filename = MARKET_FILES[market_type]
    if region == "Nasional":
        return PROCESSED_DIR / filename
    return PROCESSED_DIR / f"{Path(filename).stem}_{_region_slug(region)}.csv"


def default_start_date(market_type=1, region="Nasional"):
    """Tanggal usulan untuk pembaruan; ambil ulang 7 hari agar revisi tercakup."""
    path = raw_file(market_type, region)
    if path.exists():
        try:
            old = pd.read_csv(path, usecols=["Date", "MarketType"], parse_dates=["Date"])
            old = old[old["MarketType"].astype(str) == MARKET_TYPES[market_type]]
            if not old.empty:
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


def load_provinces():
    """Ambil nama dan ID provinsi resmi yang digunakan oleh PIHPS."""
    response = _session().get(
        PROVINCE_URL,
        params={"skip": 0, "take": 100, "requireTotalCount": "true"},
        timeout=30,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        snippet = re.sub(r"\s+", " ", response.text).strip()[:300]
        raise RuntimeError(f"Daftar provinsi PIHPS bukan JSON yang valid: {snippet or '[kosong]'}") from exc
    provinces = payload.get("data")
    if not isinstance(provinces, list) or not provinces:
        raise RuntimeError("PIHPS tidak memberikan daftar provinsi yang dapat digunakan.")
    return provinces


def _rows_from_report(payload, market_name, region="Nasional"):
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
                    "Region": region,
                })
    return rows


def download_data(start="2020-01-01", end=None, market_type=1, province_id="", region="Nasional"):
    """Unduh laporan harga harian per tahun; end bersifat inklusif."""
    if market_type not in MARKET_TYPES:
        raise ValueError("Jenis pasar harus 1, 2, 3, atau 4.")
    if region != "Nasional" and not str(province_id).strip():
        raise ValueError("ID provinsi PIHPS wajib diisi untuk mengunduh data provinsi.")
    if region != "Nasional" and not str(province_id).strip().isdigit():
        raise ValueError(
            f"province_id harus berupa kode angka PIHPS, bukan nama '{province_id}'. "
            "Masukkan kode angka provinsi pada pilihan wilayah."
        )
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
            "province_id": str(province_id),
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
        try:
            payload = response.json()
        except ValueError as exc:
            content_type = response.headers.get("Content-Type", "tidak diketahui")
            body = re.sub(r"\s+", " ", response.text).strip()[:400]
            raise RuntimeError(
                "Respons PIHPS bukan JSON yang valid "
                f"(HTTP {response.status_code}, Content-Type: {content_type}, "
                f"province_id={province_id or 'kosong'}). Isi awal respons: {body or '[kosong]'}"
            ) from exc
        if payload.get("errors"):
            raise RuntimeError(f"PIHPS mengembalikan kesalahan: {payload['errors']}")
        records.extend(_rows_from_report(payload, MARKET_TYPES[market_type], region))
        print(f"Mengambil data {cursor} s.d. {chunk_end} ...")
        cursor = chunk_end + timedelta(days=1)

    if not records:
        print(
            f"PIHPS tidak mengembalikan observasi untuk {region} "
            f"(province_id={province_id or 'kosong'}) pada {start_date} s.d. {end_date}. "
            "File tidak dibuat/diubah. Periksa ID provinsi dan rentang tanggal."
        )
        return None
    data = pd.DataFrame.from_records(records)
    downloaded_count = len(data)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop_duplicates(["Date", "Commodity", "MarketType", "Region"], keep="last")
    data = data.sort_values(["Date", "Commodity"]).reset_index(drop=True)

    # Satu file provinsi menampung histori dari semua jenis pasar. Pembaruan
    # mengganti irisan pasar yang sedang diunduh dan mempertahankan pasar lain.
    path = raw_file(market_type, region)
    if path.exists():
        try:
            old = pd.read_csv(path, parse_dates=["Date"])
            if "MarketType" in old.columns:
                current_market = old[old["MarketType"].astype(str) == MARKET_TYPES[market_type]]
                other_markets = old[old["MarketType"].astype(str) != MARKET_TYPES[market_type]]
                current_market = pd.concat([current_market, data], ignore_index=True)
                current_market = current_market.drop_duplicates(
                    ["Date", "Commodity", "MarketType", "Region"], keep="last"
                )
                data = pd.concat([other_markets, current_market], ignore_index=True)
                data = data.drop_duplicates(["Date", "Commodity", "MarketType", "Region"], keep="last")
                data = data.sort_values(["MarketType", "Date", "Commodity"]).reset_index(drop=True)
        except (ValueError, OSError, pd.errors.ParserError):
            pass

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(
        f"Data PIHPS ({region} | {MARKET_TYPES[market_type]}) diperbarui: {path}\n"
        f"Baris baru: {downloaded_count} | Total baris dalam file: {len(data)} | "
        f"Jenis pasar dalam file: {data['MarketType'].nunique()} | "
        f"Rentang aktual: {data['Date'].min().date()} s.d. {data['Date'].max().date()}"
    )
    return data


def load_data(market_type=1, path=None, region="Nasional"):
    """Baca data mentah PIHPS dan susun menjadi tanggal × komoditas."""
    path = Path(path) if path else raw_file(market_type, region)
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
    if "MarketType" in raw.columns:
        raw = raw[raw["MarketType"].astype(str) == MARKET_TYPES[market_type]]
    if raw.empty:
        print(f"Tidak ada data untuk {MARKET_TYPES[market_type]} di wilayah {region}.")
        return pd.DataFrame()
    wide = raw.pivot_table(index="Date", columns="Commodity", values="Price", aggfunc="mean")
    wide.columns.name = None
    wide.index.name = "Date"
    return wide.sort_index()


def process_data(df, market_type=1, region="Nasional"):
    """Rapikan data analisis; data provinsi tetap dalam satu CSV gabungan."""
    clean = df.copy().sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    if region != "Nasional":
        print(
            f"Analisis memakai {MARKET_TYPES[market_type]} dari dataset gabungan "
            f"{raw_file(market_type, region)} ({len(clean)} tanggal)."
        )
        return clean
    path = processed_file(market_type, region)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(path, index_label="Date")
    print(f"Data bersih PIHPS diperbarui: {path} ({len(clean)} tanggal)")
    return clean
