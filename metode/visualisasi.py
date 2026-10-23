"""Helper untuk menyimpan grafik dan membukanya saat backend mendukung GUI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tampilkan_grafik(fig, nama_file):
    import matplotlib.pyplot as plt

    folder = ROOT / "hasil" / "grafik"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / nama_file
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Grafik tersimpan: {path}")
    # Di desktop, jendela grafik muncul sampai ditutup pengguna. Pada backend
    # headless grafik tetap disimpan agar bisa dibuka dari file PNG.
    backend = plt.get_backend().lower()
    non_interactive = backend in {"agg", "pdf", "ps", "svg", "cairo", "template"} or "inline" in backend
    if not non_interactive:
        plt.show()
    plt.close(fig)
