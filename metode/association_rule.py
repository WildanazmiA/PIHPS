from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def _category(name):
    name = name.lower()
    for label in ("beras", "bawang", "cabai", "daging", "telur", "minyak", "gula"):
        if name.startswith(label):
            return label
    return "lainnya"


def run(df):
    from mlxtend.frequent_patterns import apriori, association_rules

    changes = df.sort_index().pct_change(fill_method=None)
    category_changes = {}
    for category in sorted({_category(c) for c in changes.columns}):
        cols = [c for c in changes.columns if _category(c) == category]
        category_changes[category] = changes[cols].median(axis=1)
    category_changes = pd.DataFrame(category_changes).dropna(how="all")
    basket = pd.DataFrame(index=category_changes.index)
    for category in category_changes:
        basket[f"{category}_naik"] = category_changes[category] > 0
        basket[f"{category}_turun"] = category_changes[category] < 0
    basket = basket.fillna(False).astype(bool)
    frequent = apriori(basket, min_support=.1, use_colnames=True, max_len=3)
    if frequent.empty:
        print("Tidak ditemukan pola berulang dengan support minimum 10%.")
        return
    rules = association_rules(frequent, metric="confidence", min_threshold=.5)
    if rules.empty:
        print("Pola ditemukan, tetapi tidak ada aturan dengan confidence minimal 50%.")
        return
    out = ROOT / "hasil" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    rules.to_csv(out / "association_rules_pangan.csv", index=False)
    top = rules.sort_values("lift", ascending=False).head(10).copy()
    print("Aturan dengan lift tertinggi:")
    print(top[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False))

    labels = top.apply(
        lambda row: f"{', '.join(sorted(row['antecedents']))} → {', '.join(sorted(row['consequents']))}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, max(4, .55 * len(top))))
    ax.barh(labels.iloc[::-1], top["lift"].iloc[::-1], color="#5277b8")
    ax.axvline(1, color="gray", linestyle="--", linewidth=1, label="Lift = 1")
    ax.set_title("Association Rule perubahan harga kategori pangan")
    ax.set_xlabel("Lift")
    ax.legend()
    tampilkan_grafik(fig, "association_rule_pangan.png")
