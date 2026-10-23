from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .visualisasi import tampilkan_grafik

ROOT = Path(__file__).resolve().parents[1]


def run(df):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    returns = df.sort_index().pct_change(fill_method=None)
    rows = []
    for commodity in df.columns:
        r = returns[commodity].dropna()
        if len(r) < 20:
            continue
        prices = df[commodity].dropna()
        rows.append({
            "commodity": commodity,
            "average_return_pct": r.mean() * 100,
            "volatility_pct": r.std() * (252 ** 0.5) * 100,
            "average_price": prices.mean(),
            "latest_return_pct": r.iloc[-1] * 100,
        })
    features = pd.DataFrame(rows).set_index("commodity")
    if len(features) < 3:
        print("Data valid dari minimal tiga komoditas diperlukan untuk clustering.")
        return
    columns = ["average_return_pct", "volatility_pct", "average_price", "latest_return_pct"]
    scaled = StandardScaler().fit_transform(features[columns])
    n_clusters = min(3, len(features))
    features["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    out = ROOT / "hasil" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "cluster_komoditas.csv", index_label="Commodity")
    print("Anggota cluster:")
    for cluster, group in features.groupby("cluster"):
        print(f"Cluster {cluster}: {', '.join(group.index)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster, group in features.groupby("cluster"):
        ax.scatter(group["average_return_pct"], group["volatility_pct"], label=f"Cluster {cluster}", s=65)
        for commodity, row in group.iterrows():
            ax.annotate(commodity, (row["average_return_pct"], row["volatility_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Clustering komoditas berdasarkan return dan volatilitas")
    ax.set_xlabel("Rata-rata return harian (%)")
    ax.set_ylabel("Volatilitas tahunan (%)")
    ax.legend()
    tampilkan_grafik(fig, "clustering_komoditas.png")
