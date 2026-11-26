"""BƯỚC 3 - Huấn luyện K-Means trên tập train.

Chạy:  python src/k_mean/train.py
Cần:   python src/k_mean/select_k.py đã chạy trước.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    cluster_metrics,
    cluster_profile,
    stability_ari
)
from src.k_mean.model import make_kmeans  # noqa: E402
from src.utils import (  # noqa: E402
    load_json,
    load_raw,
    load_split,
    save_csv,
    save_model
)
from src.visualization import plot_cluster_heatmaps, plot_clusters  # noqa: E402


def main():
    print("=" * 72)
    print("BƯỚC 3: HUẤN LUYỆN K-MEANS TRÊN TẬP TRAIN")
    print("=" * 72)
    k = load_json(config.KMEANS_SELECTION, "src/k_mean/select_k.py")["best_k"]
    X_train = load_split("train")
    X_train_scaled = load_split("train", "scaled")

    model = make_kmeans(k).fit(X_train.to_numpy())
    labels = model.predict(X_train.to_numpy())
    save_model(model, config.KMEANS_MODEL)
    print(f"Đã huấn luyện K-Means K={k} trên {len(X_train)} khách hàng.")

    metrics = cluster_metrics(X_train, labels, k)
    print("Kiểm tra độ ổn định (subsampling 80%, "
          f"{config.STABILITY_RUNS} lần)...")
    mean_ari, std_ari = stability_ari(
        lambda seed: make_kmeans(k, seed), X_train
    )
    metrics.update({
        "Model": "K-Means", "Split": "Train",
        "Stability_ARI_Mean": mean_ari, "Stability_ARI_Std": std_ari
    })
    train_metrics = pd.DataFrame([metrics])
    print(train_metrics.drop(columns=["Model"]).round(4).to_string(index=False))

    out = config.KMEANS_RESULTS
    save_csv(train_metrics, out / "train_metrics.csv")
    save_csv(pd.DataFrame({"CUST_ID": X_train.index, "Cluster": labels}),
             out / "train_labels.csv")
    raw_train = load_raw().loc[X_train.index]
    save_csv(cluster_profile(raw_train, labels),
             out / "train_cluster_profile.csv")
    plots = out / "plots"
    plot_clusters(X_train, {"K-Means": labels},
                  plots / "train_clusters_pc1_pc2.png",
                  title_suffix=" - tập train")
    plot_cluster_heatmaps(X_train_scaled, {"K-Means (train)": labels},
                          plots / "train_cluster_heatmap.png")
    print(f"Đã lưu mô hình tại {config.KMEANS_MODEL}")
    print(f"Đã lưu kết quả tại {out}")
    print("Bước tiếp theo: python src/k_mean/test.py")


if __name__ == "__main__":
    main()
