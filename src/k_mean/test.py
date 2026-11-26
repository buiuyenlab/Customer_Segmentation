"""BƯỚC 4 - Đánh giá K-Means trên tập test.

Chạy:  python src/k_mean/test.py
Cần:   python src/k_mean/train.py đã chạy trước.

Phân cụm không có nhãn đúng, nên tập test dùng để kiểm tra khả năng tổng
quát hóa: chỉ số cụm trên test, tỷ lệ cụm train vs test, và Generalization
ARI (huấn luyện lại độc lập trên test rồi so với nhãn dự đoán).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    CH_NOTE,
    cluster_metrics,
    cluster_profile,
    cluster_shares,
    generalization_ari,
    silhouette_by_cluster,
    train_test_silhouette_report
)
from src.k_mean.model import make_kmeans  # noqa: E402
from src.utils import load_model, load_raw, load_split, save_csv  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_cluster_heatmaps,
    plot_clusters,
    plot_silhouette_samples,
    plot_train_test_shares
)


def main():
    print("=" * 72)
    print("BƯỚC 4: ĐÁNH GIÁ K-MEANS TRÊN TẬP TEST")
    print("=" * 72)
    model = load_model(config.KMEANS_MODEL, "src/k_mean/train.py")
    k = model.n_clusters
    X_train, X_test = load_split("train"), load_split("test")
    X_test_scaled = load_split("test", "scaled")

    train_labels = model.predict(X_train.to_numpy())
    test_labels = model.predict(X_test.to_numpy())

    gen_ari = generalization_ari(
        lambda seed: make_kmeans(k, seed), X_test, test_labels
    )
    rows = []
    for split, X, labels in (("Train", X_train, train_labels),
                             ("Test", X_test, test_labels)):
        row = {"Model": "K-Means", "Split": split,
               **cluster_metrics(X, labels, k)}
        row["Generalization_ARI"] = gen_ari if split == "Test" else None
        rows.append(row)
    evaluation = pd.DataFrame(rows)
    shares = cluster_shares(train_labels, test_labels, k)

    print(evaluation.drop(columns=["Model"]).round(4).to_string(index=False))
    print("\nTỷ lệ khách hàng mỗi cụm (%):")
    print(shares.round(2).to_string(index=False))
    print(CH_NOTE)
    print(f"\nGeneralization ARI trên test = {gen_ari:.4f} "
          "(gần 1 = cụm lặp lại tốt trên khách hàng mới)")
    print("Chênh lệch tỷ lệ cụm train/test lớn nhất: "
          f"{shares['Difference_pp'].abs().max():.2f} điểm %")

    ci_table, verdict = train_test_silhouette_report(
        X_train, train_labels, X_test, test_labels)
    print(f"\nSilhouette bootstrap ({config.BOOTSTRAP_RUNS} lần, KTC 95%):")
    print(ci_table.round(4).to_string(index=False))
    print(verdict)

    per_cluster, negative_share, sample_values = silhouette_by_cluster(
        X_test, test_labels, k)
    print("\nSilhouette theo cụm trên test:")
    print(per_cluster.round(4).to_string(index=False))
    weakest = per_cluster.loc[per_cluster["Mean_Silhouette"].idxmin()]
    print(f"Tỷ lệ khách test có silhouette < 0 (dễ bị gán nhầm): "
          f"{negative_share:.1%}. Cụm tách kém nhất: cụm "
          f"{int(weakest['Cluster'])} (Silhouette={weakest['Mean_Silhouette']:.3f}).")

    out = config.KMEANS_RESULTS
    save_csv(evaluation, out / "test_evaluation.csv")
    save_csv(shares, out / "train_test_cluster_shares.csv")
    save_csv(ci_table, out / "silhouette_bootstrap_ci.csv")
    save_csv(per_cluster, out / "test_silhouette_by_cluster.csv")
    save_csv(pd.DataFrame({"CUST_ID": X_test.index, "Cluster": test_labels}),
             out / "test_labels.csv")
    raw_test = load_raw().loc[X_test.index]
    save_csv(cluster_profile(raw_test, test_labels),
             out / "test_cluster_profile.csv")
    plots = out / "plots"
    plot_clusters(X_test, {"K-Means": test_labels},
                  plots / "test_clusters_pc1_pc2.png",
                  title_suffix=" - tập test")
    plot_cluster_heatmaps(X_test_scaled, {"K-Means (test)": test_labels},
                          plots / "test_cluster_heatmap.png")
    plot_train_test_shares(shares, plots / "train_test_shares.png", "K-Means")
    plot_silhouette_samples(sample_values, test_labels,
                            plots / "test_silhouette_samples.png",
                            f"Silhouette từng khách hàng - K-Means K={k} (test)")
    print(f"Đã lưu kết quả tại {out}")
    print("Bước tiếp theo: python src/gmm/select_config.py")


if __name__ == "__main__":
    main()
