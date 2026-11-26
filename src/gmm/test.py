"""BƯỚC 7 - Đánh giá GMM trên tập test.

Chạy:  python src/gmm/test.py
Cần:   python src/gmm/train.py đã chạy trước.
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
from src.gmm.model import make_gmm  # noqa: E402
from src.utils import load_model, load_raw, load_split, save_csv  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_cluster_heatmaps,
    plot_clusters,
    plot_silhouette_samples,
    plot_train_test_shares
)


def main():
    print("=" * 72)
    print("BƯỚC 7: ĐÁNH GIÁ GMM TRÊN TẬP TEST")
    print("=" * 72)
    model = load_model(config.GMM_MODEL, "src/gmm/train.py")
    k, cov = model.n_components, model.covariance_type
    X_train, X_test = load_split("train"), load_split("test")
    X_test_scaled = load_split("test", "scaled")

    train_labels = model.predict(X_train.to_numpy())
    test_proba = model.predict_proba(X_test.to_numpy())
    test_labels = test_proba.argmax(axis=1)
    test_confidence = test_proba.max(axis=1)

    gen_ari = generalization_ari(
        lambda seed: make_gmm(k, cov, seed), X_test, test_labels
    )
    rows = []
    for split, X, labels in (("Train", X_train, train_labels),
                             ("Test", X_test, test_labels)):
        row = {"Model": "GMM", "Split": split, "Covariance_Type": cov,
               **cluster_metrics(X, labels, k),
               "Mean_Log_Likelihood": float(model.score(X.to_numpy()))}
        row["Generalization_ARI"] = gen_ari if split == "Test" else None
        row["Mean_Confidence"] = (
            float(test_confidence.mean()) if split == "Test" else None
        )
        row["Boundary_Share"] = (
            float((test_confidence < config.BOUNDARY_CONFIDENCE).mean())
            if split == "Test" else None
        )
        rows.append(row)
    evaluation = pd.DataFrame(rows)
    shares = cluster_shares(train_labels, test_labels, k)

    print(evaluation.drop(columns=["Model"]).round(4).to_string(index=False))
    print("\nTỷ lệ khách hàng mỗi cụm (%):")
    print(shares.round(2).to_string(index=False))
    print(CH_NOTE)
    print(f"\nGeneralization ARI trên test = {gen_ari:.4f}")
    ll_train = evaluation.loc[evaluation["Split"] == "Train",
                              "Mean_Log_Likelihood"].iloc[0]
    ll_test = evaluation.loc[evaluation["Split"] == "Test",
                             "Mean_Log_Likelihood"].iloc[0]
    gap = ll_train - ll_test
    print(f"Log-likelihood trung bình: train={ll_train:.4f}, test={ll_test:.4f} "
          f"(train - test = {gap:+.4f}).")
    if gap > 0.1:
        print("-> Train cao hơn test rõ rệt: có dấu hiệu overfit.")
    else:
        print("-> Chênh lệch nhỏ: không có dấu hiệu overfit.")
    print("Chênh lệch tỷ lệ cụm train/test lớn nhất: "
          f"{shares['Difference_pp'].abs().max():.2f} điểm %")

    ci_table, verdict = train_test_silhouette_report(
        X_train, train_labels, X_test, test_labels)
    print(f"\nSilhouette bootstrap ({config.BOOTSTRAP_RUNS} lần, KTC 95%):")
    print(ci_table.round(4).to_string(index=False))
    print(verdict)

    per_cluster, negative_share, sample_values = silhouette_by_cluster(
        X_test, test_labels, k)
    boundary_share = float((test_confidence < config.BOUNDARY_CONFIDENCE).mean())
    print("\nSilhouette theo cụm trên test:")
    print(per_cluster.round(4).to_string(index=False))
    print(f"Tỷ lệ khách test có silhouette < 0: {negative_share:.1%}")
    print(f"Tỷ lệ khách test ở vùng ranh giới (xác suất thuộc cụm < "
          f"{config.BOUNDARY_CONFIDENCE:.0%}): {boundary_share:.1%}")

    out = config.GMM_RESULTS
    save_csv(evaluation, out / "test_evaluation.csv")
    save_csv(shares, out / "train_test_cluster_shares.csv")
    save_csv(ci_table, out / "silhouette_bootstrap_ci.csv")
    save_csv(per_cluster, out / "test_silhouette_by_cluster.csv")
    labels_table = pd.DataFrame({"CUST_ID": X_test.index,
                                 "Cluster": test_labels,
                                 "Confidence": test_confidence})
    for component in range(k):
        labels_table[f"Probability_{component}"] = test_proba[:, component]
    save_csv(labels_table, out / "test_labels.csv")
    raw_test = load_raw().loc[X_test.index]
    save_csv(cluster_profile(raw_test, test_labels),
             out / "test_cluster_profile.csv")
    plots = out / "plots"
    plot_clusters(X_test, {f"GMM ({cov})": test_labels},
                  plots / "test_clusters_pc1_pc2.png",
                  title_suffix=" - tập test")
    plot_cluster_heatmaps(X_test_scaled, {"GMM (test)": test_labels},
                          plots / "test_cluster_heatmap.png")
    plot_train_test_shares(shares, plots / "train_test_shares.png", "GMM")
    plot_silhouette_samples(sample_values, test_labels,
                            plots / "test_silhouette_samples.png",
                            f"Silhouette từng khách hàng - GMM {cov} K={k} (test)")
    print(f"Đã lưu kết quả tại {out}")
    print("Bước tiếp theo: python src/compare_models.py")


if __name__ == "__main__":
    main()
