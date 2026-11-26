"""BƯỚC 8 - So sánh K-Means và GMM, gán phân khúc cho toàn bộ khách hàng.

Chạy:  python src/compare_models.py
Cần:   src/k_mean/train.py, src/k_mean/test.py, src/gmm/train.py,
       src/gmm/test.py đã chạy trước.

- Bảng so sánh: chỉ số trên tập test, độ ổn định (train), Generalization
  ARI, mức đồng thuận K-Means/GMM trên test.
- Dùng mô hình đã huấn luyện trên train để gán cụm cho cả 8.950 khách hàng.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score
)

from src import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    cluster_metrics,
    cluster_profile,
    cluster_profile_median,
    generalization_ari,
    silhouette_by_cluster,
    stability_ari
)
from src.gmm.model import make_gmm  # noqa: E402
from src.segments import segment_names  # noqa: E402
from src.utils import (  # noqa: E402
    load_json,
    load_model,
    load_raw,
    load_split,
    require,
    save_csv
)
from src.visualization import (  # noqa: E402
    plot_cluster_heatmaps,
    plot_cluster_sizes,
    plot_clusters
)

PROFILE_COLUMNS = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
                   "PAYMENTS", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY",
                   "PRC_FULL_PAYMENT"]

# Tiêu chí so sánh: (cột, "max" = càng cao càng tốt / "min" = càng thấp càng tốt)
CRITERIA = [
    ("Silhouette", "max"),
    ("Davies_Bouldin", "min"),
    ("Negative_Silhouette_Share", "min"),
    ("Generalization_ARI", "max"),
    ("Stability_ARI_Mean", "max"),
]


def reference_gmm_row(X_train, X_test, k, covariance_type):
    """Huấn luyện GMM tham khảo (cấu hình BIC ưu tiên) và đo trên test."""
    model = make_gmm(k, covariance_type).fit(X_train)
    labels = model.predict(X_test)
    metrics = cluster_metrics(X_test, labels, k)
    _, negative_share, _ = silhouette_by_cluster(X_test, labels, k)
    stability_mean, stability_std = stability_ari(
        lambda seed: make_gmm(k, covariance_type, seed), X_train)
    gen_ari = generalization_ari(
        lambda seed: make_gmm(k, covariance_type, seed), X_test, labels)
    row = {
        "Model": f"GMM {covariance_type} (tham khảo)",
        "Number_of_Clusters": k,
        "Silhouette": metrics["Silhouette"],
        "Davies_Bouldin": metrics["Davies_Bouldin"],
        "Calinski_Harabasz": metrics["Calinski_Harabasz"],
        "Negative_Silhouette_Share": negative_share,
        "Generalization_ARI": gen_ari,
        "Stability_ARI_Mean": stability_mean,
        "Stability_ARI_Std": stability_std,
        "BIC_Train": float(model.bic(X_train)),
    }
    return row, labels


MODELS = {
    "K-Means": (config.KMEANS_RESULTS, config.KMEANS_MODEL, "src/k_mean"),
    "GMM": (config.GMM_RESULTS, config.GMM_MODEL, "src/gmm")
}


def main():
    print("=" * 72)
    print("BƯỚC 8: SO SÁNH K-MEANS VÀ GMM")
    print("=" * 72)
    models, rows = {}, []
    for name, (results_dir, model_path, folder) in MODELS.items():
        models[name] = load_model(model_path, f"{folder}/train.py")
        require(results_dir / "test_evaluation.csv", f"{folder}/test.py")
        test_eval = pd.read_csv(results_dir / "test_evaluation.csv",
                                encoding="utf-8-sig")
        train_eval = pd.read_csv(results_dir / "train_metrics.csv",
                                 encoding="utf-8-sig")
        test_row = test_eval[test_eval["Split"] == "Test"].iloc[0]
        rows.append({
            "Model": name,
            "Number_of_Clusters": int(test_row["Number_of_Clusters"]),
            "Silhouette": test_row["Silhouette"],
            "Davies_Bouldin": test_row["Davies_Bouldin"],
            "Calinski_Harabasz": test_row["Calinski_Harabasz"],
            "Generalization_ARI": test_row["Generalization_ARI"],
            "Stability_ARI_Mean": train_eval.loc[0, "Stability_ARI_Mean"],
            "Stability_ARI_Std": train_eval.loc[0, "Stability_ARI_Std"]
        })

    kmeans, gmm = models["K-Means"], models["GMM"]
    X_train = load_split("train")
    X_test, X_all = load_split("test"), load_split("all")
    test_km = kmeans.predict(X_test.to_numpy())
    test_gm = gmm.predict(X_test.to_numpy())
    for row, labels, k_model in ((rows[0], test_km, kmeans.n_clusters),
                                 (rows[1], test_gm, gmm.n_components)):
        _, row["Negative_Silhouette_Share"], _ = silhouette_by_cluster(
            X_test, labels, k_model)
    rows[1]["BIC_Train"] = float(gmm.bic(X_train.to_numpy()))

    # Mô hình tham khảo: GMM theo cấu hình BIC ưu tiên (không dùng cho web).
    test_ref = None
    if config.GMM_REFERENCE_COVARIANCE != gmm.covariance_type:
        print(f"Huấn luyện GMM tham khảo covariance_type="
              f"{config.GMM_REFERENCE_COVARIANCE} (cấu hình BIC ưu tiên)...")
        ref_row, test_ref = reference_gmm_row(
            X_train.to_numpy(), X_test.to_numpy(), gmm.n_components,
            config.GMM_REFERENCE_COVARIANCE)
        rows.append(ref_row)
    if kmeans.n_clusters != gmm.n_components:
        print("Cảnh báo: hai mô hình khác số cụm; ARI/NMI chỉ mang tính tham khảo.")
    agreement_ari = adjusted_rand_score(test_km, test_gm)
    agreement_nmi = normalized_mutual_info_score(test_km, test_gm)
    comparison = pd.DataFrame(rows)
    # Agreement_* = mức đồng thuận của từng mô hình với K-Means trên test.
    agree_ari = [1.0, agreement_ari]
    agree_nmi = [1.0, agreement_nmi]
    if test_ref is not None:
        agree_ari.append(adjusted_rand_score(test_km, test_ref))
        agree_nmi.append(normalized_mutual_info_score(test_km, test_ref))
    comparison["Agreement_ARI"] = agree_ari
    comparison["Agreement_NMI"] = agree_nmi
    # Giữ quy ước cũ: hàng K-Means mang giá trị đồng thuận K-Means/GMM
    # (build_web.py đọc ô này).
    comparison.loc[0, ["Agreement_ARI", "Agreement_NMI"]] = [agreement_ari,
                                                            agreement_nmi]
    comparison.insert(1, "Evaluated_On", "Test (chỉ số) / Train (ổn định)")

    print("\nBẢNG SO SÁNH (chỉ số đo trên tập test)")
    print(comparison.drop(columns=["Evaluated_On"]).round(4)
          .to_string(index=False))
    print(f"Đồng thuận K-Means/GMM trên test: ARI={agreement_ari:.4f}, "
          f"NMI={agreement_nmi:.4f}")
    print("(Agreement của hàng GMM tham khảo = đồng thuận với K-Means.)")
    print("Calinski-Harabasz chỉ so được giữa các mô hình trên cùng tập test.")

    print("\nSO SÁNH THEO TỪNG TIÊU CHÍ (không chỉ dựa vào Silhouette)")
    wins = {model: 0 for model in comparison["Model"]}
    for column, direction in CRITERIA:
        values = comparison.set_index("Model")[column]
        winner = values.idxmax() if direction == "max" else values.idxmin()
        wins[winner] += 1
        detail = ", ".join(f"{m}={v:.4f}" for m, v in values.items())
        print(f"  {column:<26} -> {winner:<24} ({detail})")
    main_models = {m: w for m, w in wins.items() if "tham khảo" not in m}
    best = max(main_models, key=main_models.get)
    print("Số tiêu chí thắng:", wins)
    print(f"Mô hình chính được khuyến nghị: {best}")
    print("Lưu ý: Silhouette ưu ái cụm tròn (kiểu K-Means), nên GMM full "
          "thường có Silhouette thấp hơn dù cụm tái lập tốt.")

    # Gán phân khúc cho toàn bộ khách hàng bằng mô hình đã học trên train.
    info = load_json(config.PREPROCESSING_INFO, "src/prepare_data.py")
    raw = load_raw()
    test_ids = set(X_test.index)
    all_km = kmeans.predict(X_all.to_numpy())
    all_proba = gmm.predict_proba(X_all.to_numpy())
    all_gm = all_proba.argmax(axis=1)
    segments = raw.copy()
    segments.insert(0, "Split",
                    ["Test" if i in test_ids else "Train" for i in raw.index])
    segments["KMeans_Cluster"] = all_km
    segments["GMM_Cluster"] = all_gm
    segments["GMM_Confidence"] = all_proba.max(axis=1)
    for component in range(all_proba.shape[1]):
        segments[f"GMM_Probability_{component}"] = all_proba[:, component]

    # Đặt tên phân khúc theo hồ sơ hành vi (không phụ thuộc số thứ tự cụm).
    profile_km = cluster_profile(raw, all_km)
    profile_gm = cluster_profile(raw, all_gm)
    names_km, names_gm = segment_names(profile_km), segment_names(profile_gm)
    profile_km.insert(1, "Segment", profile_km["Cluster"].map(names_km))
    profile_gm.insert(1, "Segment", profile_gm["Cluster"].map(names_gm))
    segments.insert(segments.columns.get_loc("KMeans_Cluster") + 1,
                    "KMeans_Segment", pd.Series(all_km).map(names_km).values)
    segments.insert(segments.columns.get_loc("GMM_Cluster") + 1,
                    "GMM_Segment", pd.Series(all_gm).map(names_gm).values)
    print("\nPHÂN KHÚC K-MEANS (toàn bộ khách hàng) - TRUNG BÌNH")
    print(profile_km[["Cluster", "Segment", "Customer_Count", "Percentage",
                      "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
                      "PRC_FULL_PAYMENT"]].round(2).to_string(index=False))

    # Trung vị: biến tiền tệ lệch phải mạnh, trung bình dễ bị kéo lệch.
    median_km = cluster_profile_median(raw, all_km, PROFILE_COLUMNS)
    median_km.insert(1, "Segment", median_km["Cluster"].map(names_km))
    median_gm = cluster_profile_median(raw, all_gm, PROFILE_COLUMNS)
    median_gm.insert(1, "Segment", median_gm["Cluster"].map(names_gm))
    print("\nPHÂN KHÚC K-MEANS (toàn bộ khách hàng) - TRUNG VỊ")
    print(median_km.drop(columns=["Customer_Count"]).round(2)
          .to_string(index=False))
    # Kiểm tra tên phân khúc "Hạn mức cao" có đúng theo trung vị không.
    high_limit = [c for c, n in names_km.items() if n.startswith("Hạn mức cao")]
    if high_limit:
        limits = median_km.set_index("Cluster")["CREDIT_LIMIT"]
        rank = int(limits.rank(ascending=False)[high_limit[0]])
        print(f"Kiểm tra tên: cụm 'Hạn mức cao...' có trung vị CREDIT_LIMIT "
              f"= {limits[high_limit[0]]:.0f}, xếp hạng {rank}/{len(limits)}"
              + (" -> tên hợp lý." if rank == 1 else
                 " -> CÂN NHẮC đổi tên phân khúc."))

    boundary = all_proba.max(axis=1) < config.BOUNDARY_CONFIDENCE
    segments["GMM_Boundary"] = boundary
    print(f"\nKhách hàng ở vùng ranh giới GMM (xác suất < "
          f"{config.BOUNDARY_CONFIDENCE:.0%}): {boundary.mean():.1%} "
          f"({int(boundary.sum())} khách) - nên xem xét thủ công khi áp chính sách.")

    out = config.COMPARISON_RESULTS
    save_csv(comparison, out / "model_comparison.csv")
    save_csv(segments.reset_index(), out / "customer_segments.csv")
    save_csv(profile_km, out / "cluster_profiles_kmeans.csv")
    save_csv(profile_gm, out / "cluster_profiles_gmm.csv")
    save_csv(median_km, out / "cluster_profiles_kmeans_median.csv")
    save_csv(median_gm, out / "cluster_profiles_gmm_median.csv")

    plots = out / "plots"
    labels = {"K-Means": all_km, f"GMM ({gmm.covariance_type})": all_gm}
    plot_clusters(X_all, labels, plots / "clusters_pc1_pc2.png",
                  title_suffix=" - toàn bộ khách hàng")
    if info["n_components"] >= 3:
        plot_clusters(X_all, labels, plots / "clusters_pc1_pc3.png",
                      dims=(0, 2), title_suffix=" - toàn bộ khách hàng")
    plot_cluster_sizes(labels, plots / "cluster_sizes.png",
                       "Kích thước cụm - toàn bộ khách hàng")
    plot_cluster_heatmaps(load_split("all", "scaled"), labels,
                          plots / "cluster_feature_heatmaps.png")
    print(f"\nĐã lưu bảng so sánh, phân khúc khách hàng và hồ sơ cụm tại {out}")
    print("Cập nhật trang web (tùy chọn): python build_web.py")


if __name__ == "__main__":
    main()
