"""Định nghĩa mô hình K-Means và hàm quét số cụm K."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src import config
from src.evaluation import cluster_metrics


def make_kmeans(n_clusters, random_state=config.RANDOM_STATE):
    """K-Means với tham số cố định cho toàn dự án."""
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=random_state
    )


def scan_k(X, k_min=config.KMEANS_K_MIN, k_max=config.KMEANS_K_MAX,
           min_cluster_share=config.MIN_CLUSTER_SHARE,
           scan_min=config.KMEANS_SCAN_MIN):
    """Quét K và chọn theo thứ hạng trung bình Silhouette / Davies-Bouldin /
    Calinski-Harabasz, chỉ xét các K có cụm nhỏ nhất >= min_cluster_share.

    K từ scan_min đến k_min - 1 vẫn được quét và ghi vào bảng để minh bạch,
    nhưng không tham gia xếp hạng (dưới số phân khúc tối thiểu nghiệp vụ).
    Thứ hạng chỉ tính trong dải [k_min, k_max] nên kết quả chọn K không đổi
    so với khi chỉ quét dải này.

    Trả về (bảng kết quả theo K, K tốt nhất).
    """
    X = np.asarray(X)
    k_max = min(k_max, len(X) - 1)
    scan_min = min(scan_min, k_min)
    rows = []
    for k in range(scan_min, k_max + 1):
        model = make_kmeans(k).fit(X)
        metrics = cluster_metrics(X, model.labels_, k)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": metrics["Silhouette"],
            "davies_bouldin": metrics["Davies_Bouldin"],
            "calinski_harabasz": metrics["Calinski_Harabasz"],
            "min_cluster_share": metrics["Min_Cluster_Share"]
        })
        print(f"K={k}: Inertia={model.inertia_:.2f}, "
              f"Silhouette={metrics['Silhouette']:.4f}, "
              f"Davies-Bouldin={metrics['Davies_Bouldin']:.4f}, "
              f"Calinski-Harabasz={metrics['Calinski_Harabasz']:.2f}, "
              f"cụm nhỏ nhất={metrics['Min_Cluster_Share']:.1%}"
              + ("  [tham khảo]" if k < k_min else ""))

    table = pd.DataFrame(rows)
    table["in_business_range"] = table["k"] >= k_min
    ranked = table[table["in_business_range"]]
    table["mean_metric_rank"] = (
        ranked["silhouette"].rank(ascending=False, method="average")
        + ranked["davies_bouldin"].rank(ascending=True, method="average")
        + ranked["calinski_harabasz"].rank(ascending=False, method="average")
    ) / 3
    table["eligible"] = (table["in_business_range"]
                         & (table["min_cluster_share"] >= min_cluster_share))
    candidates = table[table["eligible"]]
    if candidates.empty:
        print("Cảnh báo: không K nào đạt tỷ lệ cụm tối thiểu; xét cả dải nghiệp vụ.")
        candidates = table[table["in_business_range"]]
    best_k = int(
        candidates.sort_values(["mean_metric_rank", "k"]).iloc[0]["k"]
    )
    return table, best_k
