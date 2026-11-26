"""Các hàm đánh giá phân cụm dùng chung cho K-Means và GMM."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score
)

from src import config


def cluster_metrics(X, labels, n_clusters):
    """Silhouette, Davies-Bouldin, Calinski-Harabasz và tỷ lệ cụm."""
    X = np.asarray(X)
    shares = np.bincount(labels, minlength=n_clusters) / len(labels)
    result = {
        "Number_of_Customers": len(labels),
        "Number_of_Clusters": int(n_clusters),
        "Silhouette": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan,
        "Min_Cluster_Share": float(shares.min()),
        "Max_Cluster_Share": float(shares.max())
    }
    if np.unique(labels).size >= 2:
        result.update({
            "Silhouette": silhouette_score(
                X, labels, sample_size=min(config.SILHOUETTE_SAMPLE, len(X)),
                random_state=config.RANDOM_STATE
            ),
            "Davies_Bouldin": davies_bouldin_score(X, labels),
            "Calinski_Harabasz": calinski_harabasz_score(X, labels)
        })
    return result


def stability_ari(make_model, X, n_runs=config.STABILITY_RUNS,
                  fraction=config.STABILITY_FRACTION):
    """Độ ổn định bằng subsampling.

    Mỗi lần huấn luyện trên `fraction` dữ liệu (seed khác nhau), dự đoán lại
    toàn bộ X, rồi tính ARI trung bình giữa mọi cặp lần chạy.
    make_model(random_state) trả về một mô hình chưa huấn luyện.
    """
    X = np.asarray(X)
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_size = int(len(X) * fraction)
    label_runs = []
    for random_state in range(n_runs):
        indices = rng.choice(len(X), size=sample_size, replace=False)
        model = make_model(random_state).fit(X[indices])
        if hasattr(model, "converged_") and not model.converged_:
            print(f"Cảnh báo: seed={random_state} không hội tụ, bỏ qua.")
            continue
        label_runs.append(model.predict(X))
    if len(label_runs) < 2:
        raise RuntimeError("Không đủ lần chạy hợp lệ để tính ARI.")
    scores = [
        adjusted_rand_score(label_runs[a], label_runs[b])
        for a, b in combinations(range(len(label_runs)), 2)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def generalization_ari(make_model, X_test, predicted_labels):
    """Huấn luyện lại mô hình độc lập chỉ trên test và so với nhãn dự đoán
    từ mô hình train. ARI gần 1 = cấu trúc cụm lặp lại trên dữ liệu mới."""
    X_test = np.asarray(X_test)
    own_labels = make_model(config.RANDOM_STATE).fit(X_test).predict(X_test)
    return float(adjusted_rand_score(predicted_labels, own_labels))


def cluster_shares(train_labels, test_labels, n_clusters):
    """Tỷ lệ (%) khách hàng mỗi cụm trên train và test."""
    train = np.bincount(train_labels, minlength=n_clusters) / len(train_labels)
    test = np.bincount(test_labels, minlength=n_clusters) / len(test_labels)
    table = pd.DataFrame({
        "Cluster": range(n_clusters),
        "Train_Percentage": train * 100,
        "Test_Percentage": test * 100
    })
    table["Difference_pp"] = (
        table["Test_Percentage"] - table["Train_Percentage"]
    )
    return table


def cluster_profile(raw, labels):
    """Trung bình biến gốc theo cụm (đơn vị gốc để dễ diễn giải)."""
    numeric = raw.select_dtypes(include=np.number).copy()
    numeric["Cluster"] = np.asarray(labels)
    grouped = numeric.groupby("Cluster")
    profile = grouped.mean()
    counts = grouped.size()
    profile.insert(0, "Customer_Count", counts)
    profile.insert(1, "Percentage", counts / len(numeric) * 100)
    return profile.reset_index()


def silhouette_ci(X, labels, n_boot=config.BOOTSTRAP_RUNS, level=0.95):
    """Khoảng tin cậy bootstrap cho Silhouette (lấy mẫu lại có hoàn lại).

    Dùng để biết chênh lệch Silhouette train/test có thật hay chỉ là dao
    động do mẫu nhỏ. Trả về (trung bình, cận dưới, cận trên).
    """
    X, labels = np.asarray(X), np.asarray(labels)
    rng = np.random.RandomState(config.RANDOM_STATE)
    size = min(config.SILHOUETTE_SAMPLE, len(X))
    values = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=size, replace=True)
        if np.unique(labels[idx]).size >= 2:
            values.append(silhouette_score(X[idx], labels[idx]))
    alpha = (1 - level) / 2 * 100
    low, high = np.percentile(values, [alpha, 100 - alpha])
    return float(np.mean(values)), float(low), float(high)


def silhouette_by_cluster(X, labels, n_clusters):
    """Silhouette từng mẫu -> trung bình theo cụm và tỷ lệ mẫu âm.

    Mẫu có silhouette < 0 gần cụm khác hơn cụm của chính nó (dễ gán nhầm).
    Trả về (bảng theo cụm, tỷ lệ mẫu âm toàn bộ, mảng silhouette từng mẫu).
    """
    X, labels = np.asarray(X), np.asarray(labels)
    values = silhouette_samples(X, labels)
    table = pd.DataFrame({
        "Cluster": range(n_clusters),
        "Mean_Silhouette": [values[labels == c].mean() if np.any(labels == c)
                            else np.nan for c in range(n_clusters)],
        "Negative_Share": [(values[labels == c] < 0).mean()
                           if np.any(labels == c) else np.nan
                           for c in range(n_clusters)]
    })
    return table, float((values < 0).mean()), values


def cluster_profile_median(raw, labels, columns=None):
    """Trung vị biến gốc theo cụm (ít bị ngoại lai kéo lệch hơn trung bình)."""
    numeric = raw.select_dtypes(include=np.number)
    if columns is not None:
        numeric = numeric[[c for c in columns if c in numeric]]
    numeric = numeric.copy()
    numeric["Cluster"] = np.asarray(labels)
    grouped = numeric.groupby("Cluster")
    profile = grouped.median()
    counts = grouped.size()
    profile.insert(0, "Customer_Count", counts)
    profile.insert(1, "Percentage", counts / len(numeric) * 100)
    return profile.reset_index()


def train_test_silhouette_report(X_train, train_labels, X_test, test_labels):
    """So sánh Silhouette train/test bằng khoảng tin cậy bootstrap 95%.

    Trả về (bảng, kết luận dạng chữ). Hai khoảng giao nhau = tương đương,
    không kết luận "test tốt hơn" hay "test kém hơn".
    """
    rows = []
    for split, X, labels in (("Train", X_train, train_labels),
                             ("Test", X_test, test_labels)):
        mean, low, high = silhouette_ci(X, labels)
        rows.append({"Split": split, "Silhouette_Bootstrap_Mean": mean,
                     "CI95_Low": low, "CI95_High": high})
    table = pd.DataFrame(rows)
    train, test = table.iloc[0], table.iloc[1]
    overlap = (test["CI95_Low"] <= train["CI95_High"]
               and train["CI95_Low"] <= test["CI95_High"])
    if overlap:
        verdict = ("Khoảng tin cậy train/test giao nhau: Silhouette hai tập "
                   "TƯƠNG ĐƯƠNG (chênh lệch chỉ là dao động do lấy mẫu).")
    elif test["Silhouette_Bootstrap_Mean"] < train["Silhouette_Bootstrap_Mean"]:
        verdict = ("Silhouette test thấp hơn train có ý nghĩa: cụm tách kém "
                   "hơn trên khách hàng mới.")
    else:
        verdict = ("Silhouette test cao hơn train có ý nghĩa: kiểm tra lại "
                   "cách chia dữ liệu (test có thể dễ hơn train).")
    return table, verdict


CH_NOTE = ("Lưu ý: Calinski-Harabasz tăng theo số mẫu nên KHÔNG so sánh "
           "trực tiếp CH giữa train và test (hai tập khác kích thước).")
