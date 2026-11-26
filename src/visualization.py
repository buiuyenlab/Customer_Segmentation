"""Các hàm vẽ biểu đồ dùng chung."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chỉ lưu file ảnh, không mở cửa sổ
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shade_reference(ax, selection):
    """Tô xám vùng K chỉ để tham khảo (nhỏ hơn K tối thiểu nghiệp vụ)."""
    if "in_business_range" not in selection:
        return
    reference = selection.loc[~selection["in_business_range"], "k"]
    if not reference.empty:
        ax.axvspan(reference.min() - 0.5, reference.max() + 0.5,
                   color="grey", alpha=0.15,
                   label="Chỉ tham khảo (dưới K tối thiểu nghiệp vụ)")


def plot_elbow(selection, path, selected_k):
    """Inertia theo K, đánh dấu K được chọn."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["inertia"], marker="o")
    ax.axvline(selected_k, color="tab:red", linestyle="--",
               label=f"K được chọn = {selected_k}")
    _shade_reference(ax, selection)
    ax.set(title="Elbow Method - K-Means (tập train)",
           xlabel="Số cụm (K)", ylabel="Inertia")
    ax.grid(True)
    ax.legend()
    _save(fig, path)


def plot_silhouette(selection, path, selected_k):
    """Silhouette theo K, đánh dấu K được chọn."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["silhouette"], marker="o")
    chosen = selection.loc[selection["k"] == selected_k, "silhouette"]
    ax.scatter([selected_k], chosen, color="tab:red", zorder=3,
               label=f"K được chọn = {selected_k}")
    _shade_reference(ax, selection)
    ax.set(title="Silhouette Score - K-Means (tập train)",
           xlabel="Số cụm (K)", ylabel="Silhouette")
    ax.grid(True)
    ax.legend()
    _save(fig, path)


def plot_gmm_information_criteria(records, chosen, path):
    """BIC/AIC theo số thành phần cho từng covariance_type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for covariance_type, group in records.groupby("covariance_type"):
        valid = group[group["valid"]]
        for ax, metric in zip(axes, ("bic", "aic")):
            ax.plot(valid["n_components"], valid[metric], marker="o",
                    label=covariance_type)
    valid_records = records[records["valid"]]
    for ax, metric in zip(axes, ("bic", "aic")):
        scan_min = valid_records.loc[valid_records[metric].idxmin()]
        ax.scatter([chosen["n_components"]], [chosen[metric]],
                   color="red", marker="*", s=180, zorder=5,
                   label=f"Cấu hình chọn: K={chosen['n_components']}, "
                         f"{chosen['covariance_type']}")
        ax.scatter([scan_min["n_components"]], [scan_min[metric]],
                   color="black", marker="x", s=90, zorder=6,
                   label=f"Thấp nhất trong dải: "
                         f"K={int(scan_min['n_components'])}")
        ax.set_xlabel("Số thành phần")
        ax.set_ylabel(metric.upper() + " (thấp hơn tốt hơn)")
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_title("BIC theo covariance_type (tập train)")
    axes[1].set_title("AIC theo covariance_type (tập train)")
    _save(fig, path)


def plot_clusters(X_pca, labels_by_model, path, dims=(0, 1), title_suffix=""):
    """Scatter cụm trên hai thành phần PCA; mỗi mô hình một ô."""
    variance = X_pca.attrs.get("explained_variance_ratio_")
    n = len(labels_by_model)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False,
                             sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    x_values = X_pca.iloc[:, dims[0]].to_numpy()
    y_values = X_pca.iloc[:, dims[1]].to_numpy()
    for ax, (name, labels) in zip(axes[0], labels_by_model.items()):
        labels = np.asarray(labels)
        for color_index, cluster in enumerate(np.unique(labels)):
            mask = labels == cluster
            ax.scatter(x_values[mask], y_values[mask], s=8, alpha=0.6,
                       color=colors(color_index % 10), label=f"Cụm {cluster}")
        ax.set_title(f"{name}{title_suffix}")
        ax.grid(alpha=0.2)
        ax.legend(title="Nhãn cụm", markerscale=2, fontsize=8)
        for axis, dim, setter in (("x", dims[0], ax.set_xlabel),
                                  ("y", dims[1], ax.set_ylabel)):
            label = f"PC{dim + 1}"
            if variance is not None:
                label += f" ({variance[dim]:.1%} phương sai)"
            setter(label)
    _save(fig, path)


def plot_cluster_sizes(labels_by_model, path, title="Kích thước cụm"):
    """Số khách hàng mỗi cụm."""
    counts = pd.concat(
        [pd.Series(np.asarray(labels)).value_counts().rename(name)
         for name, labels in labels_by_model.items()],
        axis=1
    ).fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set(title=title, xlabel="Nhãn cụm", ylabel="Số khách hàng")
    ax.legend(title="Mô hình / tập")
    _save(fig, path)


def plot_cluster_heatmaps(X_scaled, labels_by_model, path):
    """Heatmap trung bình biến chuẩn hóa theo cụm."""
    n = len(labels_by_model)
    fig, axes = plt.subplots(n, 1, figsize=(max(11, X_scaled.shape[1]), 4.5 * n),
                             squeeze=False)
    for ax, (name, labels) in zip(axes[:, 0], labels_by_model.items()):
        means = X_scaled.assign(Cluster=np.asarray(labels)) \
            .groupby("Cluster").mean()
        heatmap = ax.imshow(means.to_numpy(), cmap="coolwarm", vmin=-2,
                            vmax=2, aspect="auto")
        ax.set_xticks(np.arange(len(means.columns)))
        ax.set_xticklabels(means.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(means.index)))
        ax.set_yticklabels(means.index)
        fig.colorbar(heatmap, ax=ax, label="Trung bình chuẩn hóa", shrink=0.8)
        ax.set_title(f"Đặc trưng trung bình theo cụm - {name}")
        ax.set_xlabel("Biến")
        ax.set_ylabel("Cụm")
    _save(fig, path)


def plot_train_test_shares(shares, path, model_name):
    """So sánh tỷ lệ cụm giữa train và test."""
    fig, ax = plt.subplots(figsize=(9, 5))
    shares.set_index("Cluster")[["Train_Percentage", "Test_Percentage"]] \
        .rename(columns={"Train_Percentage": "Train",
                         "Test_Percentage": "Test"}) \
        .plot(kind="bar", ax=ax)
    ax.set(title=f"Tỷ lệ khách hàng mỗi cụm: train vs test - {model_name}",
           xlabel="Nhãn cụm", ylabel="% khách hàng")
    _save(fig, path)


def plot_silhouette_samples(values, labels, path, title):
    """Biểu đồ silhouette từng mẫu, nhóm theo cụm (dạng 'lưỡi dao')."""
    values, labels = np.asarray(values), np.asarray(labels)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_low = 10
    for index, cluster in enumerate(np.unique(labels)):
        cluster_values = np.sort(values[labels == cluster])
        y_high = y_low + len(cluster_values)
        ax.fill_betweenx(np.arange(y_low, y_high), 0, cluster_values,
                         color=colors(index % 10), alpha=0.75)
        ax.text(-0.08, (y_low + y_high) / 2, f"Cụm {cluster}", va="center",
                ha="right", fontsize=9)
        y_low = y_high + 10
    ax.axvline(values.mean(), color="tab:red", linestyle="--",
               label=f"Trung bình = {values.mean():.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title=title, xlabel="Silhouette từng khách hàng", ylabel="")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    _save(fig, path)
