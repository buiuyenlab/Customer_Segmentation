"""BƯỚC 6 - Huấn luyện GMM trên tập train.

Chạy:  python src/gmm/train.py
Cần:   python src/gmm/select_config.py đã chạy trước.
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
from src.gmm.model import make_gmm  # noqa: E402
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
    print("BƯỚC 6: HUẤN LUYỆN GMM TRÊN TẬP TRAIN")
    print("=" * 72)
    selection = load_json(config.GMM_SELECTION, "src/gmm/select_config.py")
    k, cov = selection["n_components"], selection["covariance_type"]
    X_train = load_split("train")
    X_train_scaled = load_split("train", "scaled")

    model = make_gmm(k, cov).fit(X_train.to_numpy())
    if not model.converged_:
        raise RuntimeError(f"GMM ({cov}, K={k}) không hội tụ.")
    labels = model.predict(X_train.to_numpy())
    confidence = model.predict_proba(X_train.to_numpy()).max(axis=1)
    save_model(model, config.GMM_MODEL)
    print(f"Đã huấn luyện GMM K={k}, covariance_type={cov} "
          f"trên {len(X_train)} khách hàng.")

    metrics = cluster_metrics(X_train, labels, k)
    print("Kiểm tra độ ổn định (subsampling 80%, "
          f"{config.STABILITY_RUNS} lần)...")
    mean_ari, std_ari = stability_ari(
        lambda seed: make_gmm(k, cov, seed), X_train
    )
    metrics.update({
        "Model": "GMM", "Split": "Train", "Covariance_Type": cov,
        "BIC": float(model.bic(X_train.to_numpy())),
        "Mean_Confidence": float(confidence.mean()),
        "Stability_ARI_Mean": mean_ari, "Stability_ARI_Std": std_ari
    })
    train_metrics = pd.DataFrame([metrics])
    print(train_metrics.drop(columns=["Model"]).round(4).to_string(index=False))

    out = config.GMM_RESULTS
    save_csv(train_metrics, out / "train_metrics.csv")
    save_csv(pd.DataFrame({"CUST_ID": X_train.index, "Cluster": labels,
                           "Confidence": confidence}),
             out / "train_labels.csv")
    raw_train = load_raw().loc[X_train.index]
    save_csv(cluster_profile(raw_train, labels),
             out / "train_cluster_profile.csv")
    plots = out / "plots"
    plot_clusters(X_train, {f"GMM ({cov})": labels},
                  plots / "train_clusters_pc1_pc2.png",
                  title_suffix=" - tập train")
    plot_cluster_heatmaps(X_train_scaled, {"GMM (train)": labels},
                          plots / "train_cluster_heatmap.png")
    print(f"Đã lưu mô hình tại {config.GMM_MODEL}")
    print(f"Đã lưu kết quả tại {out}")
    print("Bước tiếp theo: python src/gmm/test.py")


if __name__ == "__main__":
    main()
