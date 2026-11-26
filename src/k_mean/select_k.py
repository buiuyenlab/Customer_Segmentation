"""BƯỚC 2 - Chọn số cụm K cho K-Means (chỉ dùng tập train).

Chạy:  python src/k_mean/select_k.py
Cần:   python src/prepare_data.py đã chạy trước.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.k_mean.model import scan_k  # noqa: E402
from src.utils import load_split, save_csv, save_json  # noqa: E402
from src.visualization import plot_elbow, plot_silhouette  # noqa: E402


def main():
    print("=" * 72)
    print("BƯỚC 2: K-MEANS - CHỌN SỐ CỤM K TRÊN TẬP TRAIN")
    print("=" * 72)
    X_train = load_split("train")
    print(f"Tập train: {len(X_train)} khách, {X_train.shape[1]} thành phần PCA.")

    table, best_k = scan_k(X_train)
    print(
        f"\nChọn K={best_k} theo thứ hạng trung bình Silhouette/DB/CH trong dải "
        f"nghiệp vụ K={config.KMEANS_K_MIN}..{config.KMEANS_K_MAX} "
        f"(cụm nhỏ nhất >= {config.MIN_CLUSTER_SHARE:.0%})."
    )
    # Minh bạch: so với K nhỏ hơn mức tối thiểu nghiệp vụ.
    chosen = table.loc[table["k"] == best_k].iloc[0]
    reference = table[~table["in_business_range"]]
    if not reference.empty:
        top = reference.loc[reference["silhouette"].idxmax()]
        if top["silhouette"] > chosen["silhouette"]:
            print(f"Lưu ý: K={int(top['k'])} (ngoài dải nghiệp vụ) có Silhouette "
                  f"{top['silhouette']:.4f} > {chosen['silhouette']:.4f} của K={best_k}. "
                  f"K={best_k} được chọn vì nhu cầu tối thiểu "
                  f"{config.KMEANS_K_MIN} phân khúc, không phải vì là K tối ưu "
                  "tuyệt đối về mặt thống kê.")
        else:
            print(f"K={best_k} cũng tốt hơn các K nhỏ hơn về Silhouette.")
    runner_up = table[table["eligible"] & (table["k"] != best_k)]
    if not runner_up.empty:
        second = runner_up.sort_values("mean_metric_rank").iloc[0]
        print(f"K xếp thứ 2: K={int(second['k'])} "
              f"(Silhouette={second['silhouette']:.4f}, chênh "
              f"{chosen['silhouette'] - second['silhouette']:+.4f}).")

    save_csv(table, config.KMEANS_RESULTS / "k_selection.csv")
    save_json({
        "best_k": best_k,
        "k_min": config.KMEANS_K_MIN,
        "scan_min": config.KMEANS_SCAN_MIN,
        "k_max": config.KMEANS_K_MAX,
        "min_cluster_share": config.MIN_CLUSTER_SHARE,
        "criterion": "Thứ hạng trung bình Silhouette/Davies-Bouldin/"
                     "Calinski-Harabasz, có ràng buộc cụm nhỏ nhất"
    }, config.KMEANS_SELECTION)
    plots = config.KMEANS_RESULTS / "plots"
    plot_elbow(table, plots / "elbow_method.png", best_k)
    plot_silhouette(table, plots / "silhouette_score.png", best_k)
    print(f"Đã lưu kết quả tại {config.KMEANS_RESULTS}")
    print("Bước tiếp theo: python src/k_mean/train.py")


if __name__ == "__main__":
    main()
