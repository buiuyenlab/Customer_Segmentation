"""BƯỚC 5 - Chọn cấu hình GMM (số thành phần, covariance_type) trên tập train.

Chạy:  python src/gmm/select_config.py
Cần:   python src/prepare_data.py đã chạy trước; nếu GMM_N_COMPONENTS = None
       (mặc định, dùng cùng K với K-Means) thì cần cả src/k_mean/select_k.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.gmm.model import (  # noqa: E402
    bic_local_minima,
    scan_gmm,
    select_covariance
)
from src.utils import load_json, load_split, save_csv, save_json  # noqa: E402
from src.visualization import plot_gmm_information_criteria  # noqa: E402


def main():
    print("=" * 72)
    print("BƯỚC 5: GMM - CHỌN CẤU HÌNH TRÊN TẬP TRAIN")
    print("=" * 72)
    X_train = load_split("train")
    if config.GMM_N_COMPONENTS is None:
        k = load_json(config.KMEANS_SELECTION,
                      "src/k_mean/select_k.py")["best_k"]
        k_source = "cùng K với K-Means để so sánh công bằng"
    else:
        k = config.GMM_N_COMPONENTS
        k_source = "đặt thủ công trong src/config.py"
    print(f"Số thành phần GMM: K={k} ({k_source}).")
    print(f"Quét K={config.GMM_K_MIN}..{config.GMM_K_MAX} x "
          f"{', '.join(config.COVARIANCE_TYPES)} (mất vài phút)...\n")

    records = scan_gmm(X_train)
    chosen = select_covariance(records, k)
    valid = records[records["valid"]]
    scan_min = valid.loc[valid["bic"].idxmin()]
    minima = bic_local_minima(records)

    print(f"\nChọn GMM K={k}, covariance_type={chosen['covariance_type']} "
          f"theo quy tắc '{chosen['rule']}' "
          f"(Silhouette={chosen['silhouette']:.4f}, BIC={chosen['bic']:.2f}).")
    print(f"Tại K={k}, BIC thấp nhất thuộc {chosen['bic_best_covariance_type']} "
          f"(BIC={chosen['bic_best_at_k']:.2f}).")
    if chosen["covariance_type"] != chosen["bic_best_covariance_type"]:
        print(f"-> Quy tắc '{chosen['rule']}' và BIC chọn khác nhau. Giữ "
              f"'{chosen['covariance_type']}' cho phân khúc chính (tách cụm rõ, "
              "web chỉ hỗ trợ spherical); cấu hình "
              f"'{config.GMM_REFERENCE_COVARIANCE}' sẽ được huấn luyện làm mô "
              "hình THAM KHẢO ở bước so sánh (src/compare_models.py).")
    at_boundary = int(scan_min["n_components"]) == config.GMM_K_MAX
    print(f"BIC thấp nhất toàn dải: K={int(scan_min['n_components'])} "
          f"({scan_min['covariance_type']}, BIC={scan_min['bic']:.2f})"
          + (f" ở biên K={config.GMM_K_MAX}." if at_boundary else "."))
    print("BIC giảm liên tục khi tăng K nghĩa là dữ liệu không gồm vài cụm "
          "Gauss rõ ràng; K=5 là lựa chọn nghiệp vụ, không phải K tối ưu theo BIC.")
    if minima:
        print("Cực tiểu nội bộ BIC:", "; ".join(
            f"K={m['n_components']} {m['covariance_type']} "
            f"(BIC={m['bic']:.2f})" for m in minima))
    else:
        print(f"Không thấy cực tiểu nội bộ BIC trong "
              f"K={config.GMM_K_MIN}..{config.GMM_K_MAX}; không diễn giải K "
              "tại biên là số cụm tối ưu.")

    out = config.GMM_RESULTS
    save_csv(records, out / "gmm_selection.csv")
    save_json({**chosen, "k_source": k_source,
               "bic_scan_minimum": {
                   "n_components": int(scan_min["n_components"]),
                   "covariance_type": scan_min["covariance_type"],
                   "bic": float(scan_min["bic"]),
                   "at_search_boundary": at_boundary},
               "bic_local_minima": minima}, config.GMM_SELECTION)
    plot_gmm_information_criteria(records, chosen, out / "plots" / "gmm_aic_bic.png")
    print(f"Đã lưu kết quả tại {out}")
    print("Bước tiếp theo: python src/gmm/train.py")


if __name__ == "__main__":
    main()
