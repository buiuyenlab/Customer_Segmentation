"""Cấu hình chung: đường dẫn, tham số chia dữ liệu và tham số mô hình."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Dữ liệu
DATA_PATH = ROOT / "CC GENERAL.csv"
PROCESSED_DIR = ROOT / "data" / "processed"     # dữ liệu sau xử lý (train/test/all)
MODELS_DIR = ROOT / "models"                    # mô hình đã huấn luyện (.joblib)

# Kết quả
RESULTS_DIR = ROOT / "results"
PREPROCESSING_RESULTS = RESULTS_DIR / "preprocessing"
KMEANS_RESULTS = RESULTS_DIR / "k_mean"
GMM_RESULTS = RESULTS_DIR / "gmm"
COMPARISON_RESULTS = RESULTS_DIR / "comparison"

# File trung gian giữa các bước
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
PREPROCESSING_INFO = PROCESSED_DIR / "preprocessing_info.json"
KMEANS_SELECTION = KMEANS_RESULTS / "best_k.json"
KMEANS_MODEL = MODELS_DIR / "kmeans.joblib"
GMM_SELECTION = GMM_RESULTS / "gmm_config.json"
GMM_MODEL = MODELS_DIR / "gmm.joblib"

# Chia train/test
RANDOM_STATE = 42
TEST_SIZE = 0.2

# K-Means: dải K và ràng buộc cụm nhỏ nhất (tỷ lệ khách hàng).
# K_MIN = 4 vì lý do nghiệp vụ: ngân hàng cần ít nhất 4 phân khúc để xây
# dựng chính sách khác nhau. Với K_MIN = 2, tiêu chí xếp hạng trên tập train
# chọn K=2 (thứ hạng TB 3.33 so với 3.67 của K=5) - quá thô để phân khúc.
KMEANS_K_MIN, KMEANS_K_MAX = 4, 10
# K=2..3 vẫn được quét và ghi vào bảng để minh bạch (chỉ để THAM KHẢO,
# không tham gia xếp hạng chọn K).
KMEANS_SCAN_MIN = 2
MIN_CLUSTER_SHARE = 0.05

# GMM: dải số thành phần và các kiểu hiệp phương sai được quét
GMM_K_MIN, GMM_K_MAX = 2, 15
COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
# Số thành phần GMM: None = dùng cùng K với K-Means để so sánh công bằng.
GMM_N_COMPONENTS = None
# Quy tắc chọn covariance_type tại K đã chọn: "silhouette" hoặc "bic".
# Giữ "silhouette" vì: (1) mục tiêu là phân khúc tách bạch để so với K-Means,
# (2) index.html chỉ tính được GMM 'spherical'. BIC ưu tiên 'full'; cấu hình
# này được huấn luyện thêm làm MÔ HÌNH THAM KHẢO trong bước so sánh.
GMM_COVARIANCE_RULE = "silhouette"
GMM_REFERENCE_COVARIANCE = "full"

# Khoảng tin cậy bootstrap cho Silhouette trên test
BOOTSTRAP_RUNS = 200
SILHOUETTE_SAMPLE = 3000
# Khách hàng GMM có xác suất thuộc cụm < ngưỡng = vùng ranh giới
BOUNDARY_CONFIDENCE = 0.6

# Kiểm tra độ ổn định (subsampling)
STABILITY_RUNS = 20
STABILITY_FRACTION = 0.8
