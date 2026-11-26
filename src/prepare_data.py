"""BƯỚC 1 - Xử lý dữ liệu và chia train/test.

Chạy:  python src/prepare_data.py

- Đọc CC GENERAL.csv, lưu minh chứng chất lượng dữ liệu trước xử lý.
- Chia 80% train / 20% test (random_state cố định).
- Học tham số tiền xử lý CHỈ trên train (không rò rỉ dữ liệu test).
- Lưu dữ liệu đã xử lý (chuẩn hóa + PCA) cho train, test và toàn bộ khách
  hàng vào data/processed/, lưu bộ tiền xử lý vào models/preprocessor.joblib.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import fit_preprocessor, transform  # noqa: E402
from src.utils import load_raw, save_csv, save_json, save_model  # noqa: E402


def data_quality_report(df):
    """Thống kê chất lượng dữ liệu theo từng cột."""
    numeric = df.select_dtypes(include=np.number)
    return pd.DataFrame({
        "so_gia_tri_thieu": df.isna().sum(),
        "so_gia_tri_duy_nhat": df.nunique(dropna=True),
        "kieu_du_lieu": df.dtypes.astype(str),
        "so_gia_tri_vo_cuc": numeric.apply(lambda col: np.isinf(col).sum()),
        "min": numeric.min(),
        "median": numeric.median(),
        "max": numeric.max(),
        "skew": numeric.skew()
    })


def main():
    print("=" * 72)
    print("BƯỚC 1: XỬ LÝ DỮ LIỆU VÀ CHIA TRAIN/TEST")
    print("=" * 72)
    raw = load_raw()
    print(f"Dữ liệu gốc: {raw.shape[0]} khách hàng, {raw.shape[1]} biến.")
    print(f"Tổng số giá trị thiếu: {int(raw.isna().sum().sum())}")

    evidence_dir = config.PREPROCESSING_RESULTS
    save_csv(raw.head(10), evidence_dir / "01_du_lieu_goc_10_dong.csv",
             index=True)
    save_csv(data_quality_report(raw), evidence_dir / "02_chat_luong_du_lieu.csv",
             index=True)

    # Chia train/test trước khi học bất kỳ tham số tiền xử lý nào.
    train_ids, test_ids = train_test_split(
        raw.index, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    train_raw, test_raw = raw.loc[train_ids], raw.loc[test_ids]
    print(f"Train: {len(train_raw)} khách | Test: {len(test_raw)} khách "
          f"(test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE})")

    preprocessor = fit_preprocessor(train_raw)
    print("Biến được dùng:", ", ".join(preprocessor["features"]))
    if preprocessor["dropped"]:
        for dropped, kept, value in preprocessor["dropped"]:
            print(f"  Loại {dropped}: tương quan {value:.3f} với {kept}")
    else:
        print("Không có cặp biến nào tương quan tuyệt đối > 0.8.")
    print("Biến log1p (skew > 1):",
          ", ".join(preprocessor["log_columns"]) or "không có")
    pca = preprocessor["pca"]
    print(f"PCA giữ {pca.n_components_} thành phần, giải thích "
          f"{pca.explained_variance_ratio_.sum():.2%} phương sai (học trên train).")

    for split, part in (("train", train_raw), ("test", test_raw), ("all", raw)):
        X_scaled, X_pca = transform(part, preprocessor)
        save_csv(X_scaled, config.PROCESSED_DIR / f"{split}_scaled.csv",
                 index=True)
        save_csv(X_pca, config.PROCESSED_DIR / f"{split}_pca.csv", index=True)
        if split == "train":
            save_csv(X_scaled.head(10),
                     evidence_dir / "03_du_lieu_sau_chuan_hoa_10_dong.csv",
                     index=True)

    save_model(preprocessor, config.PREPROCESSOR_PATH)
    save_json({
        "features": preprocessor["features"],
        "dropped": preprocessor["dropped"],
        "log_columns": preprocessor["log_columns"],
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_train": len(train_raw),
        "n_test": len(test_raw),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE
    }, config.PREPROCESSING_INFO)

    summary = pd.DataFrame([
        {"giai_doan": "Du lieu goc", "so_dong": len(raw),
         "so_bien": raw.shape[1],
         "so_gia_tri_thieu": int(raw.isna().sum().sum()),
         "so_dong_trung_lap": int(raw.duplicated().sum())},
        {"giai_doan": "Train sau xu ly", "so_dong": len(train_raw),
         "so_bien": len(preprocessor["features"]),
         "so_gia_tri_thieu": 0, "so_dong_trung_lap": None},
        {"giai_doan": "Test sau xu ly", "so_dong": len(test_raw),
         "so_bien": len(preprocessor["features"]),
         "so_gia_tri_thieu": 0, "so_dong_trung_lap": None},
    ])
    save_csv(summary, evidence_dir / "04_tong_quan_truoc_sau.csv")
    print(f"\nĐã lưu dữ liệu xử lý tại {config.PROCESSED_DIR}")
    print(f"Đã lưu bộ tiền xử lý tại {config.PREPROCESSOR_PATH}")
    print(f"Minh chứng tiền xử lý tại {evidence_dir}")
    print("Bước tiếp theo: python src/k_mean/select_k.py")


if __name__ == "__main__":
    main()
