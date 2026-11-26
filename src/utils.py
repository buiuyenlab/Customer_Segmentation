"""Hàm tiện ích đọc/ghi dùng chung giữa các bước."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src import config


def require(path, step):
    """Dừng với hướng dẫn rõ ràng nếu bước trước chưa được chạy."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Thiếu file {path.relative_to(config.ROOT)}.\n"
            f"Hãy chạy bước trước: python {step}"
        )


def save_csv(df, path, index=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path, step):
    require(path, step)
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path, step):
    require(path, step)
    return joblib.load(path)


def load_raw():
    """Dữ liệu gốc, index = CUST_ID."""
    require(config.DATA_PATH, "(đặt file CC GENERAL.csv vào thư mục dự án)")
    return pd.read_csv(config.DATA_PATH).set_index("CUST_ID")


def load_split(split, kind="pca"):
    """Đọc dữ liệu đã xử lý: split = train/test/all, kind = pca/scaled."""
    path = config.PROCESSED_DIR / f"{split}_{kind}.csv"
    require(path, "src/prepare_data.py")
    data = pd.read_csv(path, index_col="CUST_ID")
    if kind == "pca":
        info = load_json(config.PREPROCESSING_INFO, "src/prepare_data.py")
        data.attrs["explained_variance_ratio_"] = (
            info["explained_variance_ratio"]
        )
    return data
