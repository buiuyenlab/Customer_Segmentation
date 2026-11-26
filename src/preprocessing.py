"""Các hàm tiền xử lý: học tham số trên train, áp dụng cho mọi tập dữ liệu.

Quy trình: chọn biến -> điền thiếu bằng median -> loại biến tương quan > 0.8
-> winsorize 1%-99% -> log1p biến lệch phải (skew > 1) -> chuẩn hóa -> PCA 95%.
Mọi tham số (median, ngưỡng, scaler, PCA) chỉ học trên tập train.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SUGGESTED_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT"
]
LOG_CANDIDATES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PURCHASES_TRX"
]
CORRELATION_THRESHOLD = 0.8
PCA_VARIANCE = 0.95


def drop_correlated_features(data, candidates,
                             threshold=CORRELATION_THRESHOLD):
    """Loại lần lượt biến trong cặp tương quan tuyệt đối cao nhất > ngưỡng.

    Giữ biến ít tương quan hơn với các biến còn lại; dùng thứ tự cột ban
    đầu để phá hòa ổn định. Trả về (biến giữ lại, danh sách biến bị loại).
    """
    remaining = list(candidates)
    dropped_features = []
    while len(remaining) > 1:
        corr = data[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        if pairs.empty or pairs.iloc[0] <= threshold:
            break
        left, right = pairs.index[0]
        left_mean = corr[left].drop(left).mean()
        right_mean = corr[right].drop(right).mean()
        dropped = right if right_mean >= left_mean else left
        kept = left if dropped == right else right
        dropped_features.append((dropped, kept, float(corr.loc[left, right])))
        remaining.remove(dropped)
    return remaining, dropped_features


def fit_preprocessor(train_df, candidates=SUGGESTED_FEATURES):
    """Học toàn bộ tham số tiền xử lý CHỈ trên tập train."""
    candidates = [column for column in candidates if column in train_df]
    if not candidates:
        raise ValueError("Không tìm thấy biến nào trong danh sách gợi ý.")
    features = train_df[candidates].replace([np.inf, -np.inf], np.nan)
    medians = features.median()
    imputed = features.fillna(medians)

    features_kept, dropped = drop_correlated_features(imputed, candidates)
    data = imputed[features_kept].copy()

    bounds = {}
    log_columns = []
    for column in features_kept:
        low, high = data[column].quantile([0.01, 0.99])
        bounds[column] = (float(low), float(high))
        data[column] = data[column].clip(low, high)
        if column in LOG_CANDIDATES and data[column].skew() > 1:
            if data[column].min() < 0:
                raise ValueError(f"{column} có giá trị âm, không thể Log1p.")
            data[column] = np.log1p(data[column])
            log_columns.append(column)

    scaler = StandardScaler().fit(data.to_numpy())
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full").fit(
        scaler.transform(data.to_numpy())
    )
    return {
        "features": features_kept,
        "dropped": dropped,
        "medians": medians[features_kept],
        "bounds": bounds,
        "log_columns": log_columns,
        "scaler": scaler,
        "pca": pca
    }


def transform(df, preprocessor):
    """Áp dụng tham số đã học. Trả về (X_scaled, X_pca), giữ nguyên index."""
    columns = preprocessor["features"]
    data = (
        df[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(preprocessor["medians"])
    )
    for column in columns:
        low, high = preprocessor["bounds"][column]
        data[column] = data[column].clip(low, high)
        if column in preprocessor["log_columns"]:
            data[column] = np.log1p(data[column])
    scaled = preprocessor["scaler"].transform(data.to_numpy())
    pca_values = preprocessor["pca"].transform(scaled)
    X_scaled = pd.DataFrame(scaled, columns=columns, index=df.index)
    X_pca = pd.DataFrame(
        pca_values,
        columns=[f"PC{i + 1}" for i in range(pca_values.shape[1])],
        index=df.index
    )
    return X_scaled, X_pca
