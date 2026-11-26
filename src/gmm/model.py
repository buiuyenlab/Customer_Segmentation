"""Định nghĩa mô hình GMM, hàm quét BIC/AIC và chọn covariance_type."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from src import config
from src.evaluation import cluster_metrics


def make_gmm(n_components, covariance_type,
             random_state=config.RANDOM_STATE):
    """GMM với tham số cố định cho toàn dự án."""
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=1e-4,
        n_init=10,
        max_iter=500,
        init_params="kmeans",
        tol=1e-3,
        random_state=random_state
    )


def scan_gmm(X, k_min=config.GMM_K_MIN, k_max=config.GMM_K_MAX,
             covariance_types=config.COVARIANCE_TYPES):
    """Quét số thành phần x covariance_type, ghi BIC/AIC và chỉ số cụm.

    Cấu hình hợp lệ: hội tụ, BIC/AIC hữu hạn, không có thành phần rỗng.
    """
    X = np.asarray(X)
    k_max = min(k_max, len(X) - 1)
    rows = []
    for covariance_type in covariance_types:
        for k in range(k_min, k_max + 1):
            row = {"n_components": k, "covariance_type": covariance_type,
                   "bic": np.nan, "aic": np.nan, "silhouette": np.nan,
                   "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                   "converged": False, "valid": False, "error": ""}
            try:
                model = make_gmm(k, covariance_type)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(X)
                labels = model.predict(X)
                row.update({
                    "bic": float(model.bic(X)),
                    "aic": float(model.aic(X)),
                    "converged": bool(model.converged_)
                })
                row["valid"] = bool(
                    model.converged_
                    and np.isfinite(row["bic"]) and np.isfinite(row["aic"])
                    and np.all(np.bincount(labels, minlength=k) > 0)
                    and np.min(model.weights_) >= 1e-4
                )
                if row["valid"]:
                    metrics = cluster_metrics(X, labels, k)
                    row.update({
                        "silhouette": metrics["Silhouette"],
                        "davies_bouldin": metrics["Davies_Bouldin"],
                        "calinski_harabasz": metrics["Calinski_Harabasz"]
                    })
            except (ValueError, FloatingPointError,
                    np.linalg.LinAlgError) as exc:
                row["error"] = str(exc)
            rows.append(row)
            status = "hợp lệ" if row["valid"] else "KHÔNG hợp lệ"
            print(f"GMM {covariance_type:9s} K={k:2d}: BIC={row['bic']:.2f}, "
                  f"AIC={row['aic']:.2f}, "
                  f"Silhouette={row['silhouette']:.4f} ({status})")
    return pd.DataFrame(rows)


def bic_local_minima(records):
    """Các K là cực tiểu nội bộ của BIC theo từng covariance_type."""
    minima = []
    valid = records[records["valid"]]
    for covariance_type, curve in valid.groupby("covariance_type"):
        curve = curve.sort_values("n_components").reset_index(drop=True)
        for i in range(1, len(curve) - 1):
            if (curve.loc[i, "bic"] < curve.loc[i - 1, "bic"]
                    and curve.loc[i, "bic"] < curve.loc[i + 1, "bic"]):
                minima.append({"n_components": int(curve.loc[i, "n_components"]),
                               "covariance_type": covariance_type,
                               "bic": float(curve.loc[i, "bic"])})
    return minima


def select_covariance(records, n_components, rule=config.GMM_COVARIANCE_RULE):
    """Chọn covariance_type tại số thành phần cố định.

    rule="silhouette": Silhouette cao nhất (BIC phá hòa).
    rule="bic": BIC thấp nhất.
    """
    candidates = records[
        (records["n_components"] == n_components) & records["valid"]
    ]
    if candidates.empty:
        raise RuntimeError(f"Không có cấu hình GMM hợp lệ tại K={n_components}.")
    if rule == "silhouette":
        chosen = candidates.sort_values(
            ["silhouette", "bic"], ascending=[False, True]
        ).iloc[0]
    elif rule == "bic":
        chosen = candidates.sort_values("bic").iloc[0]
    else:
        raise ValueError(f"Quy tắc không hỗ trợ: {rule}")
    bic_best = candidates.sort_values("bic").iloc[0]
    return {
        "n_components": int(chosen["n_components"]),
        "covariance_type": chosen["covariance_type"],
        "rule": rule,
        "bic": float(chosen["bic"]),
        "aic": float(chosen["aic"]),
        "silhouette": float(chosen["silhouette"]),
        "bic_best_covariance_type": bic_best["covariance_type"],
        "bic_best_at_k": float(bic_best["bic"])
    }
