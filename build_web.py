"""Ghi tham số mô hình đã huấn luyện vào index.html (trang demo chạy trên trình duyệt).

Chạy trong thư mục dự án SAU KHI đã chạy đủ 8 bước trong src/:
    python build_web.py

Không huấn luyện lại: đọc bộ tiền xử lý và mô hình đã lưu trong models/,
dữ liệu đã xử lý trong data/processed/ và bảng so sánh trong results/comparison/.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src import config
from src.evaluation import cluster_profile
from src.segments import SEGMENT_NAMES, segment_order
from src.utils import load_model, load_raw, load_split, require

HTML = config.ROOT / "index.html"
# Thứ tự biến mà giao diện index.html đang dùng (nhãn ô nhập, chỉ số trung vị).
WEB_FEATURES = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
                "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
RATIO_FEATURES = {"PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"}

# 1. Đọc mô hình và kết quả đã lưu.
prep = load_model(config.PREPROCESSOR_PATH, "src/prepare_data.py")
km = load_model(config.KMEANS_MODEL, "src/k_mean/train.py")
gm = load_model(config.GMM_MODEL, "src/gmm/train.py")
comparison_path = config.COMPARISON_RESULTS / "model_comparison.csv"
require(comparison_path, "src/compare_models.py")
cmp_ = pd.read_csv(comparison_path, encoding="utf-8-sig").set_index("Model")
K, COV, feats = km.n_clusters, gm.covariance_type, prep["features"]

# 2. Kiểm tra index.html có hỗ trợ cấu hình này không.
html = HTML.read_text(encoding="utf-8")
names = re.search(r"const N=\[(.*?)\];", html)
html_names = re.findall(r'"([^"]*)"', names.group(1)) if names else []
if html_names != SEGMENT_NAMES or K != len(SEGMENT_NAMES):
    raise SystemExit(f"index.html có {len(html_names)} tên cụm, mô hình có K={K}; "
                     "tên trong const N phải khớp SEGMENT_NAMES (src/segments.py).")
if gm.n_components != K:
    raise SystemExit("K-Means và GMM phải cùng số cụm để hiển thị trên web.")
if COV != "spherical":
    raise SystemExit(f"GMM covariance_type='{COV}', nhưng hàm gm() trong "
                     "index.html chỉ tính được 'spherical'. Cần sửa gm() trước.")
if feats != WEB_FEATURES:
    raise SystemExit(f"Tập biến đã đổi ({feats}); giao diện index.html cần cập nhật.")

# 3. Dự đoán cho toàn bộ khách hàng và sắp lại nhãn K-Means theo tên phân khúc.
raw = load_raw()
P = load_split("all").to_numpy()
km_labels = km.predict(P)
order = segment_order(cluster_profile(raw, km_labels))   # order[i] = cụm của tên i
new_label = np.empty(K, dtype=int)
new_label[order] = np.arange(K)
kl = new_label[km_labels]                                # nhãn theo thứ tự tên
centers = km.cluster_centers_[order]
gl = gm.predict(P)

ct = pd.crosstab(gl, kl).reindex(index=range(K), columns=range(K), fill_value=0).values
r, c = linear_sum_assignment(-ct)
gmap = [int(c[list(r).index(j)]) for j in range(K)]      # cụm GMM -> phân khúc

# 4. Tham số tiền xử lý cho trình duyệt.
params = [dict(name=f, lo=prep["bounds"][f][0], hi=prep["bounds"][f][1],
               log=f in prep["log_columns"]) for f in feats]
scaler, pca = prep["scaler"], prep["pca"]
imp = raw[feats].fillna(prep["medians"])

presets = []
for k in range(K):
    idx = np.where(kl == k)[0]
    j = idx[np.argmin(((P[idx] - centers[k]) ** 2).sum(1))]
    presets.append({f: round(float(imp.iloc[j][f]), 2) for f in feats})
sidx = np.random.RandomState(7).choice(len(P), 240, replace=False)
pts = [[round(float(P[i, 0]), 2), round(float(P[i, 1]), 2), int(kl[i]), int(gl[i])]
       for i in sidx]


def row(model):
    r_ = cmp_.loc[model]
    return dict(sil=float(r_.Silhouette), db=float(r_.Davies_Bouldin),
                ch=float(r_.Calinski_Harabasz), ari=float(r_.Stability_ARI_Mean))


M = dict(feats=params, mu=scaler.mean_.round(6).tolist(), sd=scaler.scale_.round(6).tolist(),
    pca=dict(components=pca.components_.round(6).tolist(), mean=pca.mean_.round(8).tolist(),
             ev=pca.explained_variance_ratio_.round(4).tolist()),
    km=dict(centers=centers.round(5).tolist(), sizes=np.bincount(kl, minlength=K).tolist()),
    gmm=dict(weights=gm.weights_.round(6).tolist(), means=gm.means_.round(5).tolist(),
             var=gm.covariances_.round(6).tolist(), sizes=np.bincount(gl, minlength=K).tolist()),
    gmap=gmap, presets=presets, pts=pts,
    mx=[1 if f in RATIO_FEATURES else round(float(raw[f].quantile(.99))) for f in feats],
    medians=[[round(v, 2) for v in raw.loc[kl == k, feats].median()] for k in range(K)],
    metrics=dict(kmeans=row("K-Means"), gmm=row("GMM"),
                 agree_ari=float(cmp_.loc["K-Means", "Agreement_ARI"]), n=int(len(raw))))

# 5. Ghi vào index.html (dữ liệu mô hình + dòng mô tả K / số thành phần PCA).
js = json.dumps(M, separators=(",", ":"), ensure_ascii=False)
html, n = re.subn(r"const M=\{.*?\};\n", lambda _: "const M=" + js + ";\n", html, count=1, flags=re.S)
assert n == 1, "Không tìm thấy dòng 'const M=...;' trong index.html"
html = re.sub(r"GMM \(K=\d+\)", f"GMM (K={K})", html, count=1)
html = re.sub(r"PCA \d+ thành phần", f"PCA {pca.n_components_} thành phần", html, count=1)
HTML.write_text(html, encoding="utf-8")
print(f"Đã cập nhật {HTML.name} | K={K}, GMM covariance={COV}")
print("Phân khúc -> cụm K-Means gốc:",
      {SEGMENT_NAMES[i]: int(order[i]) for i in range(K)})
print("Cụm GMM -> phân khúc:", gmap)
