"""Huấn luyện lại mô hình bằng pipeline của dự án và ghi tham số vào index.html.
Chạy trong thư mục dự án SAU KHI đã chạy main.py (cần results/*.csv):
    python main.py
    python build_web.py
K và covariance_type của GMM được đọc từ kết quả của main.py, không ghi cứng.
"""
import json, re, io, contextlib
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import data_preprocessing as dp
from gmm_model import select_gmm_config_for_k

CSV, HTML = "CC GENERAL.csv", "index.html"
COMPARISON_CSV, GMM_EVAL_CSV = "results/model_comparison.csv", "results/gmm_evaluation.csv"
# Thứ tự biến mà giao diện index.html đang dùng (nhãn ô nhập, chỉ số trung vị).
WEB_FEATURES = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
                "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
RATIO_FEATURES = {"PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"}

# 1. Đọc K và covariance_type từ kết quả main.py.
cmp_ = pd.read_csv(COMPARISON_CSV, encoding="utf-8-sig")
cmp_ = cmp_[cmp_["Comparison_Phase"] == "Sau cải tiến"].set_index("Model")
K = int(cmp_.loc["K-Means", "Number_of_Clusters"])
gmm_eval = pd.read_csv(GMM_EVAL_CSV, encoding="utf-8-sig")
COV = select_gmm_config_for_k({"records": gmm_eval.to_dict("records")}, K)["covariance_type"]

# 2. Kiểm tra index.html có hỗ trợ cấu hình này không.
html = open(HTML, encoding="utf-8").read()
names = re.search(r"const N=\[(.*?)\];", html)
n_names = len(re.findall(r'"[^"]*"', names.group(1))) if names else 0
if n_names != K:
    raise SystemExit(f"index.html có {n_names} tên cụm (const N) nhưng main.py chọn K={K}. "
                     "Cập nhật tên/mô tả cụm (const N, const D) trong index.html trước.")
if COV != "spherical":
    raise SystemExit(f"main.py chọn GMM covariance_type='{COV}', nhưng hàm gm() trong "
                     "index.html chỉ tính được 'spherical'. Cần sửa gm() trước.")

with contextlib.redirect_stdout(io.StringIO()):
    df, Xs, Xp, feats = dp.load_and_preprocess(CSV)
if feats != WEB_FEATURES:
    raise SystemExit(f"Tập biến đã đổi ({feats}); giao diện index.html cần cập nhật.")
raw = pd.read_csv(CSV)
imp = raw[feats].fillna(raw[feats].median())

# 3. Tái tạo tiền xử lý để lấy tham số đưa lên trình duyệt.
LOG = ["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PURCHASES_TRX"]
params, Mx = [], imp.copy()
for c in feats:
    lo, hi = Mx[c].quantile([.01, .99]); Mx[c] = Mx[c].clip(lo, hi)
    lg = c in LOG and Mx[c].skew() > 1
    if lg: Mx[c] = np.log1p(Mx[c])
    params.append(dict(name=c, lo=float(lo), hi=float(hi), log=bool(lg)))
mu, sd = Mx.mean().values, Mx.std(ddof=0).values
Z = (Mx.values - mu) / sd
assert np.abs(Z - Xs.values).max() < 1e-9, "Tiền xử lý không khớp data_preprocessing.py"
pca = PCA(n_components=0.95, svd_solver="full").fit(Z); P = pca.transform(Z)

# 4. Huấn luyện lại với cùng tham số như kmeans_model.py / gmm_model.py.
km = KMeans(n_clusters=K, init="k-means++", n_init=50, max_iter=500, random_state=42).fit(P)
gm = GaussianMixture(n_components=K, covariance_type=COV, reg_covar=1e-4, n_init=10,
                     max_iter=500, init_params="kmeans", tol=1e-3, random_state=42).fit(P)
kl, gl = km.labels_, gm.predict(P)

ct = pd.crosstab(gl, kl).values
r, c = linear_sum_assignment(-ct); gmap = [int(c[list(r).index(j)]) for j in range(K)]

presets = []
for k in range(K):
    idx = np.where(kl == k)[0]
    j = idx[np.argmin(((P[idx] - km.cluster_centers_[k])**2).sum(1))]
    presets.append({f: round(float(imp.loc[j, f]), 2) for f in feats})
sidx = np.random.RandomState(7).choice(len(P), 240, replace=False)
pts = [[round(float(P[i,0]),2), round(float(P[i,1]),2), int(kl[i]), int(gl[i])] for i in sidx]

def row(model): r_ = cmp_.loc[model]; return dict(sil=float(r_.Silhouette), db=float(r_.Davies_Bouldin),
    ch=float(r_.Calinski_Harabasz), ari=float(r_.Stability_ARI_Mean))
M = dict(feats=params, mu=mu.round(6).tolist(), sd=sd.round(6).tolist(),
    pca=dict(components=pca.components_.round(6).tolist(), mean=pca.mean_.round(8).tolist(),
             ev=pca.explained_variance_ratio_.round(4).tolist()),
    km=dict(centers=km.cluster_centers_.round(5).tolist(), sizes=np.bincount(kl).tolist()),
    gmm=dict(weights=gm.weights_.round(6).tolist(), means=gm.means_.round(5).tolist(),
             var=gm.covariances_.round(6).tolist(), sizes=np.bincount(gl).tolist()),
    gmap=gmap, presets=presets, pts=pts,
    mx=[1 if f in RATIO_FEATURES else round(float(raw[f].quantile(.99))) for f in feats],
    medians=[[round(v,2) for v in raw.loc[kl==k, feats].median()] for k in range(K)],
    metrics=dict(kmeans=row("K-Means"), gmm=row("GMM (cùng K)"),
                 agree_ari=float(cmp_.loc["K-Means", "Agreement_ARI"]), n=int(len(raw))))

# 5. Ghi vào index.html (dữ liệu mô hình + dòng mô tả K / số thành phần PCA).
js = json.dumps(M, separators=(",",":"), ensure_ascii=False)
html, n = re.subn(r"const M=\{.*?\};\n", lambda _: "const M=" + js + ";\n", html, count=1, flags=re.S)
assert n == 1, "Không tìm thấy dòng 'const M=...;' trong index.html"
html = re.sub(r"GMM \(K=\d+\)", f"GMM (K={K})", html, count=1)
html = re.sub(r"PCA \d+ thành phần", f"PCA {pca.n_components_} thành phần", html, count=1)
open(HTML, "w", encoding="utf-8").write(html)
print(f"Đã cập nhật {HTML} | K={K}, GMM covariance={COV} | GMM->K-Means: {gmap}")
