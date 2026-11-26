"""Đặt tên phân khúc nghiệp vụ cho các cụm dựa trên hồ sơ hành vi.

Số thứ tự cụm do thuật toán sinh ra là ngẫu nhiên (đổi khi dữ liệu train đổi),
nên tên phân khúc được gán theo quy tắc trên trung bình biến gốc của cụm,
không theo số thứ tự. Quy tắc áp dụng khi có đúng 5 cụm:
  1. PURCHASES_FREQUENCY thấp nhất           -> Chuyên rút tiền mặt
  2. PRC_FULL_PAYMENT cao nhất (còn lại)      -> Chi tiêu chủ động, trả đủ nợ
  3. CASH_ADVANCE cao nhất (còn lại)          -> Hạn mức cao, mua sắm và rút tiền
  4. PURCHASES thấp nhất (còn lại)            -> Ít hoạt động
  5. Cụm còn lại                              -> Chi tiêu đều đặn
"""
# Thứ tự này khớp với const N / const D trong index.html.
SEGMENT_NAMES = [
    "Ít hoạt động",
    "Chi tiêu đều đặn",
    "Hạn mức cao, mua sắm và rút tiền",
    "Chuyên rút tiền mặt",
    "Chi tiêu chủ động, trả đủ nợ"
]

_RULES = [  # (tên phân khúc, biến, chọn lớn nhất?)
    ("Chuyên rút tiền mặt", "PURCHASES_FREQUENCY", False),
    ("Chi tiêu chủ động, trả đủ nợ", "PRC_FULL_PAYMENT", True),
    ("Hạn mức cao, mua sắm và rút tiền", "CASH_ADVANCE", True),
    ("Ít hoạt động", "PURCHASES", False),
]


def segment_order(profile):
    """Trả về danh sách cluster id theo thứ tự SEGMENT_NAMES,
    hoặc None nếu số cụm khác 5 (khi đó dùng tên chung "Cụm i").

    profile: DataFrame có cột "Cluster" và trung bình các biến gốc.
    """
    if len(profile) != len(SEGMENT_NAMES):
        return None
    remaining = profile.set_index("Cluster")
    assigned = {}
    for name, column, largest in _RULES:
        cluster = (remaining[column].idxmax() if largest
                   else remaining[column].idxmin())
        assigned[name] = int(cluster)
        remaining = remaining.drop(index=cluster)
    assigned["Chi tiêu đều đặn"] = int(remaining.index[0])
    return [assigned[name] for name in SEGMENT_NAMES]


def segment_names(profile):
    """Ánh xạ {cluster id: tên phân khúc}."""
    order = segment_order(profile)
    if order is None:
        return {int(c): f"Cụm {int(c)}" for c in profile["Cluster"]}
    return {cluster: SEGMENT_NAMES[i] for i, cluster in enumerate(order)}
