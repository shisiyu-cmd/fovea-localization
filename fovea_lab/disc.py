"""视盘（Optic Disc）定位。

视盘是眼底图中最亮的结构，颜色偏白（低饱和度），呈直径约 1/8 视场大小的盘状。
检测思路：

1. **亮度 + 低饱和度响应**：视盘比周围视网膜亮，且因巩膜反光而偏白（饱和度低），
   而血管、黄斑、背景都偏红（饱和度高）。用二者加权相减得到初响应。

2. **盘状结构增强（DoG）**：单纯找最亮像素会被反光点和视场边缘的亮环吸引。
   视盘是一个有确定尺寸的「亮团块」，因此用高斯差分（DoG，即 LoG 的近似）
   增强「中心亮、周围暗、尺寸匹配」的结构：
       DoG = G_σ * I − G_{kσ} * I
   当 σ 与视盘半径匹配时，盘中心响应最强，而大面积的渐变背景响应为 0。

3. **视场内缩权重**：视场圆周上常有一圈拍摄伪影，比视盘还亮。
   用距离变换得到「到边界的距离」，贴边区域的响应直接压制为 0。
   这是纯几何约束，不依赖图像内容，非常稳健。

为什么要先找视盘：
黄斑中心凹与视盘有稳定的解剖学关系——两者近似位于同一水平线上，
水平相距约 2.3 个视盘半径，垂直偏移约 +0.4 个半径，位于视盘的一侧。
利用这一先验可把搜索范围从整幅图缩小到一个很小的窗口。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .preprocess import interior_weight

__all__ = ["locate_optic_disc", "DiscResult"]


class DiscResult:
    """视盘定位结果。坐标均为工作分辨率下的像素坐标。"""

    __slots__ = ("x", "y", "radius", "response", "side")

    def __init__(self, x: float, y: float, radius: float, response: float):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.response = float(response)
        self.side = 0  # 黄斑位于视盘的哪一侧：-1 左，+1 右，0 未定

    def __repr__(self) -> str:
        return (f"DiscResult(x={self.x:.1f}, y={self.y:.1f}, "
                f"r={self.radius:.1f}, side={self.side})")


def _norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把视场内的取值归一化到 [0, 1]。"""
    v = x[mask]
    if v.size == 0:
        return x
    lo, hi = float(v.min()), float(v.max())
    return (x - lo) / max(hi - lo, 1e-6)


def locate_optic_disc(rgb: np.ndarray, mask: np.ndarray,
                      smooth: float = 10.0,
                      sat_weight: float = 1.0,
                      dog_weight: float = 1.0,
                      margin: float = 0.18,
                      refine: bool = True) -> DiscResult:
    """定位视盘中心。

    参数
    ----
    rgb : (H, W, 3) float32，0~255
    mask : (H, W) bool，视场掩膜
    smooth : 亮/饱和度响应的平滑尺度
    sat_weight : 饱和度惩罚权重。视盘偏白（低饱和），血管/背景偏红（高饱和）
    dog_weight : 盘状结构增强项权重
    margin : 视场内缩比例，用于压制边缘伪影
    refine : 是否用局部亮区质心细化中心与半径
    """
    a = rgb.astype(np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)

    r_eff = float(np.sqrt(mask.sum() / np.pi)) if mask.any() else 100.0

    # --- 项 1：亮且不饱和 ---
    bright_n = _norm(ndi.gaussian_filter(lum, smooth), mask)
    white_n = _norm(ndi.gaussian_filter(sat, smooth), mask)
    resp = bright_n - sat_weight * white_n

    # --- 项 2：盘状结构（DoG）---
    # 内尺度对应视盘半径量级，外尺度为其 2.2 倍，形成「亮团块」选择性响应
    sigma_in = max(3.0, 0.30 * r_eff * 0.42)
    sigma_out = sigma_in * 2.2
    dog = (ndi.gaussian_filter(lum, sigma_in)
           - ndi.gaussian_filter(lum, sigma_out))
    resp = resp + dog_weight * _norm(dog, mask)

    # --- 项 3：视场内缩权重（压制圆周伪影）---
    iw = interior_weight(mask, margin=margin)
    resp = resp * iw

    resp = ndi.gaussian_filter(resp, max(2.0, smooth / 2))
    resp = np.where(mask & (iw > 0), resp, -np.inf)

    y, x = np.unravel_index(int(np.argmax(resp)), resp.shape)
    peak = float(resp[y, x])

    cx, cy, radius = float(x), float(y), 0.30 * r_eff
    if refine:
        cx, cy, radius = _refine_disc(lum, mask, x, y, initial_r=radius)

    return DiscResult(cx, cy, radius, peak)


def _refine_disc(lum: np.ndarray, mask: np.ndarray, x: int, y: int,
                 initial_r: float = 30.0, window: float = 2.0):
    """在峰值附近用局部高亮区质心细化视盘中心与半径。

    视盘是一片比周围亮的盘状区域，用局部阈值分割后取连通域质心，
    比单纯的响应峰值更稳定（峰值可能落在反光点上）。
    """
    h, w = lum.shape
    r0 = int(max(24, initial_r * window))
    y0, y1 = max(0, y - r0), min(h, y + r0)
    x0, x1 = max(0, x - r0), min(w, x + r0)

    patch = lum[y0:y1, x0:x1]
    pm = mask[y0:y1, x0:x1]
    if pm.sum() < 20:
        return float(x), float(y), initial_r

    thr = np.percentile(patch[pm], 78)
    bright = (patch > thr) & pm
    lab, n = ndi.label(bright)
    if n == 0:
        return float(x), float(y), initial_r

    lid = lab[y - y0, x - x0]
    comp = (lab == lid) if lid > 0 else bright

    area = comp.sum()
    if area < 40:
        return float(x), float(y), initial_r

    cy, cx = ndi.center_of_mass(comp)
    radius = float(np.sqrt(area / np.pi))
    # 半径做合理区间约束，避免细化时被大片亮区带偏
    radius = float(np.clip(radius, 0.6 * initial_r, 1.8 * initial_r))
    return x0 + float(cx), y0 + float(cy), radius
