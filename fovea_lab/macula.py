"""黄斑（Macula）/ 中心凹（Fovea）检测。

核心观察
--------
黄斑中心凹是视网膜上的一块**局部暗区**：它没有血管（无血管区 FAZ），
色素上皮层较厚，因此在绿通道中呈现为一个直径约 0.3~0.5 个视盘直径的
类圆形暗斑，边缘逐渐过渡。

检测策略
--------
1. **多尺度 LoG 暗斑响应**：对每个尺度 s，计算
       R_s = blur_s(I) - I
   即「大尺度平滑后的亮度」减去「原图亮度」。中心比周围暗的地方 R_s > 0，
   且当 s 与暗斑尺寸匹配时响应最强。多尺度叠加可兼顾不同大小的黄斑。
   这一步等价于用一个「墨西哥帽」小波（LoG 的近似）做斑点检测。

2. **血管抑制**：血管交叉点、血管主干同样是暗结构，且响应往往强于黄斑。
   因此先用顶帽变换提取血管掩膜，在计算响应时把这些像素的贡献抹掉。

3. **解剖学先验约束**：黄斑中心凹位于视盘的一侧，近似同一水平线，
   水平距离约为 2.3 个视盘半径，垂直偏移约 +0.4 个视盘半径。
   用一个各向异性的高斯窗口对响应加权，大幅压缩搜索空间。

4. **亚像素精化**：在响应峰值附近做亮度加权质心，抵消平滑带来的偏移。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .preprocess import estimate_vessel_mask

__all__ = ["detect_macula", "MaculaResult", "anatomical_prior"]


class MaculaResult:
    """黄斑检测结果（工作分辨率坐标）。"""

    __slots__ = ("x", "y", "response", "score_map", "prior_map")

    def __init__(self, x: float, y: float, response: float,
                 score_map: np.ndarray | None = None,
                 prior_map: np.ndarray | None = None):
        self.x = float(x)
        self.y = float(y)
        self.response = float(response)
        self.score_map = score_map
        self.prior_map = prior_map

    def __repr__(self) -> str:
        return f"MaculaResult(x={self.x:.1f}, y={self.y:.1f}, resp={self.response:.4f})"


def anatomical_prior(shape, disc_x: float, disc_y: float, disc_r: float,
                     side: int,
                     dx_ratio: float = 2.3,
                     dy_ratio: float = 0.4,
                     sigma_x: float = 1.3,
                     sigma_y: float = 0.55) -> np.ndarray:
    """构造基于解剖学先验的空间权重图。

    以视盘为原点，向黄斑一侧平移 dx_ratio 个视盘半径，再向下 dy_ratio 个半径，
    得到一个「期望位置」，在其周围放一个各向异性的高斯：
    水平方向允许较大偏差（sigma_x 大），垂直方向限制严格（sigma_y 小）。

    依据来自对本数据 80 张训练图的统计：
      水平偏移 / 视盘半径 ≈ 2.3（标准差较大，因人眼轴长而异）
      垂直偏移 / 视盘半径 ≈ 0.4（标准差仅 0.43 个半径，非常稳定）
    """
    h, w = shape[:2]
    cx = disc_x + side * dx_ratio * disc_r
    cy = disc_y + dy_ratio * disc_r

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = sigma_x * disc_r
    sy = sigma_y * disc_r
    d2 = ((xx - cx) / max(sx, 1e-3)) ** 2 + ((yy - cy) / max(sy, 1e-3)) ** 2
    return np.exp(-0.5 * d2).astype(np.float32)


def _multi_scale_dark_response(green: np.ndarray, mask: np.ndarray,
                               scales=(35, 50, 70)) -> np.ndarray:
    """多尺度「中心比周围暗」响应。

    对每个尺度 s：
        R_s = gaussian_blur(g, s) - g
    R_s 在比 s 小的暗结构中心取正最大值。多尺度取平均而非取最大，
    是为了让响应峰值落在暗区中心而不是某一尺度的边缘伪响应上。
    """
    g = ndi.gaussian_filter(green.astype(np.float32), 1.5)
    resp = np.zeros_like(g)
    for s in scales:
        resp += ndi.gaussian_filter(g, s) - g
    resp /= len(scales)
    resp = ndi.gaussian_filter(resp, 10.0)
    return resp


def detect_macula(green: np.ndarray, mask: np.ndarray,
                  disc_x: float, disc_y: float, disc_r: float,
                  side: int,
                  scales=(35, 50, 70),
                  vessel_gain: float = 1.0,
                  use_vessel: bool = True,
                  use_prior: bool = True,
                  prior_weight: float = 1.0,
                  prior_kwargs: dict | None = None) -> MaculaResult:
    """在给定视盘位置的前提下检测黄斑中心凹。

    参数
    ----
    green : 光照校正后的绿通道图像（float32）
    mask : 视场掩膜
    disc_x, disc_y, disc_r : 视盘中心与半径
    side : 黄斑在视盘哪一侧，-1 左 / +1 右
    use_vessel : 是否启用血管抑制（消融开关）
    use_prior : 是否启用解剖学先验（消融开关）
    """
    resp = _multi_scale_dark_response(green, mask, scales)

    # --- 血管抑制 ---
    if use_vessel:
        vessel = estimate_vessel_mask(green, mask, scale=9, thresh=0.06, dilate=3)
        # 不直接把响应置零（会在血管密集处形成大量假极大值），
        # 而是按比例衰减，并对衰减后的图做一次中值滤波填补
        resp = resp * np.where(vessel, 0.15 * vessel_gain, 1.0)
        resp = ndi.median_filter(resp, size=7)

    # --- 解剖学先验 ---
    prior = None
    score = resp
    if use_prior:
        kw = dict(prior_kwargs or {})
        prior = anatomical_prior(green.shape, disc_x, disc_y, disc_r, side, **kw)
        # 加权而非硬性裁剪：先验弱的区域仍然可能被选中，但需要更强的暗响应
        score = resp * (prior ** prior_weight)

    # 掩膜外统一置为 -inf。必须放在加权之后：-inf * 0 会产生 nan，
    # 而 nan 会让 argmax 返回 (0, 0)，导致预测点跑到图像左上角。
    score = np.where(mask & np.isfinite(score), score, -np.inf)

    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    fx, fy = _refine_centroid(green, mask, x, y, disc_r)

    return MaculaResult(fx, fy, float(resp[y, x]), score_map=score, prior_map=prior)


def _refine_centroid(green: np.ndarray, mask: np.ndarray,
                     x: int, y: int, disc_r: float,
                     radius_ratio: float = 0.30) -> tuple[float, float]:
    """在峰值附近用「暗度加权质心」做亚像素精化。

    高斯平滑会把响应峰值从暗斑真实中心拉向较亮一侧，
    用局部暗度作为权重求质心可以抵消这一偏移。
    """
    h, w = green.shape
    r = max(6, int(round(radius_ratio * disc_r)))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)

    patch = green[y0:y1, x0:x1]
    pm = mask[y0:y1, x0:x1]
    if pm.sum() < 10:
        return float(x), float(y)

    # 以局部均值为参照，越暗权重越大
    ref = np.median(patch[pm])
    wgt = np.clip(ref - patch, 0, None) * pm
    if wgt.sum() < 1e-6:
        return float(x), float(y)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy = float((wgt * yy).sum() / wgt.sum())
    cx = float((wgt * xx).sum() / wgt.sum())
    return cx, cy
