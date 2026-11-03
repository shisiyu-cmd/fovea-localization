"""预处理模块：视场掩膜、光照不均校正、对比度增强、血管掩膜。

眼底图像的两个主要成像缺陷：
1. 视场（FOV）外是纯黑背景，且圆形视场边缘有拍摄伪影；
2. 光照不均——中心亮、四周暗，且常有反光斑。
这两点会严重干扰后续的暗区检测，因此必须先处理。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    "to_gray",
    "extract_fov_mask",
    "correct_illumination",
    "apply_clahe",
    "estimate_vessel_mask",
    "interior_weight",
]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 亮度。使用 float32 以避免 uint8 溢出。"""
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def extract_fov_mask(gray: np.ndarray, thresh: float = 12.0) -> np.ndarray:
    """提取眼底视场（圆形有效区域）掩膜。

    步骤：阈值 -> 开运算去噪点 -> 闭运算补边缘 -> 填洞 -> 取最大连通域 -> 腐蚀收缩。

    为什么不用霍夫圆检测：本数据中视场并非标准圆（多为椭圆，且有切边），
    霍夫圆对参数敏感且容易失败；连通域方法对形状不敏感，更稳健。
    """
    m = gray > thresh
    m = ndi.binary_opening(m, np.ones((5, 5)))
    m = ndi.binary_closing(m, np.ones((15, 15)))
    m = ndi.binary_fill_holes(m)

    lab, n = ndi.label(m)
    if n > 1:  # 只保留面积最大的连通域，剔除边缘亮斑
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))

    # 腐蚀掉视场边缘的黑色环状伪影
    return ndi.binary_erosion(m, np.ones((9, 9)))


def correct_illumination(channel: np.ndarray, sigma: float = 40.0,
                         eps: float = 1e-3) -> np.ndarray:
    """用「同态滤波」思想做光照不均校正。

    观测模型近似为 I(x,y) = L(x,y) · R(x,y)，其中 L 是缓变的照明分量，
    R 是反映组织真实反射率的快变分量。用大尺度高斯模糊估计 L，再相除即得 R。

    相比直接做直方图均衡，除法校正能保留绝对亮度关系——这一点对本任务很重要，
    因为黄斑正是靠「局部偏暗」这一特征被检测的。
    """
    c = channel.astype(np.float32)
    bg = ndi.gaussian_filter(c, sigma)
    bg = np.maximum(bg, eps)
    return c / bg


def apply_clahe(channel: np.ndarray, clip_limit: float = 2.0,
                tile: int = 8) -> np.ndarray:
    """限制对比度的自适应直方图均衡（CLAHE）。

    CLAHE 在局部小窗口内做直方图均衡，并限制对比度放大倍数，
    避免噪声被过度放大。此处用于增强黄斑区与周围视网膜的微小亮度差异。
    """
    import cv2

    c = channel.astype(np.float32)
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(c)
    c = np.clip((c - lo) / (hi - lo), 0, 1)
    c8 = (c * 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(c8)
    return out.astype(np.float32) / 255.0


def interior_weight(mask: np.ndarray, margin: float = 0.18) -> np.ndarray:
    """视场内部的「距边界距离」权重图，用于抑制边缘伪响应。

    眼底图的视场圆周上常有一圈高亮或高饱和的拍摄伪影（镜头边缘反光、
    遮罩环）。这些伪影在亮度图上的响应往往比视盘还强，是视盘定位失败的主因。

    这里用距离变换计算每个像素到视场边界的距离，归一化到 [0,1] 后做截断：
    距边界小于 margin×视场半径的区域权重快速衰减到 0。
    这是一个与图像内容无关的几何先验——真实解剖结构（视盘、黄斑）
    不可能贴在视场最外圈。
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)

    dist = ndi.distance_transform_edt(mask)
    r_eff = np.sqrt(mask.sum() / np.pi)          # 等效视场半径
    margin_px = max(1.0, margin * r_eff)

    # 平滑的软阈值：距离超过 margin 的部分线性上升到 1
    wgt = np.clip((dist - margin_px) / margin_px, 0.0, 1.0)
    # 做一次高斯平滑，避免权重图出现硬边导致新的假极值
    wgt = ndi.gaussian_filter(wgt.astype(np.float32), max(1.5, 0.02 * r_eff))
    return (wgt * mask).astype(np.float32)


def estimate_vessel_mask(green: np.ndarray, mask: np.ndarray,
                         scale: int = 9, gain: float = 4.0,
                         thresh: float = 0.06,
                         dilate: int = 3) -> np.ndarray:
    """估计视网膜血管掩膜（形态学顶帽变换）。

    血管在绿通道中呈暗细线，宽度约 2~12 px（工作分辨率下）。
    用顶帽变换（原图 - 开运算）提取比周围暗的细小结构：
      开运算会抹去比结构元窄的暗结构，相减后只剩血管与黄斑暗区。
    再做阈值分割、连通域过滤（去小噪点）与膨胀（覆盖血管边缘）。

    返回 True 表示「是血管」，后续暗区检测需要把这些位置排除掉，
    否则血管交叉点会形成比黄斑更强的暗响应。
    """
    g = green.astype(np.float32)
    opened = ndi.grey_opening(g, footprint=np.ones((scale, scale)))
    tophat = g - opened

    # 自适应阈值：按视场内顶帽响应的分布确定
    vals = tophat[mask]
    if vals.size == 0:
        return np.zeros_like(mask, dtype=bool)
    thr = max(thresh, np.percentile(vals, 97) * 0.5)
    vessel = (tophat > thr) & mask

    # 去掉面积过小的连通域（噪声）与过大的（背景残留）
    lab, n = ndi.label(vessel)
    if n > 0:
        areas = ndi.sum(vessel, lab, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        lo, hi = 8, 0.25 * mask.sum()
        keep[1:] = (areas >= lo) & (areas <= hi)
        vessel = keep[lab]

    if dilate > 0:
        vessel = ndi.binary_dilation(vessel, np.ones((dilate, dilate)))
    return vessel & mask
