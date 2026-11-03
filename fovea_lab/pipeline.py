"""完整流程封装：单张眼底图像 -> 黄斑中心凹坐标。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .disc import locate_optic_disc
from .macula import detect_macula
from .preprocess import correct_illumination, extract_fov_mask, to_gray

__all__ = ["FoveaDetector", "detect_fovea", "DetectionResult"]


@dataclass
class DetectionResult:
    """一次检测的全部中间结果，便于可视化与调试。"""

    x: float                       # 原始分辨率下的黄斑 x
    y: float                       # 原始分辨率下的黄斑 y
    width: int
    height: int
    disc_x: float = 0.0
    disc_y: float = 0.0
    disc_r: float = 0.0
    side: int = 0
    intermediates: dict = field(default_factory=dict)

    @property
    def relative(self) -> tuple[float, float]:
        return self.x / self.width, self.y / self.height


class FoveaDetector:
    """黄斑中心凹检测器。

    参数
    ----
    work_width : 内部工作分辨率宽度。原图宽约 2000~3000 px，
        缩到 512 既保留了黄斑（直径约 30~60 px）的尺度特征，
        又把计算量降低约 20 倍。
    use_disc : 是否使用视盘先验。False 时退化为全图搜索（基线方法）。
    use_vessel : 是否启用血管抑制。
    use_prior : 是否启用解剖学空间先验。
    """

    def __init__(self, work_width: int = 512, use_disc: bool = True,
                 use_vessel: bool = True, use_prior: bool = True,
                 scales=(35, 50, 70), prior_weight: float = 0.5):
        self.work_width = work_width
        self.use_disc = use_disc
        self.use_vessel = use_vessel
        self.use_prior = use_prior
        self.scales = scales
        self.prior_weight = prior_weight

    # ---------- 图像 IO ----------

    def load(self, path_or_array) -> tuple[np.ndarray, float, tuple[int, int]]:
        """读取并缩放到工作分辨率。返回 (rgb, scale, 原始尺寸)。"""
        if isinstance(path_or_array, np.ndarray):
            rgb0 = path_or_array
            h0, w0 = rgb0.shape[:2]
        else:
            im = Image.open(path_or_array).convert("RGB")
            w0, h0 = im.size
            rgb0 = np.asarray(im, dtype=np.float32)

        s = self.work_width / float(w0)
        w = self.work_width
        h = max(1, int(round(h0 * s)))
        im = Image.fromarray(rgb0.astype(np.uint8)).resize((w, h), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32), s, (w0, h0)

    # ---------- 主流程 ----------

    def detect(self, path_or_array, keep_intermediates: bool = True) -> DetectionResult:
        rgb, s, (w0, h0) = self.load(path_or_array)
        gray = to_gray(rgb)
        mask = extract_fov_mask(gray)

        # 绿通道对黄斑/血管的对比度最好，先做光照校正
        green = correct_illumination(rgb[..., 1], sigma=40.0)

        inter = {}
        if keep_intermediates:
            inter.update(rgb=rgb, gray=gray, mask=mask, green=green)

        if self.use_disc:
            disc = locate_optic_disc(rgb, mask)
            side = self._infer_side(green, mask, disc)
            disc.side = side
            res = detect_macula(
                green, mask, disc.x, disc.y, disc.radius, side,
                scales=self.scales,
                use_vessel=self.use_vessel,
                use_prior=self.use_prior,
                prior_weight=self.prior_weight,
            )
            mx, my = res.x, res.y
        else:
            # 基线模式：不定位视盘，全图搜索，无任何解剖学约束
            from .macula import _multi_scale_dark_response
            disc = None
            side = 0
            resp = _multi_scale_dark_response(green, mask, self.scales)
            resp = np.where(mask, resp, -np.inf)
            y, x = np.unravel_index(int(np.argmax(resp)), resp.shape)
            mx, my = float(x), float(y)
            res = None

        if keep_intermediates:
            if res is not None:
                inter.update(score_map=res.score_map, prior_map=res.prior_map)
            if disc is not None:
                inter.update(disc_x=disc.x, disc_y=disc.y, disc_r=disc.radius)

        return DetectionResult(
            x=mx / s, y=my / s, width=w0, height=h0,
            disc_x=(disc.x / s) if disc else 0.0,
            disc_y=(disc.y / s) if disc else 0.0,
            disc_r=(disc.radius / s) if disc else 0.0,
            side=side,
            intermediates=inter,
        )

    # ---------- 辅助 ----------

    @staticmethod
    def _infer_side(green: np.ndarray, mask: np.ndarray, disc) -> int:
        """判断黄斑在视盘的哪一侧。

        方法：以视盘为中心，分别向左右各取一条水平带（高度 = 1.2 个视盘半径，
        长度 = 1.5~4 个视盘半径），比较两侧「局部暗度」的总和。
        黄斑所在的一侧整体更暗。

        该判断在训练集上准确率约 91%——因为视盘-黄斑的相对方向由解剖结构决定，
        而左右眼的差异恰好体现在视盘位于图像左侧还是右侧。
        """
        h, w = green.shape
        r = disc.radius
        y0 = int(max(0, disc.y - 0.6 * r))
        y1 = int(min(h, disc.y + 0.6 * r))
        if y1 <= y0:
            return -1

        def darkness(x_from, x_to):
            xa = int(max(0, min(w, x_from)))
            xb = int(max(0, min(w, x_to)))
            if xb <= xa:
                return 0.0
            patch = green[y0:y1, xa:xb]
            pm = mask[y0:y1, xa:xb]
            if pm.sum() < 20:
                return 0.0
            ref = np.median(green[mask])
            return float(np.clip(ref - patch, 0, None)[pm].sum())

        left = darkness(disc.x - 4.0 * r, disc.x - 1.4 * r)
        right = darkness(disc.x + 1.4 * r, disc.x + 4.0 * r)
        return -1 if left >= right else 1


def detect_fovea(path, **kwargs) -> DetectionResult:
    """便捷函数：一行完成检测。"""
    keep = kwargs.pop("keep_intermediates", False)
    return FoveaDetector(**kwargs).detect(path, keep_intermediates=keep)
