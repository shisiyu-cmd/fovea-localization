"""fovea_lab —— 眼底图像黄斑中心凹自动定位（传统图像处理方法）

模块结构
--------
preprocess.py   预处理：FOV 掩膜、光照不均校正、CLAHE、血管掩膜、视场内缩权重
disc.py         视盘（Optic Disc）定位：亮度/饱和度响应 + DoG 盘状增强 + 边缘抑制
macula.py       黄斑检测：多尺度 LoG 暗区响应 + 血管抑制 + 解剖学先验
pipeline.py     完整流程封装：单张图像 -> 黄斑坐标
evaluate.py     评估指标：欧氏误差、命中率、CDF
"""

from .pipeline import FoveaDetector, detect_fovea

__all__ = ["FoveaDetector", "detect_fovea"]
__version__ = "1.0.0"
