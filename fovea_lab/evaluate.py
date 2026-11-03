"""评估指标与数据集读取。"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

__all__ = ["load_dataset", "load_testset", "euclidean_errors", "summarize", "cdf_points"]

DATA_ROOT = os.environ.get(
    "FOVEA_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "detection"),
)


def load_dataset(root: str = DATA_ROOT):
    """读取训练集图像路径与 GT 坐标。返回 (paths, gt_df)。"""
    gt = pd.read_csv(os.path.join(root, "fovea_localization_train_GT.csv"))
    files = sorted(glob.glob(os.path.join(root, "train", "*.jpg")))
    if len(files) != len(gt):
        raise RuntimeError(f"图像数({len(files)})与标注数({len(gt)})不一致")
    return files, gt


def load_testset(root: str = DATA_ROOT):
    """读取测试集图像路径与提交模板。"""
    ss = pd.read_csv(os.path.join(root, "sample_submission.csv"))
    files = sorted(glob.glob(os.path.join(root, "test", "*.jpg")))
    return files, ss


def euclidean_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """预测点与 GT 点的欧氏距离（像素）。形状 (N, 2)，顺序 (x, y)。"""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])


def summarize(errors: np.ndarray, thresholds=(10, 25, 50, 100)) -> dict:
    """汇总统计量：均值、中位数、标准差、分位数与各阈值命中率。"""
    e = np.asarray(errors, dtype=float)
    out = {
        "n": int(e.size),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "std": float(e.std()),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
    }
    for t in thresholds:
        out[f"acc@{t}"] = float((e <= t).mean())
    return out


def cdf_points(errors: np.ndarray, max_err: float = 200.0, n: int = 200):
    """累积分布曲线：返回 (x, y)，y 为误差 <= x 的比例。"""
    e = np.sort(np.asarray(errors, dtype=float))
    xs = np.linspace(0, max_err, n)
    ys = np.searchsorted(e, xs, side="right") / max(e.size, 1)
    return xs, ys
