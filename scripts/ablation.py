"""消融实验：逐项验证各模块的贡献。

对照实验设计（逐步累加，观察每一步的增益）：
  A 基线      : 全图多尺度 LoG 暗区搜索，无任何先验
  B +视盘     : 定位视盘并判断黄斑方向，但不使用空间先验加权
  C +血管抑制 : 在 B 基础上加入血管掩膜抑制
  D 完整      : 在 C 基础上加入解剖学空间先验（最终方法）
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea_lab.evaluate import euclidean_errors, load_dataset, summarize
from fovea_lab.pipeline import FoveaDetector

CONFIGS = [
    ("A 基线（全图搜索）", dict(use_disc=False, use_vessel=False, use_prior=False)),
    ("B +视盘方向判断", dict(use_disc=True, use_vessel=False, use_prior=False)),
    ("C +血管抑制", dict(use_disc=True, use_vessel=True, use_prior=False)),
    ("D 完整（+解剖先验）", dict(use_disc=True, use_vessel=True, use_prior=True)),
]


def main():
    files, gt = load_dataset()
    gtxy = gt[["Fovea_X", "Fovea_Y"]].to_numpy(float)

    rows = []
    for name, kw in CONFIGS:
        det = FoveaDetector(**kw)
        t0 = time.time()
        preds = np.array([(lambda r: (r.x, r.y))(det.detect(p, keep_intermediates=False))
                          for p in files])
        dt = time.time() - t0
        errs = euclidean_errors(preds, gtxy)
        s = summarize(errs)
        rows.append(dict(name=name, **s, time=dt))
        print(f"{name:<22} mean={s['mean']:7.1f}  median={s['median']:6.1f}  "
              f"<50={s['acc@50']*100:5.1f}%  <100={s['acc@100']*100:5.1f}%  "
              f"p90={s['p90']:7.1f}  ({dt:.1f}s)")

    print("\n=== 逐步增益 ===")
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        print(f"  {cur['name']:<22} "
              f"平均误差 {prev['mean']:7.1f} -> {cur['mean']:7.1f} "
              f"({(prev['mean']-cur['mean'])/prev['mean']*100:+5.1f}%)   "
              f"<50px {(cur['acc@50']-prev['acc@50'])*100:+5.1f} pp")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "ablation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
