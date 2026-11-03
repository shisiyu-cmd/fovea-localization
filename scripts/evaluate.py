"""训练集评估：对比基线方法与完整方法。

用法：
    python scripts/evaluate.py                  # 两种方法都跑
    python scripts/evaluate.py --method full    # 只跑完整方法
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea_lab.evaluate import euclidean_errors, load_dataset, summarize
from fovea_lab.pipeline import FoveaDetector


def run(detector: FoveaDetector, files, gt, verbose=True):
    preds = np.zeros((len(files), 2), dtype=float)
    t0 = time.time()
    for i, p in enumerate(files):
        r = detector.detect(p, keep_intermediates=False)
        preds[i] = (r.x, r.y)
        if verbose:
            gx, gy = gt.Fovea_X[i], gt.Fovea_Y[i]
            d = float(np.hypot(r.x - gx, r.y - gy))
            print(f"  [{i+1:3d}/{len(files)}] {os.path.basename(p)}  "
                  f"pred=({r.x:7.1f},{r.y:7.1f})  gt=({gx:7.1f},{gy:7.1f})  err={d:6.1f}px")
    dt = time.time() - t0
    return preds, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["base", "full", "both"], default="both")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save", default="", help="结果保存为 npz 的路径")
    args = ap.parse_args()

    files, gt = load_dataset()
    gtxy = gt[["Fovea_X", "Fovea_Y"]].to_numpy(dtype=float)
    print(f"数据集：{len(files)} 张训练图\n")

    results = {}

    if args.method in ("base", "both"):
        print("=" * 78)
        print("基线方法：全图多尺度暗区搜索（无视盘先验、无血管抑制）")
        print("=" * 78)
        det = FoveaDetector(use_disc=False, use_vessel=False, use_prior=False)
        preds, dt = run(det, files, gt, verbose=not args.quiet)
        errs = euclidean_errors(preds, gtxy)
        s = summarize(errs)
        results["base"] = dict(preds=preds, errors=errs, summary=s, time=dt)
        print(f"\n  平均误差 {s['mean']:.1f}px | 中位数 {s['median']:.1f}px | "
              f"<50px {s['acc@50']*100:.0f}% | <100px {s['acc@100']*100:.0f}% | "
              f"耗时 {dt:.1f}s\n")

    if args.method in ("full", "both"):
        print("=" * 78)
        print("完整方法：视盘定位 + 血管抑制 + 解剖学先验 + 多尺度 LoG")
        print("=" * 78)
        det = FoveaDetector()
        preds, dt = run(det, files, gt, verbose=not args.quiet)
        errs = euclidean_errors(preds, gtxy)
        s = summarize(errs)
        results["full"] = dict(preds=preds, errors=errs, summary=s, time=dt)
        print(f"\n  平均误差 {s['mean']:.1f}px | 中位数 {s['median']:.1f}px | "
              f"<50px {s['acc@50']*100:.0f}% | <100px {s['acc@100']*100:.0f}% | "
              f"耗时 {dt:.1f}s\n")

    if "base" in results and "full" in results:
        b = results["base"]["summary"]; f = results["full"]["summary"]
        print("=" * 78)
        print("改进幅度")
        print("=" * 78)
        print(f"  平均误差 : {b['mean']:8.1f}px  ->  {f['mean']:8.1f}px   "
              f"({(b['mean']-f['mean'])/b['mean']*100:+.1f}%)")
        print(f"  中位误差 : {b['median']:8.1f}px  ->  {f['median']:8.1f}px   "
              f"({(b['median']-f['median'])/b['median']*100:+.1f}%)")
        print(f"  成功率<50: {b['acc@50']*100:7.1f}%  ->  {f['acc@50']*100:7.1f}%   "
              f"({(f['acc@50']-b['acc@50'])*100:+.1f} pp)")
        print(f"  成功率<100:{b['acc@100']*100:7.1f}%  ->  {f['acc@100']*100:7.1f}%   "
              f"({(f['acc@100']-b['acc@100'])*100:+.1f} pp)")

    if args.save:
        np.savez(args.save,
                 **{f"{k}_preds": v["preds"] for k, v in results.items()},
                 **{f"{k}_errors": v["errors"] for k, v in results.items()},
                 gt=gtxy)
        print(f"\n已保存到 {args.save}")


if __name__ == "__main__":
    main()
