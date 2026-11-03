"""在测试集上生成提交文件。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea_lab.evaluate import load_testset
from fovea_lab.pipeline import FoveaDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    files, ss = load_testset()
    det = FoveaDetector()

    preds = {}
    for p in files:
        r = det.detect(p, keep_intermediates=False)
        iid = os.path.splitext(os.path.basename(p))[0].lstrip("0") or "0"
        preds[int(iid)] = (r.x, r.y)
        print(f"  {os.path.basename(p)}  ->  ({r.x:8.1f}, {r.y:8.1f})")

    rows = []
    for _, row in ss.iterrows():
        # ImageID 形如 "81_Fovea_X"，去掉 _Fovea_X / _Fovea_Y 后缀得到图像号
        iid_str = row["ImageID"].split("_")[0]
        axis = row["ImageID"].rsplit("_", 1)[1]
        iid = int(iid_str)
        x, y = preds[iid]
        rows.append({"ImageID": row["ImageID"], "value": x if axis == "X" else y})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n已写入 {args.out}（{len(out)} 行，{len(preds)} 张图）")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
