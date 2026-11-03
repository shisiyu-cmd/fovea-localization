"""生成报告所需图表：
  fig1_pipeline.png    方法流程图（各阶段中间结果）
  fig2_cdf.png         误差累积分布曲线（基线 vs 完整）
  fig3_ablation.png    消融实验柱状图
  fig4_qualitative.png 定性结果对比（含成功与失败案例）
  fig5_scatter.png     预测值散点 vs 真值
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea_lab.evaluate import cdf_points, euclidean_errors, load_dataset
from fovea_lab.macula import _multi_scale_dark_response
from fovea_lab.pipeline import FoveaDetector
from fovea_lab.preprocess import estimate_vessel_mask, to_gray, extract_fov_mask, correct_illumination
from fovea_lab.disc import locate_optic_disc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

files, gt = load_dataset()
gtxy = gt[["Fovea_X", "Fovea_Y"]].to_numpy(float)

print("计算基线…")
det_base = FoveaDetector(use_disc=False, use_vessel=False, use_prior=False)
preds_base = np.array([(lambda r: (r.x, r.y))(det_base.detect(p, False)) for p in files])
errs_base = euclidean_errors(preds_base, gtxy)

print("计算完整方法…")
det_full = FoveaDetector()
results = []
for p in files:
    r = det_full.detect(p, keep_intermediates=True)
    results.append(r)
preds_full = np.array([(r.x, r.y) for r in results])
errs_full = euclidean_errors(preds_full, gtxy)

print(f"基线 mean={errs_base.mean():.1f}  完整 mean={errs_full.mean():.1f}")

# ============ 图1：流程各阶段 ============
idx = 32  # 0-based -> #33
r = results[idx]
p = files[idx]
rgb, s, (w0, h0) = det_full.load(p)
gray = to_gray(rgb)
mask = extract_fov_mask(gray)
green = correct_illumination(rgb[..., 1], 40.0)
disc = locate_optic_disc(rgb, mask)
vessel = estimate_vessel_mask(green, mask)
raw_resp = _multi_scale_dark_response(green, mask, det_full.scales)
gx, gy = gtxy[idx] * s

fig = plt.figure(figsize=(24, 4.6))
axes = [fig.add_subplot(1, 6, i + 1) for i in range(6)]
panels = [
    (rgb.astype(np.uint8), "① 输入原图", None),
    (mask, "② FOV 视场掩膜", "gray"),
    (green, "③ 绿通道+光照校正", "gray"),
    (raw_resp, "④ 多尺度 LoG 暗区响应", "viridis"),
    (vessel, "⑤ 血管掩膜（顶帽变换）", "gray"),
    (np.where(np.isfinite(r.intermediates["score_map"]), r.intermediates["score_map"], np.nan),
     "⑥ 先验加权后的最终响应", "inferno"),
]
for ax, (img, title, cm) in zip(axes, panels):
    ax.imshow(img, cmap=cm, aspect="equal")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# 在第1、6幅上标注
for ax_i in (0, 5):
    ax = axes[ax_i]
    ax.plot(disc.x, disc.y, "c*", ms=16, markeredgecolor="k")
    ax.plot(gx, gy, "o", ms=11, mfc="none", mec="#00ff44", mew=2.5)
    ax.plot(r.x, r.y, "+", ms=17, mec="red", mew=3)

axes[0].annotate("视盘", xy=(disc.x, disc.y), xytext=(disc.x + 40, disc.y - 70),
                 color="cyan", fontsize=10, weight="bold",
                 arrowprops=dict(arrowstyle="->", color="cyan", lw=1.4))
axes[0].annotate("真值", xy=(gx, gy), xytext=(gx - 100, gy - 75),
                 color="#00ff44", fontsize=10, weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#00ff44", lw=1.4))
axes[0].annotate("预测", xy=(r.x, r.y), xytext=(r.x - 40, r.y + 85),
                 color="red", fontsize=10, weight="bold",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.4))

fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)

fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=150)
plt.close()
print("fig1 done")

# ============ 图2：CDF ============
fig, ax = plt.subplots(figsize=(7, 5))
for errs, label, color, ls in [
    (errs_base, f"基线方法 (均值 {errs_base.mean():.0f}px)", "#d62728", "--"),
    (errs_full, f"完整方法 (均值 {errs_full.mean():.0f}px)", "#2ca02c", "-"),
]:
    xs, ys = cdf_points(errs, 250, 400)
    ax.plot(xs, ys * 100, label=label, color=color, lw=2.2, ls=ls)
ax.axhline(90, color="gray", lw=0.8, ls=":")
ax.axvline(50, color="gray", lw=0.8, ls=":")
ax.set_xlabel("定位误差（像素）", fontsize=12)
ax.set_ylabel("误差不超过该值的样本比例 (%)", fontsize=12)
ax.set_title("误差累积分布曲线（80 张训练图）", fontsize=13)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 250)
ax.set_ylim(0, 102)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_cdf.png"), bbox_inches="tight")
plt.close()
print("fig2 done")

# ============ 图3：消融 ============
abl_path = os.path.join(OUT, "ablation.json")
if os.path.exists(abl_path):
    abl = json.load(open(abl_path, encoding="utf-8"))
    names = [a["name"] for a in abl]
    means = [a["mean"] for a in abl]
    acc50 = [a["acc@50"] * 100 for a in abl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ["#999999", "#7f9fd0", "#e0a03c", "#2ca02c"]
    bars = ax1.bar(range(len(names)), means, color=colors[:len(names)])
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace("（", "\n（") for n in names], fontsize=9)
    ax1.set_ylabel("平均定位误差（像素）", fontsize=11)
    ax1.set_title("各模块对平均误差的影响（越低越好）", fontsize=12)
    for b, v in zip(bars, means):
        ax1.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}",
                 ha="center", fontsize=10, weight="bold")
    ax1.set_ylim(0, max(means) * 1.18)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(range(len(names)), acc50, color=colors[:len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace("（", "\n（") for n in names], fontsize=9)
    ax2.set_ylabel("命中率 @50px (%)", fontsize=11)
    ax2.set_title("各模块对成功率的影响（越高越好）", fontsize=12)
    for b, v in zip(bars2, acc50):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                 ha="center", fontsize=10, weight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_ablation.png"), bbox_inches="tight")
    plt.close()
    print("fig3 done")

# ============ 图4：定性对比 ============
order = np.argsort(-errs_full)
picks = list(order[:3]) + list(order[len(order)//2 - 1: len(order)//2 + 2]) + list(order[-3:])

fig, axes = plt.subplots(3, 6, figsize=(24, 12))
for col, i in enumerate(picks[:6]):
    for row, (preds, title) in enumerate([
        (preds_base, "基线方法"),
        (preds_full, "完整方法"),
    ]):
        im = Image.open(files[i]).convert("RGB")
        w, h = im.size
        im.thumbnail((420, 420))
        sx, sy = im.size[0] / w, im.size[1] / h
        ax = axes[row, col]
        ax.imshow(im)
        gx2, gy2 = gtxy[i] * [sx, sy]
        px, py = preds[i] * [sx, sy]
        ax.plot(gx2, gy2, "o", ms=11, mfc="none", mec="#00ff44", mew=2.5)
        ax.plot(px, py, "+", ms=16, mec="red", mew=3)
        e = np.hypot(*(preds[i] - gtxy[i]))
        ax.set_title(f"#{i+1}  {title}  误差 {e:.0f}px", fontsize=10)
        ax.axis("off")
    # 第三行：完整方法的响应图
    ax = axes[2, col]
    sc = results[i].intermediates.get("score_map")
    if sc is not None:
        sc = np.where(np.isfinite(sc), sc, np.nan)
        ax.imshow(sc, cmap="inferno")
    ax.set_title(f"#{i+1} 最终响应图", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_qualitative.png"), bbox_inches="tight")
plt.close()
print("fig4 done")

# ============ 图5：散点 ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, comp, title in [(axes[0], 0, "X 坐标"), (axes[1], 1, "Y 坐标")]:
    for preds, label, color, marker in [
        (preds_base, "基线", "#d62728", "x"),
        (preds_full, "完整", "#2ca02c", "o"),
    ]:
        ax.scatter(gtxy[:, comp], preds[:, comp], s=38, alpha=0.7,
                   label=label, color=color, marker=marker, edgecolors="k", linewidths=0.4)
    lo = min(gtxy[:, comp].min(), preds_base[:, comp].min(), preds_full[:, comp].min()) - 50
    hi = max(gtxy[:, comp].max(), preds_base[:, comp].max(), preds_full[:, comp].max()) + 50
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="理想 y=x")
    ax.set_xlabel(f"真值 {title}（像素）", fontsize=11)
    ax.set_ylabel(f"预测 {title}（像素）", fontsize=11)
    ax.set_title(f"{title}：预测 vs 真值", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_scatter.png"), bbox_inches="tight")
plt.close()
print("fig5 done")

# 保存指标
metrics = {
    "base": {k: float(v) for k, v in
             __import__("fovea_lab.evaluate", fromlist=["summarize"]).summarize(errs_base).items()},
    "full": {k: float(v) for k, v in
             __import__("fovea_lab.evaluate", fromlist=["summarize"]).summarize(errs_full).items()},
}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n指标:", json.dumps(metrics, ensure_ascii=False, indent=2))
