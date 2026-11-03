"""生成报告图1（流程各阶段）——用 PIL 精确拼接，避免 matplotlib 布局问题。"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea_lab.disc import locate_optic_disc
from fovea_lab.evaluate import load_dataset
from fovea_lab.macula import _multi_scale_dark_response
from fovea_lab.pipeline import FoveaDetector
from fovea_lab.preprocess import (correct_illumination, estimate_vessel_mask,
                                  extract_fov_mask, to_gray)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def get_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def to_rgb_img(arr, cmap=None, vmin=None, vmax=None):
    """把任意二维/三维数组转成 uint8 RGB 图像。"""
    a = np.asarray(arr)
    if a.ndim == 3:
        return Image.fromarray(a.astype(np.uint8))
    a = a.astype(np.float32)
    a = np.where(np.isfinite(a), a, np.nan)
    if vmin is None:
        vmin = np.nanmin(a)
    if vmax is None:
        vmax = np.nanmax(a)
    if vmax - vmin < 1e-9:
        norm = np.zeros_like(a)
    else:
        norm = (a - vmin) / (vmax - vmin)
    norm = np.nan_to_num(np.clip(norm, 0, 1), nan=0.0)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    rgba = (colormaps[cmap or "gray"](norm) * 255).astype(np.uint8)
    return Image.fromarray(rgba[..., :3])


def main():
    files, gt = load_dataset()
    gtxy = gt[["Fovea_X", "Fovea_Y"]].to_numpy(float)
    det = FoveaDetector()

    idx = 32
    p = files[idx]
    r = det.detect(p, keep_intermediates=True)

    rgb, s, (w0, h0) = det.load(p)
    gray = to_gray(rgb)
    mask = extract_fov_mask(gray)
    green = correct_illumination(rgb[..., 1], 40.0)
    disc = locate_optic_disc(rgb, mask)
    vessel = estimate_vessel_mask(green, mask)
    raw_resp = _multi_scale_dark_response(green, mask, det.scales)
    score = r.intermediates["score_map"]
    score = np.where(np.isfinite(score), score, np.nan)

    gx, gy = gtxy[idx] * s

    panels = [
        (to_rgb_img(rgb), "① 输入原图", "orig"),
        (to_rgb_img(mask.astype(np.float32)), "② FOV 视场掩膜", None),
        (to_rgb_img(green), "③ 绿通道 + 光照校正", None),
        (to_rgb_img(raw_resp, "viridis"), "④ 多尺度 LoG 暗区响应", None),
        (to_rgb_img(vessel.astype(np.float32)), "⑤ 血管掩膜（顶帽变换）", None),
        (to_rgb_img(score, "inferno"), "⑥ 先验加权后的最终响应", "orig"),
    ]

    H = 340
    tiles = []
    for img, title, kind in panels:
        w = int(round(img.width * H / img.height))
        im = img.resize((w, H), Image.LANCZOS).convert("RGB")
        if kind == "orig":
            d = ImageDraw.Draw(im)
            sc = w / rgb.shape[1]
            # 视盘
            dx, dy = disc.x * sc, disc.y * sc
            d.ellipse([dx - 9, dy - 9, dx + 9, dy + 9], outline=(0, 255, 255), width=3)
            d.line([dx - 16, dy, dx + 16, dy], fill=(0, 255, 255), width=2)
            d.line([dx, dy - 16, dx, dy + 16], fill=(0, 255, 255), width=2)
            # 真值
            gxp, gyp = gx * sc, gy * sc
            d.ellipse([gxp - 11, gyp - 11, gxp + 11, gyp + 11],
                      outline=(0, 255, 68), width=3)
            # 预测
            pxp, pyp = r.x / s * sc, r.y / s * sc
            d.line([pxp - 12, pyp - 12, pxp + 12, pyp + 12], fill=(255, 0, 0), width=3)
            d.line([pxp - 12, pyp + 12, pxp + 12, pyp - 12], fill=(255, 0, 0), width=3)
        tiles.append((im, title))

    # 2 行 × 3 列布局，让每幅图在 A4 页宽下足够大
    cols, rows = 3, 2
    gap = 18
    title_h = 52
    tile_w = max(t.width for t, _ in tiles)
    cell_w = tile_w
    cell_h = H + title_h
    total_w = cols * cell_w + gap * (cols + 1)
    total_h = rows * cell_h + gap * (rows + 1)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    font = get_font(26)

    for k, (im, title) in enumerate(tiles):
        c, rw = k % cols, k // cols
        x0 = gap + c * (cell_w + gap)
        y0 = gap + rw * (cell_h + gap)
        d.text((x0 + cell_w / 2, y0), title, fill=(20, 20, 20), font=font, anchor="ma")
        canvas.paste(im, (x0 + (cell_w - im.width) // 2, y0 + title_h))

    canvas.save(os.path.join(OUT, "fig1_pipeline.png"))
    print("fig1 done ->", canvas.size)

    # 图例（竖排时单独说明）
    lf = get_font(24)
    leg = Image.new("RGB", (total_w, 54), (255, 255, 255))
    dl = ImageDraw.Draw(leg)
    items = [((0, 255, 255), "视盘定位"), ((0, 255, 68), "黄斑真值"), ((255, 0, 0), "算法预测")]
    cx = gap + 20
    for col, label in items:
        dl.ellipse([cx, 16, cx + 18, 34], outline=col, width=3)
        dl.text((cx + 28, 13), label, fill=(30, 30, 30), font=lf)
        cx += 28 + int(dl.textlength(label, font=lf)) + 50
    leg.save(os.path.join(OUT, "fig1_legend.png"))
    print("fig1 legend done")


if __name__ == "__main__":
    main()
