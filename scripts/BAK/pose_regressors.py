#!/usr/bin/env python3
"""
姿态估计：DINOv3 特征 + Ridge 线性热图回归

修复与改进（相比原始版本）:
  1. [Bug]    sigma 参数从未传入 make_heatmap，始终使用默认值
  2. [Bug]    patch 坐标用左上角 (i,j) 而非中心 (i+0.5,j+0.5)，
              导致 make_heatmap 与 soft-argmax 存在系统性 0.5 patch 偏移
  3. [设计]   热图极度稀疏（~0.4% 正样本），Ridge MSE 退化为预测全零
              → 引入正负 patch 平衡采样
  4. [设计]   缺少空间位置信息，左右对称关节难以区分
              → 拼接归一化 patch 中心坐标作为位置编码
  5. [数值]   Ridge cholesky solver 法方程病态（rcond=1.6e-9）
              → 换用 lsqr solver，不显式构造 X^T X
  6. [内存]   原始方案先存全图特征再 vstack，峰值 ~18 GB
              → 流式提取 + 即时采样，峰值显著降低
"""

import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge
from tqdm import tqdm

from dinov3_utils import (
    load_model, resize_transform, extract_features,
    get_patch_grid_size, PATCH_SIZE,
)

# ============================================================
# 关键点定义（与 pose_check.py 完全一致）
# ============================================================
KPT_MAP = [43, 37, 1, 41, 39, 34, 33, 35, 32, 36, 31, 28, 27, 29, 26, 30, 25]

JOINT_NAMES = [
    "head", "neck", "thorax", "spine", "pelvis",
    "lshoulder", "rshoulder", "lelbow", "relbow",
    "lwrist", "rwrist", "lhip", "rhip",
    "lknee", "rknee", "lankle", "rankle",
]

KINTREE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (5, 7), (7, 9),
    (2, 6), (6, 8), (8, 10),
    (4, 11), (11, 13), (13, 15),
    (4, 12), (12, 14), (14, 16),
]

_C, _L, _R  = (200, 200, 200), (0, 200, 0), (0, 200, 255)
BONE_COLORS = [_C, _C, _C, _C, _L, _L, _L, _R, _R, _R, _L, _L, _L, _R, _R, _R]
N_JOINTS    = len(JOINT_NAMES)


# ============================================================
# 数据加载
# ============================================================

def find_data_pairs(folder, max_samples=None):
    """扫描 {folder}/h36m-train/ 下的 (jpg, pyd) 对"""
    train_dir = Path(folder) / "h36m-train"
    if not train_dir.exists():
        return []
    pairs = []
    for pyd in sorted(train_dir.glob("*.data.pyd")):
        jpg = pyd.with_suffix("").with_suffix(".jpg")
        if jpg.exists():
            pairs.append((jpg, pyd))
            if max_samples and len(pairs) >= max_samples:
                break
    return pairs


def load_kpts17(pyd_path):
    """返回 (17, 3) 关键点 [x, y, vis]"""
    with open(pyd_path, "rb") as f:
        data = pickle.load(f)
    return data[0]["keypoints_2d"][KPT_MAP]  # (17, 3)


# ============================================================
# 特征处理
# ============================================================

def add_pos_enc(feats, h_p, w_p):
    """拼接归一化 patch 中心坐标，(N, D) -> (N, D+2)"""
    pi  = (np.repeat(np.arange(h_p), w_p) + 0.5) / h_p  # 行
    pj  = (np.tile(np.arange(w_p), h_p)   + 0.5) / w_p  # 列
    pos = np.stack([pi, pj], axis=1).astype(np.float32)
    return np.concatenate([feats, pos], axis=1)


def process_image(model, img, model_name):
    """提取 DINOv3 特征并拼接位置编码

    Returns:
        feats   : (N_patches, D+2)
        h_p, w_p: patch 网格尺寸
        orig_w, orig_h: 原始图像尺寸
    """
    orig_w, orig_h = img.size
    img_tensor     = resize_transform(img)
    h_p, w_p       = get_patch_grid_size(img_tensor)
    feats          = extract_features(model, img_tensor, model_name).numpy()
    return add_pos_enc(feats, h_p, w_p), h_p, w_p, orig_w, orig_h


# ============================================================
# 热图 & 采样
# ============================================================

def _patch_grid(h_p, w_p):
    """返回 patch 中心行列坐标 (N,), (N,)，行主序"""
    pi = np.repeat(np.arange(h_p), w_p).astype(np.float32) + 0.5
    pj = np.tile(np.arange(w_p), h_p).astype(np.float32)   + 0.5
    return pi, pj


def make_heatmap(kpt, h_p, w_p, orig_w, orig_h, sigma):
    """单关键点高斯热图 (N_patches,)

    坐标系: patch 中心 (i+0.5, j+0.5) 与 soft-argmax 保持严格一致。
    kpt: [x, y, vis]，原始图像像素坐标。
    """
    sx, sy   = (w_p * PATCH_SIZE) / orig_w, (h_p * PATCH_SIZE) / orig_h
    pj_k     = kpt[0] * sx / PATCH_SIZE   # 关键点在 patch 坐标系的列位置
    pi_k     = kpt[1] * sy / PATCH_SIZE   # 关键点在 patch 坐标系的行位置
    pi, pj   = _patch_grid(h_p, w_p)
    dist2    = (pi - pi_k) ** 2 + (pj - pj_k) ** 2
    heatmap = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)
    # print(heatmap.shape, heatmap.min(), heatmap.max(), h_p, w_p, orig_w, orig_h, PATCH_SIZE)
    # cv2.imshow("heatmap", (heatmap.reshape(h_p, w_p) * 255).astype(np.uint8))
    # cv2.waitKey(0)
    return heatmap


def sample_patches(feats, heatmap, pos_thr, neg_ratio, rng):
    """平衡采样正 (heatmap > pos_thr) / 负 patch

    Returns: (X, y) or (None, None)
    """
    pos_idx = np.where(heatmap > pos_thr)[0]
    if len(pos_idx) == 0:
        return None, None
    neg_idx = np.where(heatmap <= pos_thr)[0]
    n_neg   = min(len(neg_idx), len(pos_idx) * neg_ratio)
    idx     = np.concatenate([pos_idx, rng.choice(neg_idx, n_neg, replace=False)])
    return feats[idx], heatmap[idx]


# ============================================================
# 训练：流式提取 + 采样 → Ridge 回归
# ============================================================

def extract_and_sample(model, pairs, model_name, sigma=2.0, pos_thr=0.2, neg_ratio=3):
    """流式提取特征并按关节平衡采样

    内存高效：每张图特征用完即弃，只积累采样后的 patch。

    Returns:
        per_X, per_y: list[list[ndarray]]，按关节索引组织
    """
    per_X = [[] for _ in range(N_JOINTS)]
    per_y = [[] for _ in range(N_JOINTS)]
    rng   = np.random.default_rng(42)

    for jpg, pyd in tqdm(pairs, desc="extract features"):
        img                              = Image.open(jpg).convert("RGB")
        feats, h_p, w_p, orig_w, orig_h = process_image(model, img, model_name)
        kpts17                           = load_kpts17(pyd)
        
        # cv2.imshow("img", cv2.imread(str(jpg)))
    
        for k in range(N_JOINTS):
            if kpts17[k, 2] <= 0:
                continue
            hm       = make_heatmap(kpts17[k], h_p, w_p, orig_w, orig_h, sigma)
            X_s, y_s = sample_patches(feats, hm, pos_thr, neg_ratio, rng)
            if X_s is not None:
                per_X[k].append(X_s)
                per_y[k].append(y_s)

    return per_X, per_y


def train_regressors(per_X, per_y, alpha=1.0):
    """逐关节训练 Ridge 回归器（lsqr solver，逐关节释放内存）"""
    regressors = []
    for k in range(N_JOINTS):
        Xs, ys         = per_X[k], per_y[k]
        per_X[k] = per_y[k] = None          # 释放引用，允许 GC

        if not Xs:
            print(f"  [{JOINT_NAMES[k]:12s}] no visible samples, skip")
            regressors.append(None)
            continue

        X, y   = np.vstack(Xs), np.concatenate(ys)
        del Xs, ys
        n_pos  = int((y > 0.5).sum())
        print(f"  [{JOINT_NAMES[k]:12s}] X={X.shape} y={y.shape}  pos>0.5={n_pos}")

        reg = Ridge(alpha=alpha, solver="lsqr")
        reg.fit(X, y)
        regressors.append(reg)
        del X, y

    print(f"\n✓ {sum(r is not None for r in regressors)}/{N_JOINTS} regressors trained")
    return regressors


# ============================================================
# 推理：soft-argmax 还原坐标
# ============================================================

def predict_kpts(regressors, feats, h_p, w_p, orig_w, orig_h):
    """soft-argmax 推理

    以 patch 中心坐标 (i+0.5, j+0.5) 做加权均值，再还原至原始坐标系。
    与 make_heatmap 使用同一坐标约定，坐标系严格自洽。

    Returns:
        kpts_pred : (17, 2)，原始图像像素坐标
        valid     : (17,) bool
    """
    sx, sy    = (w_p * PATCH_SIZE) / orig_w, (h_p * PATCH_SIZE) / orig_h
    pi, pj    = _patch_grid(h_p, w_p)
    kpts_pred = np.zeros((N_JOINTS, 2), dtype=np.float32)
    valid     = np.zeros(N_JOINTS, dtype=bool)

    for k, reg in enumerate(regressors):
        if reg is None:
            continue
        h      = np.clip(reg.predict(feats), 0.0, None)
        total  = h.sum() + 1e-8
        kpts_pred[k, 0] = np.clip((pj * h).sum() / total * PATCH_SIZE / sx, 0, orig_w - 1)
        kpts_pred[k, 1] = np.clip((pi * h).sum() / total * PATCH_SIZE / sy, 0, orig_h - 1)
        valid[k] = True

    return kpts_pred, valid


# ============================================================
# 评估：PCK@0.2
# ============================================================

def compute_pck(kpts_pred, kpts17, thr_ratio=0.2):
    """PCK @ thr_ratio（以 neck–pelvis 距离为基准）

    Returns: dict {joint_name: float | None}
    """
    vis   = kpts17[:, 2] > 0
    torso = np.linalg.norm(kpts17[1, :2] - kpts17[4, :2]) + 1e-6
    thr   = thr_ratio * torso
    return {
        JOINT_NAMES[k]: (
            float(np.linalg.norm(kpts_pred[k] - kpts17[k, :2]) < thr) if vis[k] else None
        )
        for k in range(N_JOINTS)
    }


def evaluate(model, regressors, pairs, model_name):
    """批量计算并打印 PCK@0.2，返回整体均值"""
    joint_scores = {n: [] for n in JOINT_NAMES}
    all_pck      = []

    for jpg, pyd in tqdm(pairs, desc="evaluate"):
        img                              = Image.open(jpg).convert("RGB")
        feats, h_p, w_p, orig_w, orig_h = process_image(model, img, model_name)
        kpts17                           = load_kpts17(pyd)
        kpts_pred, _                     = predict_kpts(regressors, feats, h_p, w_p, orig_w, orig_h)
        scores                           = compute_pck(kpts_pred, kpts17)

        vis_vals = [v for v in scores.values() if v is not None]
        if vis_vals:
            all_pck.append(np.mean(vis_vals))
        for name, val in scores.items():
            if val is not None:
                joint_scores[name].append(val)

    if not all_pck:
        print("  no valid samples")
        return 0.0

    sep = "=" * 52
    print(f"\n{sep}\n  PCK@0.2  ({len(all_pck)} images)\n{sep}")
    print(f"  Overall  : {np.mean(all_pck) * 100:.1f}%\n")
    for name in JOINT_NAMES:
        vals = joint_scores[name]
        s    = f"{np.mean(vals) * 100:.1f}%  (n={len(vals)})" if vals else "N/A"
        print(f"  {name:12s}: {s}")
    print(sep)
    return float(np.mean(all_pck))


# ============================================================
# 可视化
# ============================================================

def _draw_skeleton(canvas, kpts, vis_mask, kpt_color):
    """在 canvas 上原地绘制骨架。kpts shape: (N, 2+)"""
    for (i, j), color in zip(KINTREE, BONE_COLORS):
        if vis_mask[i] and vis_mask[j]:
            cv2.line(canvas,
                     tuple(kpts[i, :2].astype(int)),
                     tuple(kpts[j, :2].astype(int)), color, 2)
    for k in range(N_JOINTS):
        if vis_mask[k]:
            cv2.circle(canvas, tuple(kpts[k, :2].astype(int)), 4, kpt_color, -1)


def draw_comparison(img, kpts_pred, kpts17, valid, save_path, pck_score=None):
    """左: GT 骨架 | 右: 预测骨架，保存对比图"""
    bgr      = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gt_img   = bgr.copy()
    pred_img = bgr.copy()

    _draw_skeleton(gt_img,   kpts17,    kpts17[:, 2] > 0, (255, 255, 255))
    _draw_skeleton(pred_img, kpts_pred, valid,             (0, 80, 255))

    combined = np.hstack([gt_img, pred_img])
    W        = bgr.shape[1]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Ground Truth", (10, 30),    font, 0.9, (0, 255, 255), 2)
    cv2.putText(combined, "Prediction",  (W + 10, 30), font, 0.9, (0, 80, 255),  2)
    if pck_score is not None:
        cv2.putText(combined, f"PCK@0.2: {pck_score:.1f}%", (W + 10, 60), font, 0.7, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, combined)
    print(f"  saved: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="DINOv3 Pose Estimation via Ridge Regression")
    p.add_argument("--root",            default="/media/baidu/8C1A3A981A3A7F70/DATAS/h36m/h36m-tars")
    p.add_argument("--model",           default="dinov3_vitl16")
    p.add_argument("--max-train",       type=int,   default=1000)
    p.add_argument("--max-eval",        type=int,   default=200)
    p.add_argument("--save-vis",        type=int,   default=10)
    p.add_argument("--alpha",           type=float, default=1.0,  help="Ridge 正则化系数")
    p.add_argument("--sigma",           type=float, default=5.0,  help="热图高斯半径 (patch 单位)")
    p.add_argument("--pos-thr",         type=float, default=0.2,  help="热图正样本阈值")
    p.add_argument("--neg-ratio",       type=int,   default=3,    help="负:正采样比")
    p.add_argument("--skip-train",      action="store_true")
    p.add_argument("--regressors-path", default="output/pose_regressors.pkl")
    args = p.parse_args()

    os.makedirs("output", exist_ok=True)

    print("loading DINOv3 model...")
    model = load_model(args.model)

    # 测试集
    print("\nscan test data (000312)...")
    test_pairs = find_data_pairs(f"{args.root}/000312",
                                 max_samples=args.max_eval + args.save_vis)
    print(f"  test samples: {len(test_pairs)}")

    # 加载 or 训练
    if args.skip_train and Path(args.regressors_path).exists():
        print(f"\nload regressors: {args.regressors_path}")
        with open(args.regressors_path, "rb") as f:
            regressors = pickle.load(f)
        print(f"  loaded {sum(r is not None for r in regressors)}/{N_JOINTS} regressors")
    else:
        print("\nscan train data (000281–000311)...")
        train_pairs = []
        for i in range(281, 312):
            folder = f"{args.root}/{i:06d}"
            if not Path(folder).exists():
                continue
            pairs = find_data_pairs(folder, max_samples=100) #args.max_train - len(train_pairs))
            train_pairs.extend(pairs)
            print(f"  {i:06d}: {len(pairs)} pairs  total {len(train_pairs)}")
            if len(train_pairs) >= args.max_train:
                break
        print(f"  train set: {len(train_pairs)} pairs")

        print(f"\nextract + sample  "
              f"(sigma={args.sigma}, pos_thr={args.pos_thr}, neg_ratio={args.neg_ratio})...")
        per_X, per_y = extract_and_sample(
            model, train_pairs, args.model,
            sigma=args.sigma, pos_thr=args.pos_thr, neg_ratio=args.neg_ratio,
        )

        print(f"\ntrain regressors  (alpha={args.alpha}, solver=lsqr)...")
        regressors = train_regressors(per_X, per_y, args.alpha)

        with open(args.regressors_path, "wb") as f:
            pickle.dump(regressors, f)
        print(f"regressors saved: {args.regressors_path}")

    # 评估
    print()
    evaluate(model, regressors, test_pairs[:args.max_eval], args.model)

    # 可视化
    if args.save_vis > 0:
        print(f"\ngenerate visualization ({args.save_vis} images)...")
        for idx, (jpg, pyd) in enumerate(test_pairs[:args.save_vis]):
            img                              = Image.open(jpg).convert("RGB")
            feats, h_p, w_p, orig_w, orig_h = process_image(model, img, args.model)
            kpts17                           = load_kpts17(pyd)
            kpts_pred, valid                 = predict_kpts(regressors, feats, h_p, w_p, orig_w, orig_h)
            scores                           = compute_pck(kpts_pred, kpts17)
            vis_vals                         = [v for v in scores.values() if v is not None]
            pck_score                        = np.mean(vis_vals) * 100 if vis_vals else 0.0
            draw_comparison(img, kpts_pred, kpts17, valid,
                            f"output/vis_{idx:03d}.jpg", pck_score)


if __name__ == "__main__":
    main()
