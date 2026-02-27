#!/usr/bin/env python3
"""
DINOv3 Pose Estimation — PyTorch
  --mode heatmap     热图 + soft-argmax（默认）
  --mode regression  直接归一化坐标回归
  --unfreeze-dino    微调 DINO 主干（默认冻结）
  --amp              自动混合精度
"""

import os, pickle, argparse
from pathlib import Path

import cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as TF
from torch.amp import autocast, GradScaler          # ← 新 API，无 deprecation warning
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from dinov3_utils import load_model, PATCH_SIZE      # 只需这两个

# ══════ 常量 ══════════════════════════════════════════════════
KPT_MAP = [43,37,1,41,39,34,33,35,32,36,31,28,27,29,26,30,25]
JOINT_NAMES = ["head","neck","thorax","spine","pelvis",
               "lshoulder","rshoulder","lelbow","relbow",
               "lwrist","rwrist","lhip","rhip","lknee","rknee","lankle","rankle"]
KINTREE = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,7),(7,9),
           (2,6),(6,8),(8,10),(4,11),(11,13),(13,15),(4,12),(12,14),(14,16)]
_C,_L,_R    = (200,200,200),(0,200,0),(0,200,255)
BONE_COLORS = [_C]*4+[_L]*3+[_R]*3+[_L]*3+[_R]*3
N_JOINTS    = 17
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════ Transform（固定尺寸，解决 stack 报错）══════════════════

def build_transform(img_size: int):
    """所有图片统一缩放到 img_size×img_size，保证批内形状一致"""
    return TF.Compose([
        TF.Resize((img_size, img_size)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


# ══════ 数据 ══════════════════════════════════════════════════

def find_data_pairs(folder, max_samples=None):
    d = Path(folder) / "h36m-train"
    if not d.exists(): return []
    pairs = []
    for pyd in sorted(d.glob("*.data.pyd")):
        jpg = pyd.with_suffix("").with_suffix(".jpg")
        if jpg.exists():
            pairs.append((jpg, pyd))
            if max_samples and len(pairs) >= max_samples: break
    return pairs

def load_kpts17(pyd_path):
    with open(pyd_path, "rb") as f:
        data = pickle.load(f)
    return data[0]["keypoints_2d"][KPT_MAP].astype(np.float32)  # (17,3)

class PoseDataset(Dataset):
    def __init__(self, pairs, transform, img_size):
        self.pairs = pairs; self.transform = transform; self.img_size = img_size

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        jpg, pyd  = self.pairs[idx]
        img       = Image.open(jpg).convert("RGB")
        orig_w, orig_h = img.width, img.height
        k17       = load_kpts17(pyd).copy()
        # 关键点坐标同步缩放到 img_size 空间
        k17[:, 0] *= self.img_size / orig_w
        k17[:, 1] *= self.img_size / orig_h
        owh       = torch.tensor([self.img_size, self.img_size], dtype=torch.float32)
        return self.transform(img), torch.from_numpy(k17), owh


# ══════ DINO 特征提取 ══════════════════════════════════════════

def get_patch_tokens(dino, x):
    """→ (B, N, D)，兼容 DINOv2 dict / timm CLS+patch 两种接口"""
    if hasattr(dino, "forward_features"):
        out = dino.forward_features(x)
        if isinstance(out, dict):
            for k in ("x_norm_patchtokens", "patch_tokens"):
                if k in out: return out[k]
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            return out[:, 1:]   # 去掉 CLS token
    return dino(x)


# ══════ 模型头 ════════════════════════════════════════════════

class HeatmapHead(nn.Module):
    """patch tokens (B,N,D) → 热图 (B,J,h_p,w_p)"""
    def __init__(self, D, h_p, w_p):
        super().__init__()
        self.hw = (h_p, w_p)
        self.net = nn.Sequential(
            nn.Conv2d(D,   256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, N_JOINTS, 1))
    def forward(self, x):
        B, _, D = x.shape
        return self.net(x.permute(0,2,1).reshape(B, D, *self.hw))

class RegressionHead(nn.Module):
    """
    patch tokens (B,N,D) → 归一化坐标 (B,J,2) ∈ [0,1]

    ⚠ 旧实现 x.mean(1) 把 784 个空间 token 压成 1 个均值向量，
      彻底丢弃了空间结构，导致模型无法从冻结的 DINO 特征中恢复位置，
      PCK 只有 ~16%。

    新实现：保留 28×28 空间网格，用 Conv2d 为每个关节学习注意力权重，
    再做加权坐标期望（本质 = "无高斯目标的软热图回归"）。
    """
    def __init__(self, D, h_p, w_p):
        super().__init__()
        self.hw = (h_p, w_p)
        self.key_proj = nn.Sequential(
            nn.Conv2d(D,   256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, N_JOINTS, 1))   # (B,J,h_p,w_p) 注意力 logits
        # 固定坐标网格（∈[0,1]），不参与梯度
        gy = torch.arange(h_p).float().view(1, 1, -1, 1) / h_p   # 行方向 (y)
        gx = torch.arange(w_p).float().view(1, 1, 1, -1) / w_p   # 列方向 (x)
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

    def forward(self, x):
        B, N, D = x.shape
        fm  = x.permute(0, 2, 1).reshape(B, D, *self.hw)           # (B,D,h_p,w_p)
        att = F.softmax(
                self.key_proj(fm).reshape(B, N_JOINTS, -1), dim=-1
              ).reshape(B, N_JOINTS, *self.hw)                      # (B,J,h_p,w_p)
        px  = (att * self.grid_x).sum([-2, -1])                     # (B,J)
        py  = (att * self.grid_y).sum([-2, -1])
        return torch.stack([px, py], dim=-1)                        # (B,J,2) ∈ [0,1]

class PoseModel(nn.Module):
    def __init__(self, dino, D, h_p, w_p, mode="heatmap", freeze_dino=True):
        super().__init__()
        self.dino = dino; self.mode = mode; self.freeze_dino = freeze_dino
        if freeze_dino:
            for p in dino.parameters(): p.requires_grad_(False)
        self.head = HeatmapHead(D, h_p, w_p) if mode == "heatmap" else RegressionHead(D, h_p, w_p)

    def train(self, mode=True):
        """冻结模式下 DINO 永远保持 eval（禁用 Dropout/BN 扰动）"""
        super().train(mode)
        if self.freeze_dino: self.dino.eval()
        return self

    def forward(self, x):
        if self.freeze_dino:
            with torch.no_grad():
                feats = get_patch_tokens(self.dino, x).detach()
        else:
            feats = get_patch_tokens(self.dino, x)
        return self.head(feats.float())


# ══════ 热图工具 ══════════════════════════════════════════════

def make_target_heatmaps(kpts, owh, h_p, w_p, sigma):
    """(B,J,3), (B,2) → (B,J,h_p,w_p) 高斯目标热图"""
    B, J, _ = kpts.shape; dev = kpts.device
    r  = torch.arange(h_p, device=dev).float() + 0.5
    c  = torch.arange(w_p, device=dev).float() + 0.5
    gr, gc = torch.meshgrid(r, c, indexing="ij")
    sx = (w_p * PATCH_SIZE) / owh[:, 0]    # (B,)
    sy = (h_p * PATCH_SIZE) / owh[:, 1]
    pc = kpts[..., 0] * sx[:, None] / PATCH_SIZE   # (B,J) 列 patch 坐标
    pr = kpts[..., 1] * sy[:, None] / PATCH_SIZE   # (B,J) 行 patch 坐标
    dr = gr[None, None] - pr[:, :, None, None]      # (B,J,h_p,w_p)
    dc = gc[None, None] - pc[:, :, None, None]
    return torch.exp(-(dr**2 + dc**2) / (2 * sigma**2))

def soft_argmax(hm, h_p, w_p, owh):
    """hm:(B,J,h_p,w_p), owh:(B,2) → (B,J,2) 像素坐标"""
    B, J, _, _ = hm.shape; dev = hm.device
    r  = torch.arange(h_p, device=dev).float() + 0.5
    c  = torch.arange(w_p, device=dev).float() + 0.5
    wt = F.softmax(hm.reshape(B, J, -1), dim=-1).reshape(B, J, h_p, w_p)
    pr = (wt * r.view(1,1,-1,1)).sum([-2,-1])
    pc = (wt * c.view(1,1,1,-1)).sum([-2,-1])
    sx = (w_p * PATCH_SIZE) / owh[:, 0:1]
    sy = (h_p * PATCH_SIZE) / owh[:, 1:2]
    return torch.stack([pc * PATCH_SIZE / sx,
                        pr * PATCH_SIZE / sy], dim=-1)


# ══════ 损失 & 解码 ════════════════════════════════════════════

def compute_loss(pred, kpts, owh, h_p, w_p, mode, sigma):
    """
    热图模式：积分回归损失（Integral Regression Loss）
    ─────────────────────────────────────────────────────────
    ⚠ 不能对原始热图做 MSE！
    原因：高斯目标 ≈98% 格子值接近 0，MSE 被全零预测轻易最小化。
    全零热图经 soft-argmax → 均匀分布 → 期望坐标 = 网格中心 → 所有点堆在图像中央。

    正确做法：soft-argmax 解码预测坐标 → 对坐标做 smooth-L1。
    梯度直接推动热图质心到正确位置，彻底消除全零坍塌。
    """
    vis = (kpts[..., 2] > 0).float()                       # (B,J)
    if mode == "heatmap":
        # 积分回归：解码坐标 → 归一化 → smooth-L1
        px_pred = soft_argmax(pred, h_p, w_p, owh)         # (B,J,2) 像素坐标（img_size 空间）
        gt_px   = kpts[..., :2]                             # (B,J,2)
        norm    = owh[:, None, :]                           # (B,1,2) 归一化到 [0,1]
        per_j   = F.smooth_l1_loss(px_pred / norm,
                                   gt_px   / norm,
                                   reduction="none").mean(-1)
    else:
        gt    = torch.stack([kpts[...,0] / owh[:,0:1],
                             kpts[...,1] / owh[:,1:2]], dim=-1)
        per_j = F.smooth_l1_loss(pred, gt, reduction="none").mean(-1)
    return (per_j * vis).sum() / (vis.sum() + 1e-6)

def decode_to_pixels(pred, owh, h_p, w_p, mode):
    if mode == "heatmap": return soft_argmax(pred, h_p, w_p, owh)
    return pred * owh[:, None, :]


# ══════ 评估 ══════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, h_p, w_p, mode, verbose=False):
    model.eval()
    preds, gts = [], []
    for imgs, kpts, owh in loader:
        pred = model(imgs.to(DEVICE))
        px   = decode_to_pixels(pred, owh.to(DEVICE), h_p, w_p, mode)
        preds.append(px.cpu().numpy()); gts.append(kpts.numpy())
    P  = np.concatenate(preds); K = np.concatenate(gts)
    torso = np.linalg.norm(K[:,1,:2] - K[:,4,:2], axis=-1, keepdims=True) + 1e-6
    dist  = np.linalg.norm(P - K[:,:,:2], axis=-1)
    vis   = K[:,:,2] > 0
    hit   = (dist < 0.2 * torso) & vis
    pck   = hit.sum() / (vis.sum() + 1e-6)
    if verbose:
        print(f"\n{'='*50}\n  PCK@0.2  ({len(P)} images)\n{'='*50}")
        print(f"  Overall : {pck*100:.1f}%\n")
        for k, nm in enumerate(JOINT_NAMES):
            v = vis[:,k]
            s = f"{hit[:,k][v].mean()*100:.1f}%  (n={v.sum()})" if v.any() else "N/A"
            print(f"  {nm:12s}: {s}")
        print("="*50)
    return float(pck)


# ══════ 可视化 ════════════════════════════════════════════════

def _draw_skel(canvas, kpts, mask, color):
    for (i,j), bc in zip(KINTREE, BONE_COLORS):
        if mask[i] and mask[j]:
            cv2.line(canvas, tuple(kpts[i,:2].astype(int)),
                     tuple(kpts[j,:2].astype(int)), bc, 2)
    for k in range(N_JOINTS):
        if mask[k]: cv2.circle(canvas, tuple(kpts[k,:2].astype(int)), 4, color, -1)

def draw_comparison(img, pred_px, kpts17, save_path, pck=None):
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lft, rgt = bgr.copy(), bgr.copy()
    _draw_skel(lft, kpts17,  kpts17[:,2]>0,           (255,255,255))
    _draw_skel(rgt, pred_px, np.ones(N_JOINTS, bool),  (0,80,255))
    out = np.hstack([lft,rgt]); W=bgr.shape[1]; f=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "GT",          (10,   30), f, .9, (0,255,255), 2)
    cv2.putText(out, "Prediction",  (W+10, 30), f, .9, (0,80, 255), 2)
    if pck is not None:
        cv2.putText(out, f"PCK={pck:.1f}%", (W+10, 60), f, .7, (0,255,0), 2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, out); print(f"  saved: {save_path}")


# ══════ 主程序 ════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default="/media/baidu/8C1A3A981A3A7F70/DATAS/h36m/h36m-tars")
    p.add_argument("--model",         default="dinov3_vitl16")
    p.add_argument("--mode",          default="heatmap", choices=["heatmap","regression"])
    p.add_argument("--img-size",      type=int,   default=448,
                   help="输入固定边长（须为 PATCH_SIZE 整数倍，默认 448=28×16）")
    p.add_argument("--unfreeze-dino", action="store_true")
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--max-train",     type=int,   default=3000)
    p.add_argument("--max-eval",      type=int,   default=150)
    p.add_argument("--sigma",         type=float, default=2.0)
    p.add_argument("--save-vis",      type=int,   default=10)
    p.add_argument("--eval-every",    type=int,   default=5)
    p.add_argument("--ckpt",          default=None)
    args = p.parse_args()
    os.makedirs("output", exist_ok=True)

    # img_size 须为 PATCH_SIZE 整数倍，否则 patch 网格无法整除
    assert args.img_size % PATCH_SIZE == 0, \
        f"--img-size {args.img_size} 须为 PATCH_SIZE={PATCH_SIZE} 的整数倍"
    h_p = w_p   = args.img_size // PATCH_SIZE
    transform   = build_transform(args.img_size)

    print(f"device={DEVICE}  mode={args.mode}  "
          f"dino={'unfrozen' if args.unfreeze_dino else 'frozen'}  amp={args.amp}")
    print(f"img_size={args.img_size}  grid={h_p}×{w_p}")

    # ── 数据 ──────────────────────────────────────────────────
    print("\n[data]")
    train_pairs = []
    for i in range(281, 312):
        d = f"{args.root}/{i:06d}"
        if not Path(d).exists(): continue
        train_pairs.extend(find_data_pairs(d, 100))
        if len(train_pairs) >= args.max_train: break
    test_pairs = find_data_pairs(f"{args.root}/000312",
                                 args.max_eval + args.save_vis)
    print(f"  train={len(train_pairs)}  test={len(test_pairs)}")

    train_ld = DataLoader(
        PoseDataset(train_pairs[:args.max_train], transform, args.img_size),
        args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    eval_ld = DataLoader(
        PoseDataset(test_pairs[:args.max_eval], transform, args.img_size),
        args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── 模型 ──────────────────────────────────────────────────
    print(f"\n[model] {args.model}")
    dino = load_model(args.model).to(DEVICE)
    with torch.no_grad():
        dummy    = torch.zeros(1, 3, args.img_size, args.img_size, device=DEVICE)
        feat_dim = get_patch_tokens(dino, dummy).shape[-1]
    print(f"  grid={h_p}×{w_p}  D={feat_dim}")

    model = PoseModel(dino, feat_dim, h_p, w_p,
                      mode=args.mode,
                      freeze_dino=not args.unfreeze_dino).to(DEVICE)
    print(f"  head params: {sum(p.numel() for p in model.head.parameters())/1e6:.2f}M")

    # ── 加载 or 训练 ──────────────────────────────────────────
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"  loaded: {args.ckpt}")
    else:
        opt_groups = [{"params": list(model.head.parameters()), "lr": args.lr}]
        if args.unfreeze_dino:
            opt_groups.append({"params": list(model.dino.parameters()),
                               "lr": args.lr * 0.05})
        opt    = torch.optim.AdamW(opt_groups, weight_decay=1e-4)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        scaler = GradScaler("cuda", enabled=args.amp)   # ← 新 API
        best   = 0.0

        print(f"\n[train] epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
        for ep in range(1, args.epochs + 1):
            model.train(); total = 0.0
            bar = tqdm(train_ld, desc=f"ep{ep:3d}", leave=False)
            for imgs, kpts, owh in bar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                kpts = kpts.to(DEVICE, non_blocking=True)
                owh  = owh.to(DEVICE,  non_blocking=True)
                opt.zero_grad()
                with autocast("cuda", enabled=args.amp):   # ← 新 API
                    loss = compute_loss(model(imgs), kpts, owh,
                                        h_p, w_p, args.mode, args.sigma)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update()
                total += loss.item()
                bar.set_postfix(loss=f"{loss.item():.4f}")
            sch.step(); avg = total / len(train_ld)

            if ep % args.eval_every == 0 or ep == args.epochs:
                pck = evaluate(model, eval_ld, h_p, w_p, args.mode,
                               verbose=(ep == args.epochs))
                tag = " ★" if pck > best else ""
                print(f"ep{ep:3d}  loss={avg:.6f}  PCK@0.2={pck*100:.1f}%{tag}")
                if pck > best:
                    best = pck
                    torch.save(model.state_dict(), "output/pose_best.pth")
            else:
                print(f"ep{ep:3d}  loss={avg:.6f}")
        print(f"\nbest PCK@0.2: {best*100:.2f}%")

    # ── 可视化（坐标还原到原始图像空间）────────────────────────
    if args.save_vis > 0:
        print(f"\n[vis] {args.save_vis} samples")
        model.eval()
        vis_pairs = test_pairs[args.max_eval : args.max_eval + args.save_vis]
        for idx, (jpg, pyd) in enumerate(vis_pairs):
            img            = Image.open(jpg).convert("RGB")
            orig_w, orig_h = img.width, img.height
            owh_t          = torch.tensor([[args.img_size, args.img_size]],
                                          dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                pred = model(transform(img).unsqueeze(0).to(DEVICE))
                px   = decode_to_pixels(pred, owh_t, h_p, w_p,
                                        args.mode)[0].cpu().numpy()
            # 还原到原始图像坐标系
            px[:, 0] *= orig_w / args.img_size
            px[:, 1] *= orig_h / args.img_size
            k17   = load_kpts17(pyd)                    # 原始坐标
            torso = np.linalg.norm(k17[1,:2] - k17[4,:2]) + 1e-6
            vis   = k17[:,2] > 0
            pck_s = (np.linalg.norm(px - k17[:,:2], axis=-1)[vis]
                     < 0.2 * torso).mean() * 100 if vis.any() else 0.0
            draw_comparison(img, px, k17, f"output/vis_{idx:03d}.jpg", pck_s)

if __name__ == "__main__":
    main()