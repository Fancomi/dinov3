# Pose Estimation：热图模式 vs 回归模式 分析与修复

> 对应脚本：`scripts/pose_regressors.py`

---

## 问题现象

| 模式 | 配置 | loss | PCK@0.2 |
|---|---|---|---|
| 热图（冻结 DINO） | `--max-train 3000 --epochs 5 --batch-size 32 --amp` | 0.000655 | **1.0%** ❌ |
| 回归（冻结 DINO） | `--mode regression --lr 5e-4 --amp` | 0.0005 | **17.3%** |
| 热图（解冻 DINO） | `--unfreeze-dino --batch-size 8 --lr 5e-4 --amp` | 0.0001 | **1.0%** ❌ |
| 回归（解冻 DINO） | `--unfreeze-dino --batch-size 8 --lr 5e-4 --amp --mode regression` | 0.0001 | **54.3%** |

**症状**：热图模式 loss 极低但 PCK 毫无好转，可视化显示所有关键点聚集在图像中央。

---

## Bug 1：热图模式全零坍塌（Zero-Collapse）

### 根因

旧 `compute_loss` 对热图像素做 MSE：

```python
# ❌ 旧代码
tgt   = make_target_heatmaps(kpts, owh, h_p, w_p, sigma)
per_j = F.mse_loss(pred, tgt, reduction="none").mean([-2,-1])
```

问题链条：

```
高斯目标热图 ≈ 98% 格子值 ≈ 0
        ↓
MSE 最优解 = 全零预测（loss → 0，模型"骗过"了损失函数）
        ↓
全零热图 → F.softmax → 均匀分布
        ↓
soft-argmax 期望坐标 = 网格中心 = 图像中心
        ↓
所有关键点堆在中央，PCK ≈ 1%
```

这也解释了为什么解冻 DINO 后 loss 极低（≈0.0001）但 PCK 依然是 1%——网络只是更高效地把全图预测为零。

### 修复：积分回归损失（Integral Regression Loss）

**原则：不能对热图本身做 MSE，必须对解码后的坐标做损失。**

```python
# ✅ 新代码
if mode == "heatmap":
    px_pred = soft_argmax(pred, h_p, w_p, owh)   # 热图 → 坐标
    gt_px   = kpts[..., :2]
    norm    = owh[:, None, :]
    per_j   = F.smooth_l1_loss(px_pred / norm,
                               gt_px   / norm,
                               reduction="none").mean(-1)
```

梯度穿过 `soft_argmax` 反向传播，直接推动热图质心移向正确位置，从根本上消除全零坍塌。

**修复效果**：PCK@0.2 从 **1%** 跳到 **71.9%**（冻结 DINO，ViT-S）。

---

## Bug 2：回归模式 mean-pool 丢失空间信息

### 根因

旧 `RegressionHead` 对所有 patch token 做均值池化：

```python
# ❌ 旧代码
def forward(self, x):
    return self.net(x.mean(1)).reshape(-1, N_JOINTS, 2)
```

DINO 的 patch token 是**空间局部描述子**：每个 token 主要编码对应图像块的局部特征。均值池化后：

- 784 个空间位置被**彻底压缩**为一个向量
- 位置信息完全消失——模型不再知道"左肩特征"在图像哪个位置
- 在冻结 backbone 下，从均值向量中**凭空恢复坐标**几乎不可能泛化

此外旧代码末尾用 `nn.Sigmoid()`，在接近 0/1（图像边缘关键点）时梯度趋近于零，导致脚踝、手腕等靠边关节严重欠拟合。

### 修复：空间注意力加权坐标期望

```python
# ✅ 新代码
class RegressionHead(nn.Module):
    def __init__(self, D, h_p, w_p):
        super().__init__()
        self.hw = (h_p, w_p)
        self.key_proj = nn.Sequential(
            nn.Conv2d(D,   256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, N_JOINTS, 1))   # (B,J,h_p,w_p) 注意力 logits
        # 固定坐标网格 ∈ [0,1]
        gy = torch.arange(h_p).float().view(1, 1, -1, 1) / h_p
        gx = torch.arange(w_p).float().view(1, 1, 1, -1) / w_p
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

    def forward(self, x):
        B, N, D = x.shape
        fm  = x.permute(0, 2, 1).reshape(B, D, *self.hw)
        att = F.softmax(
                self.key_proj(fm).reshape(B, N_JOINTS, -1), dim=-1
              ).reshape(B, N_JOINTS, *self.hw)
        px  = (att * self.grid_x).sum([-2, -1])
        py  = (att * self.grid_y).sum([-2, -1])
        return torch.stack([px, py], dim=-1)   # (B,J,2) ∈ [0,1]
```

本质是"无高斯目标的软热图回归"：保留完整 28×28 空间网格，Conv2d 为每个关节学习空间检测权重，softmax 归一化后做坐标加权期望。

---

## 两种模式的完整对比

### 网络结构

```
DINO patch tokens (B, 784, D)
        │
   ┌────┴────┐
   ▼         ▼
HeatmapHead  RegressionHead（新）
reshape →    reshape →
(B,D,28,28)  (B,D,28,28)
Conv2d       Conv2d
(B,J,28,28)  softmax + 加权期望
   │                  │
   ▼                  ▼
热图（可视化）   归一化坐标 (B,J,2)
   │
soft_argmax
   │
像素坐标
```

### Loss 机制

| | 热图模式 | 回归模式 |
|---|---|---|
| 网络输出 | 热图 `(B,J,28,28)` | 归一化坐标 `(B,J,2) ∈ [0,1]` |
| Loss | smooth-L1 on **解码后坐标** | smooth-L1 on **直接输出坐标** |
| 解码 | `soft_argmax(热图)` → 像素坐标 | `pred × owh` → 像素坐标 |
| 梯度路径 | 穿过 soft_argmax 流回热图 | 直接到 Conv2d 权重 |
| 热图用途 | 中间载体（可提取可视化） | 内部隐式（不显式输出） |

**两种模式的 loss 信号语义完全相同**（都是坐标 smooth-L1），区别只在于网络是否有显式的热图中间表示。

### 热图模式为何优于旧回归模式

修复后热图模式（71.9% PCK）远优于旧回归模式（17.3%）的原因不是"热图本身神奇"，而是：

1. **热图头保留了空间结构**，旧回归头用 mean-pool 丢弃了
2. **Sigmoid 梯度消失**导致边缘关节欠拟合

新回归头（空间注意力版）与热图头结构几乎相同，预期 PCK 可与热图模式相当。

---

## 修改摘要

| 位置 | 改动 |
|---|---|
| `compute_loss`，heatmap 分支 | `MSE(pred_hm, gaussian_tgt)` → `smooth_L1(soft_argmax(pred)/owh, gt/owh)` |
| `RegressionHead.__init__` | 新增 `h_p, w_p` 参数；删除 MLP+Sigmoid；改为 Conv2d + 注意力加权坐标 |
| `PoseModel.__init__` | `RegressionHead(D)` → `RegressionHead(D, h_p, w_p)` |
