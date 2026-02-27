#!/usr/bin/env python3
"""H36M数据集姿态可视化脚本 - 自定义17点骨架（避开空数据关键点）"""
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 44点 -> 自定义17点 索引映射（全部使用有效非空关键点）
# 目标: head, neck, thorax, spine, pelvis,
#       lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist,
#       lhip, rhip, lknee, rknee, lankle, rankle
#  0:head        <- 43: Head(H36M)
#  1:neck        <- 37: neck
#  2:thorax      <- 1:  OP Neck (thorax#40为空，用OP Neck近似)
#  3:spine       <- 41: Spine(H36M)
#  4:pelvis      <- 39: hip(MPII)
#  5:lshoulder   <- 34: lshoulder
#  6:rshoulder   <- 33: rshoulder
#  7:lelbow      <- 35: lelbow
#  8:relbow      <- 32: relbow
#  9:lwrist      <- 36: lwrist
# 10:rwrist      <- 31: rwrist
# 11:lhip        <- 28: lhip
# 12:rhip        <- 27: rhip
# 13:lknee       <- 29: lknee
# 14:rknee       <- 26: rknee
# 15:lankle      <- 30: lankle
# 16:rankle      <- 25: rankle
KPT_MAP = [43, 37, 1, 41, 39, 34, 33, 35, 32, 36, 31, 28, 27, 29, 26, 30, 25]

JOINT_NAMES = [
    'head', 'neck', 'thorax', 'spine', 'pelvis',
    'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'lhip', 'rhip',
    'lknee', 'rknee', 'lankle', 'rankle',
]

KINTREE = [
    (0, 1), (1, 2), (2, 3), (3, 4),              # head -> pelvis
    (2, 5), (5, 7), (7, 9),                        # L arm
    (2, 6), (6, 8), (8, 10),                       # R arm
    (4, 11), (11, 13), (13, 15),                   # L leg
    (4, 12), (12, 14), (14, 16),                   # R leg
]

_C = (200, 200, 200)
_L = (0, 200, 0)
_R = (0, 200, 200)
BONE_COLORS = [_C, _C, _C, _C, _L, _L, _L, _R, _R, _R, _L, _L, _L, _R, _R, _R]


def load_pyd(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def map_keypoints(kpts44):
    """将44点关键点映射为自定义17点，返回 (17, 3)"""
    return kpts44[KPT_MAP]  # (17, 3): x, y, visibility


def draw_pose(img, kpts44, radius=4, thickness=2, show_name=False):
    """在图像上绘制自定义17点骨架"""
    kpts = map_keypoints(kpts44)  # (17, 3)
    for (i, j), color in zip(KINTREE, BONE_COLORS):
        if kpts[i, 2] > 0 and kpts[j, 2] > 0:
            cv2.line(img, tuple(kpts[i, :2].astype(int)), tuple(kpts[j, :2].astype(int)), color, thickness)
    for idx, pt in enumerate(kpts):
        if pt[2] > 0:
            p = tuple(pt[:2].astype(int))
            cv2.circle(img, p, radius, (255, 255, 255), -1)
            if show_name:
                cv2.putText(img, JOINT_NAMES[idx], p, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 255), 1)
    return img


def find_data_pairs(root_dir, max_samples=None):
    """查找(jpg, pyd)数据对"""
    pairs = []
    for train_dir in sorted(Path(root_dir).glob("*/h36m-train")):
        for pyd in tqdm(sorted(train_dir.glob("*.data.pyd")), desc=train_dir.parent.name):
            jpg = pyd.with_suffix('').with_suffix('.jpg')
            if jpg.exists():
                pairs.append((jpg, pyd))
                if max_samples and len(pairs) >= max_samples:
                    return pairs
    return pairs


def visualize_samples(pairs, num_samples=5, show_name=False):
    for jpg, pyd in pairs[:num_samples]:
        img = cv2.cvtColor(np.array(Image.open(jpg).convert('RGB')), cv2.COLOR_RGB2BGR)
        kpts = load_pyd(pyd)[0]['keypoints_2d']  # (44, 3)
        vis = draw_pose(img.copy(), kpts, show_name=show_name)
        cv2.imshow("Pose-17 Skeleton", np.hstack([img, vis]))
        if cv2.waitKey(0) == 27:  # ESC退出
            break
    cv2.destroyAllWindows()


def main():
    root_dir = "/media/baidu/8C1A3A981A3A7F70/DATAS/h36m/h36m-tars/"
    print("扫描数据...")
    pairs = find_data_pairs(root_dir, max_samples=100)
    print(f"找到 {len(pairs)} 对数据")
    if pairs:
        visualize_samples(pairs, num_samples=5, show_name=True)


if __name__ == "__main__":
    main()
