#!/usr/bin/env python3
"""前景分割训练脚本 - 使用DINOv3特征训练线性前景分割模型"""

import io
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from dinov3_utils import load_model, resize_transform, extract_features, get_patch_grid_size, PATCH_SIZE


def load_images_from_dir(dir_path):
    """从本地目录加载图像"""
    images = []
    path = Path(dir_path)
    # 按文件名排序以确保图像和标签对应
    for img_path in sorted(path.glob("*")):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            images.append(Image.open(img_path).convert('RGB'))
    return images


def load_labels_from_dir(dir_path):
    """从本地目录加载标签"""
    labels = []
    path = Path(dir_path)
    for label_path in sorted(path.glob("*")):
        if label_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            labels.append(Image.open(label_path))
    return labels


def prepare_training_data(model, images, labels, model_name):
    """准备训练数据和标签 - 与原始notebook完全对齐"""
    import torch
    
    xs, ys, image_index = [], [], []
    
    # 量化滤波器(与原始notebook相同)
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
    
    for i, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc="提取特征")):
        # 提取特征
        img_tensor = resize_transform(img.convert('RGB'))
        feats = extract_features(model, img_tensor, model_name)
        xs.append(feats)
        
        # 处理标签: 使用Conv2d量化(与原始notebook相同)
        mask_i = label.split()[-1]  # alpha通道
        mask_i_resized = resize_transform(mask_i)
        with torch.no_grad():
            mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
        ys.append(mask_i_quantized)
        
        # 记录图像索引
        image_index.append(i * torch.ones(ys[-1].shape))
    
    # 拼接所有数据
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)
    
    # 关键步骤: 过滤中间值(与原始notebook相同)
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]
    
    print(f"过滤前样本数: {len(ys) + (~idx).sum()}")
    print(f"过滤后样本数: {len(ys)}")
    print(f"过滤掉样本数: {(~idx).sum()}")
    
    # 二值化标签: >0(与原始notebook相同)
    y_binary = (ys > 0).long().numpy()
    
    return xs.numpy(), y_binary


def train_classifier(X, y):
    """训练逻辑回归分类器 - 与原始notebook参数对齐"""
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    print(f"正样本比例: {y.mean():.2%}")
    
    # 使用与原始notebook相同的参数: C=0.1, max_iter=100000
    clf = LogisticRegression(random_state=0, C=0.1, max_iter=100000, verbose=1)
    clf.fit(X, y)
    
    train_score = clf.score(X, y)
    print(f"训练准确率: {train_score:.2%}")
    
    return clf


def evaluate_classifier(clf, X, y):
    """评估分类器性能"""
    y_pred_proba = clf.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    ap_score = average_precision_score(y, y_pred_proba)
    
    print(f"平均精度(AP): {ap_score:.4f}")
    return precision, recall, ap_score


def visualize_result(image, fg_score, save_path="output/fg_seg_result.jpg"):
    """使用OpenCV可视化前景分割结果"""
    # 转换图像
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 调整热图尺寸并归一化
    heatmap = cv2.resize(fg_score.astype(np.float32), (img_np.shape[1], img_np.shape[0]))
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 叠加显示
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # 拼接原图和结果
    combined = np.hstack([img_bgr, overlay])
    
    # 添加文字
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Foreground Score", (img_bgr.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, combined)
    print(f"结果已保存至: {save_path}")


def main():
    """主函数"""
    model_name = "dinov3_vits16"
    
    # 加载模型
    model = load_model(model_name)
    
    # 加载训练数据
    print("加载训练数据...")
    images = load_images_from_dir("datas/foreground_segmentation_images")
    labels = load_labels_from_dir("datas/foreground_segmentation_labels")
    print(f"加载了 {len(images)} 张图像")
    
    # 准备训练数据
    X, y = prepare_training_data(model, images, labels, model_name)
    
    # 训练分类器
    print("\n训练前景分类器...")
    clf = train_classifier(X, y)
    
    # 评估
    print("\n评估分类器...")
    evaluate_classifier(clf, X, y)
    
    # 保存模型
    save_path = "output/fg_classifier.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\n分类器已保存至: {save_path}")
    
    # 可视化示例
    print("\n生成可视化示例...")
    # 使用另一处本地读取的测试图
    test_img_path = "datas/local.png"
    test_img = Image.open(test_img_path).convert('RGB')
    test_tensor = resize_transform(test_img)
    h_patches, w_patches = get_patch_grid_size(test_tensor)
    
    print("test_img", np.asarray(test_img).shape)
    print("test_tensor", np.asarray(test_tensor).shape)
    print(f"h_patches, w_patches: {h_patches} {w_patches}")
    
    feats = extract_features(model, test_tensor, model_name)
    fg_score = clf.predict_proba(np.asarray(feats))[:, 1].reshape(h_patches, w_patches)
    fg_score_filtered = signal.medfilt2d(fg_score, kernel_size=3)
    
    print("feats", np.asarray(feats).shape)
    print("fg_score", fg_score.shape)
    print(f"fg_score_filtered: {fg_score_filtered.shape}")
    
    visualize_result(test_img, fg_score_filtered)


if __name__ == "__main__":
    main()
