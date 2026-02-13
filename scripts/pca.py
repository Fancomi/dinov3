#!/usr/bin/env python3
"""前景对象PCA可视化脚本 - 计算前景对象的PCA并生成彩虹色可视化"""

import os
import pickle
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import signal
from sklearn.decomposition import PCA

from dinov3_utils import load_model, resize_transform, extract_features, get_patch_grid_size


def load_classifier(clf_path="fg_classifier.pkl"):
    """加载前景分类器"""
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"找不到分类器文件: {clf_path}\n请先运行 foreground_segmentation.py 训练分类器")
    
    with open(clf_path, 'rb') as f:
        return pickle.load(f)


def load_image_from_url(url):
    """从URL加载图像"""
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def compute_foreground_score(clf, features, h_patches, w_patches):
    """计算前景得分并应用中值滤波"""
    fg_score = clf.predict_proba(features.numpy())[:, 1].reshape(h_patches, w_patches)
    return torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))


def compute_pca_projection(features, fg_mask, n_components=3):
    """在前景特征上拟合PCA并投影"""
    fg_features = features[fg_mask.view(-1) > 0.5]
    
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(fg_features.numpy())
    
    projected = pca.transform(features.numpy())
    return torch.from_numpy(projected)


def apply_pca_colorization(projected_features, fg_mask, h_patches, w_patches):
    """应用PCA着色并屏蔽背景"""
    # 重塑为图像尺寸
    projected_img = projected_features.view(h_patches, w_patches, 3)
    
    # 增强对比度并应用sigmoid获得鲜艳颜色
    projected_img = torch.nn.functional.sigmoid(projected_img * 2.0).permute(2, 0, 1)
    
    # 屏蔽背景
    projected_img *= (fg_mask.unsqueeze(0) > 0.5)
    
    return projected_img


def visualize_pca(image, projected_img, fg_score, save_path="pca_result.jpg"):
    """使用OpenCV可视化PCA结果"""
    # 原始图像
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # PCA彩虹图
    pca_np = (projected_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pca_bgr = cv2.cvtColor(pca_np, cv2.COLOR_RGB2BGR)
    pca_resized = cv2.resize(pca_bgr, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 前景得分热图
    fg_np = (fg_score.numpy() * 255).astype(np.uint8)
    fg_colored = cv2.applyColorMap(fg_np, cv2.COLORMAP_VIRIDIS)
    fg_resized = cv2.resize(fg_colored, (img_np.shape[1], img_np.shape[0]))
    
    # 调整所有图像为相同高度
    h = min(img_bgr.shape[0], pca_resized.shape[0], fg_resized.shape[0])
    img_bgr = cv2.resize(img_bgr, (int(img_bgr.shape[1] * h / img_bgr.shape[0]), h))
    pca_resized = cv2.resize(pca_resized, (int(pca_resized.shape[1] * h / pca_resized.shape[0]), h))
    fg_resized = cv2.resize(fg_resized, (int(fg_resized.shape[1] * h / fg_resized.shape[0]), h))
    
    # 水平拼接
    combined = np.hstack([img_bgr, fg_resized, pca_resized])
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Foreground Score", (img_bgr.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "PCA Rainbow", (img_bgr.shape[1] + fg_resized.shape[1] + 10, 30), 
                font, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, combined)
    print(f"PCA可视化已保存至: {save_path}")


def main():
    """主函数"""
    model_name = "dinov3_vitl16"
    
    # 加载模型和分类器
    print(f"加载模型: {model_name}")
    model = load_model(model_name)
    
    print("加载前景分类器...")
    clf = load_classifier()
    
    # 加载测试图像
    print("加载测试图像...")
    image_url = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"
    image = load_image_from_url(image_url)
    print(f"图像尺寸: {image.size}")
    
    # 预处理图像
    image_tensor = resize_transform(image)
    h_patches, w_patches = get_patch_grid_size(image_tensor)
    print(f"Patch网格尺寸: {h_patches}x{w_patches}")
    
    # 提取特征
    print("提取DINOv3特征...")
    features = extract_features(model, image_tensor, model_name)
    
    # 计算前景得分
    print("计算前景得分...")
    fg_score = compute_foreground_score(clf, features, h_patches, w_patches)
    print(f"前景像素比例: {(fg_score > 0.5).float().mean():.2%}")
    
    # 在前景上拟合PCA
    print("在前景特征上拟合PCA...")
    projected_features = compute_pca_projection(features, fg_score)
    
    # 应用PCA着色
    print("应用PCA着色...")
    projected_img = apply_pca_colorization(projected_features, fg_score, h_patches, w_patches)
    
    # 可视化结果
    print("生成可视化...")
    visualize_pca(image, projected_img, fg_score)


if __name__ == "__main__":
    main()
