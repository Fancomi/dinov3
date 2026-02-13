#!/usr/bin/env python3
"""前景分割训练脚本 - 使用DINOv3特征训练线性前景分割模型"""

import io
import pickle
import tarfile
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from dinov3_utils import load_model, resize_transform, extract_features, get_patch_grid_size, PATCH_SIZE


def load_images_from_tar(tar_uri):
    """从远程tar包加载图像"""
    images = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in sorted(tar.getmembers(), key=lambda x: x.name):
            if member.isfile():
                images.append(Image.open(tar.extractfile(member)))
    return images


def prepare_training_data(model, images, labels, model_name):
    """准备训练数据和标签"""
    X_all, y_all = [], []
    
    for img, label in tqdm(zip(images, labels), total=len(images), desc="提取特征"):
        img_tensor = resize_transform(img)
        h_patches, w_patches = get_patch_grid_size(img_tensor)
        
        # 提取特征
        feats = extract_features(model, img_tensor, model_name)
        
        # 处理标签: 提取alpha通道并量化到patch级别
        mask = np.array(label.split()[-1])  # alpha通道
        mask_tensor = resize_transform(Image.fromarray(mask))
        mask_quantized = (mask_tensor.view(h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
                         .mean(dim=(1, 3)).squeeze() > 0.5).float()
        
        X_all.append(feats.numpy())
        y_all.append(mask_quantized.view(-1).numpy())
    
    return np.vstack(X_all), np.concatenate(y_all)


def train_classifier(X, y):
    """训练逻辑回归分类器"""
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    print(f"正样本比例: {y.mean():.2%}")
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
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


def visualize_result(image, fg_score, save_path="fg_seg_result.jpg"):
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
    model_name = "dinov3_vitl16"
    
    # 加载模型
    print(f"加载模型: {model_name}")
    model = load_model(model_name)
    
    # 加载训练数据
    print("加载训练数据...")
    images = load_images_from_tar(
        "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_images.tar.gz"
    )
    labels = load_images_from_tar(
        "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_labels.tar.gz"
    )
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
    save_path = "fg_classifier.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\n分类器已保存至: {save_path}")
    
    # 可视化示例
    print("\n生成可视化示例...")
    test_img = images[0]
    test_tensor = resize_transform(test_img)
    h_patches, w_patches = get_patch_grid_size(test_tensor)
    
    feats = extract_features(model, test_tensor, model_name)
    fg_score = clf.predict_proba(feats.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score_filtered = signal.medfilt2d(fg_score, kernel_size=3)
    
    visualize_result(test_img, fg_score_filtered)


if __name__ == "__main__":
    main()
