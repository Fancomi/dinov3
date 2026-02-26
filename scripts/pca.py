#!/usr/bin/env python3
"""DINOv3 PCA 可视化脚本 - 直接基于模型输出计算全图 PCA，支持批量处理"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from dinov3_utils import load_model, resize_transform, extract_features, get_patch_grid_size

def compute_pca(features, n_components=3):
    """直接在所有特征上计算 PCA 投影并归一化"""
    pca = PCA(n_components=n_components, whiten=True)
    projected = pca.fit_transform(features.numpy())
    # 使用 sigmoid 映射到 [0, 1] 以获得更好的色彩表现
    return torch.nn.functional.sigmoid(torch.from_numpy(projected) * 2.0)

def visualize_and_save(image, projected_img, save_path):
    """生成并保存原图与 PCA 彩虹图的对比"""
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    # PCA 结果转为 BGR 图像
    pca_np = (projected_img.numpy() * 255).astype(np.uint8)
    pca_res = cv2.resize(cv2.cvtColor(pca_np, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 水平拼接并添加标注
    combined = np.hstack([cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), pca_res])
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, "PCA", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)

def main(model_name="dinov3_vits16", input_dir="datas", output_dir="output"):
    """主循环：加载模型并处理指定目录下的所有图像"""
    model = load_model(model_name)
    
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found.")
        return

    img_exts = ('.jpg', '.jpeg', '.png', '.webp')
    img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
    
    if not img_paths:
        print(f"No valid images found in '{input_dir}'.")
        return

    print(f"Processing {len(img_paths)} images...")
    for path in img_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = resize_transform(img)
            h_p, w_p = get_patch_grid_size(img_tensor)
            
            # 提取特征并计算 PCA
            feats = extract_features(model, img_tensor, model_name)
            projected = compute_pca(feats).view(h_p, w_p, 3)
            
            # 保存结果
            save_path = os.path.join(output_dir, f"pca_{os.path.basename(path)}")
            visualize_and_save(img, projected, save_path)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

if __name__ == "__main__":
    main()
