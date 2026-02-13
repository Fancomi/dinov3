#!/usr/bin/env python3
"""DINOv3工具函数模块 - 提供模型加载、特征提取等公共功能"""

import torch
import torchvision.transforms.functional as TF


# 常量定义
PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}


def load_model(model_name="dinov3_vitl16", local_path="/home/baidu/Documents/workspace/dinov3"):
    """从本地路径加载DINOv3模型
    
    Args:
        model_name: 模型名称
        local_path: 本地仓库路径
        
    Returns:
        加载的模型(已移至CUDA并设为eval模式)
    """
    model = torch.hub.load(repo_or_dir=local_path, model=model_name, source="local")
    model.cuda()
    model.eval()
    return model


def resize_transform(img, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """调整图像尺寸为patch_size的整数倍
    
    Args:
        img: PIL图像或张量
        image_size: 目标高度
        patch_size: patch大小
        
    Returns:
        调整后的张量
    """
    w, h = img.size if hasattr(img, 'size') else (img.shape[2], img.shape[1])
    h_patches = image_size // patch_size
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(img, (h_patches * patch_size, w_patches * patch_size)))


def extract_features(model, image_tensor, model_name):
    """提取DINOv3特征
    
    Args:
        model: DINOv3模型
        image_tensor: 图像张量 [C, H, W]
        model_name: 模型名称
        
    Returns:
        特征张量 [N_patches, D]
    """
    n_layers = MODEL_LAYERS[model_name]
    image_norm = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float32):
        feats = model.get_intermediate_layers(
            image_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
    
    return x.view(x.shape[0], -1).permute(1, 0)


def get_patch_grid_size(image_tensor, patch_size=PATCH_SIZE):
    """获取patch网格尺寸
    
    Args:
        image_tensor: 图像张量 [C, H, W]
        patch_size: patch大小
        
    Returns:
        (h_patches, w_patches)
    """
    return tuple(d // patch_size for d in image_tensor.shape[1:])
