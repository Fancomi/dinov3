# source /home/baidu/envs/dino/bin/activate

# python3 scripts/pose_check.py

# 冻结 DINO，热图模式（推荐起点）
# python scripts/pose_regressors.py --epochs 50 --batch-size 32 --amp
# VITS: ep  5  loss=0.000070  PCK@0.2=72.1% ★ 
# VITS: ep 50  loss=0.000011  PCK@0.2=78.6%
# VITL: ep  5  loss=0.000043  PCK@0.2=80.9% ★  
# VITL: ep 50  loss=0.000004  PCK@0.2=87.1%



# # 直接回归坐标
# python scripts/pose_regressors.py --mode regression --lr 5e-4 --amp --epochs 50
# VITS: ep  5  loss=0.000075  PCK@0.2=72.2% ★  
# VITS: ep 50  loss=0.000011  PCK@0.2=77.9%
# VITL: ep  5  loss=0.000049  PCK@0.2=79.2% ★
# VITL: ep 50  loss=0.000004  PCK@0.2=86.5%


# # 解冻 DINO 微调（显存14.2G）
# python scripts/pose_regressors.py --unfreeze-dino --batch-size 8 --lr 5e-4 --amp --epochs 5
# VITS: ep  5  loss=0.000025  PCK@0.2=77.9% ★
# VITL: ep  5  loss=0.000008  PCK@0.2=91.3% ★


# python scripts/pose_regressors.py --unfreeze-dino --batch-size 8 --lr 5e-4 --amp --mode regression --epochs 5
# VITS: ep  5  loss=0.000026  PCK@0.2=78.7% ★
# VITL: ep  5  loss=0.000010  PCK@0.2=90.4% ★


# # 加载已有模型直接评估 + 可视化
# python scripts/pose_regressors.py --ckpt output/pose_best.pth