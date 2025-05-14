import os
import shutil
import random       # 用于打乱图像顺序（实现随机划分）
from tqdm import tqdm
source_dir = 'flower_dataset_raw'
target_dir = 'flower_dataset'
train_ratio = 0.8           #题目要求比例
categories = os.listdir(source_dir)
# 遍历每一个类别进行处理
for category in categories:
    cat_path = os.path.join(source_dir, category)
    images = os.listdir(cat_path)
    # 随机打乱图像顺序（确保每次划分都不一样）
    random.shuffle(images)
    # 按比例计算划分索引
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]  # 前80%作为训练集
    val_imgs = images[split_idx:]    # 后20%作为验证集
    # 创建对应的目标文件夹（按类别分别建立）
    train_cat_dir = os.path.join(target_dir, 'train', category)
    val_cat_dir = os.path.join(target_dir, 'val', category)
    os.makedirs(train_cat_dir, exist_ok=True)
    os.makedirs(val_cat_dir, exist_ok=True)
    # 将训练图像复制到 train/ 类别目录
    for img in tqdm(train_imgs, desc=f'Train: {category}'):
        src = os.path.join(cat_path, img)              # 源文件路径
        dst = os.path.join(train_cat_dir, img)         # 目标文件路径
        shutil.copyfile(src, dst)                      # 拷贝文件
    # 将验证图像复制到 val/ 类别目录
    for img in tqdm(val_imgs, desc=f'Val: {category}'):
        src = os.path.join(cat_path, img)
        dst = os.path.join(val_cat_dir, img)
        shutil.copyfile(src, dst)
# 所有类别完成后输出提示
print("数据集划分完成！")
