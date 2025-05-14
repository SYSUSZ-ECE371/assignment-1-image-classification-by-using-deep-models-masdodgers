import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# 选择训练设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
data_dir = r'D:\pcdesktop\flower_dataset\EX2\flower_dataset'

# 图像增强与预处理（用于训练与验证）
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),               # 随机裁剪为224x224
    transforms.RandomHorizontalFlip(),               # 随机水平翻转
    transforms.RandomRotation(15),                   # 随机旋转
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),       # 色彩扰动
    transforms.RandomVerticalFlip(),                 # 随机垂直翻转
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],       # 标准化（ImageNet mean/std）
                         [0.229, 0.224, 0.225])
])

# 加载数据集
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
train_size = int(0.8 * len(full_dataset))  # 训练集 80%
val_size = len(full_dataset) - train_size  # 验证集 20%
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# DataLoader 设置
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

# 加载预训练的 ResNet18 并替换最后一层
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 替换分类头
model = model.to(device)

# 设置损失函数、优化器和学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练函数（带训练曲线可视化）
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 用于绘图
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n{"-" * 10}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # 保存最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_dir = 'EX2/work_dir'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
                print("Best model saved.\n")

    time_elapsed = time.time() - since
    print(f"Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('EX2/work_dir/training_curves.png')
    plt.show()

    return model

# 主程序入口（适配 Windows）
if __name__ == "__main__":
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
