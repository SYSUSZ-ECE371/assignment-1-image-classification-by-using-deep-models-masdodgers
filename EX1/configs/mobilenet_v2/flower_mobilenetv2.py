# 继承基础配置文件，其中包含 MobileNetV2 的主干网络结构和默认设置
_base_ = ['../mobilenet_v2/mobilenet-v2_8xb32_in1k.py']
# 模型结构配置
model = dict(
    head=dict(
        num_classes=5,  # 自定义为花卉数据集的 5 个类别
    )
)
data_root = 'data/flower_dataset'
# 训练集加载器配置
train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='ImageNet',           # 数据集采用 ImageNet 风格组织
        data_prefix=f'{data_root}/train',
        ann_file=f'{data_root}/train.txt',
        classes=f'{data_root}/classes.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224, backend='pillow'),  # 随机裁剪并缩放为 224x224
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),  # 水平随机翻转
            dict(
                type='Normalize',                 # 图像标准化
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]
    )
)
# 验证集加载器配置
val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='ImageNet',
        data_prefix=f'{data_root}/val',
        ann_file=f'{data_root}/val.txt',
        classes=f'{data_root}/classes.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),  # 先缩放图像
            dict(type='CenterCrop', crop_size=224),                 # 中心裁剪
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
)
# 验证评估器配置
val_evaluator = dict(metric='accuracy', topk=(1, ))  # 使用 Top-1 准确率作为评估标准
# 数据集引用（简写形式）
data = dict(
    train=train_dataloader['dataset'],
    val=val_dataloader['dataset']
)
# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',                # 使用随机梯度下降
        lr=0.001,                  # 初始学习率
        momentum=0.9,              # 动量参数
        weight_decay=4e-5          # 权重衰减防止过拟合
    )
)
default_hooks = dict(
    checkpoint=dict(
        save_best='auto',         # 自动保存表现最好的模型
        rule='greater'
    ),
    logger=dict(
        type='TextLoggerHook',    # 使用文本输出日志
        interval=100              # 每 100 次迭代输出一次日志
    ),
)
# 加载预训练模型路径（ImageNet 预训练权重）
load_from = r'D:\pcdesktop\flower_dataset\mmclassification\checkpoints\mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
# 设置训练设备和轮数
device = 'cuda'
runner = dict(type='EpochBasedRunner', max_epochs=20)  # 总共训练 20 个 epoch
# 可视化配置（TensorBoard）
visualizer = dict(
    type='TensorboardVisualizer',
    log_dir='work_dirs/mobilenet_v2_flower'
)
