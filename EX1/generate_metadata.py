import os
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# 写入 classes.txt
with open('flower_dataset/classes.txt', 'w') as f:
    for cls in classes:
        f.write(cls + '\n')
# 辅助函数：生成 train.txt 和 val.txt
def generate_list(split):
    output_file = f'flower_dataset/{split}.txt'
    with open(output_file, 'w') as f:
        for idx, cls in enumerate(classes):
            cls_folder = f'flower_dataset/{split}/{cls}'
            for filename in os.listdir(cls_folder):
                if filename.endswith('.jpg'):
                    relative_path = f'{cls}/{filename}'
                    f.write(f'{relative_path} {idx}\n')
generate_list('train')
generate_list('val')
print("已生成 classes.txt, train.txt, val.txt！")
