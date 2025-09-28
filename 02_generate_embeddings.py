# 从头开始处理 cropped_faces 文件夹里的所有猫脸图片。
# 利用深度学习模型（ResNet-50）为每一张图片生成一个独特的数字“面部指纹”（Embedding）。
# 最后，将所有这些“指纹”及其对应的文件名打包存入一个全新的数据库文件 (embeddings.pkl) 中。

# 02_generate_embeddings.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle

# --- 配置 ---
CROPPED_DIR = 'cat_retrieval/cropped_faces'
EMBEDDINGS_FILE = 'embeddings.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

# --- 加载预训练的ResNet-50模型 ---
model = models.resnet50(pretrained=True)
# 移除最后一层 (分类层)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(DEVICE)
model.eval() # 设置为评估模式

# --- 定义图像预处理转换 ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 生成Embeddings ---
image_paths = glob(os.path.join(CROPPED_DIR, '*.[jp][pn]g'))
all_embeddings = []
all_filenames = []

print(f"找到 {len(image_paths)} 张裁剪后的猫脸，开始生成Embeddings...")

with torch.no_grad(): # 关闭梯度计算，加速推理
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE) # 增加一个batch维度
            
            embedding = model(img_tensor)
            
            # ResNet-50输出是 (1, 2048, 1, 1)，需要展平
            embedding_flat = embedding.squeeze().cpu().numpy()
            
            all_embeddings.append(embedding_flat)
            all_filenames.append(img_path)
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")

# 将列表转换为Numpy数组
all_embeddings = np.array(all_embeddings, dtype='float32')

# 保存结果
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump({'embeddings': all_embeddings, 'filenames': all_filenames}, f)

print(f"Embeddings生成完毕，并保存到 {EMBEDDINGS_FILE}")
print(f"总共生成了 {all_embeddings.shape[0]} 个向量，每个向量维度为 {all_embeddings.shape[1]}")