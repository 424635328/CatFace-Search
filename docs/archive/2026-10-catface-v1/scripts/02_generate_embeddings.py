# 从本地文件加载预训练好的ResNet-50模型，无需网络连接。
# 高效地将cropped_faces文件夹里的所有猫脸图片分批次读入。
# 对每张图片及其水平翻转版本都生成特征向量，然后取平均（TTA），以获得更稳定、更鲁棒的“面部指纹”。
# 最终，将所有生成的“指纹”及其文件名打包存入数据库文件 (embeddings_tta.pkl)。

# 02_generate_embeddings.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import torchvision.transforms.functional as F
import sys

# --- 1. 配置 ---
# 确保这个路径指向你高质量的裁剪图片文件夹
CROPPED_DIR = 'cat_retrieval/cropped_faces' 
# 建议保存为新文件
EMBEDDINGS_FILE = 'embeddings_tta.pkl'
# 本地模型权重路径
MODEL_WEIGHTS_PATH = 'pretrained_models/resnet50-weights.pth'
# 批处理大小，根据你的GPU显存调整。常见的有16, 32, 64
BATCH_SIZE = 16
# DataLoader使用的CPU核心数
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}, 批处理大小: {BATCH_SIZE}, CPU核心数: {NUM_WORKERS}")


# --- 2. 自定义数据集类 ---
class CatFaceDataset(Dataset):
    """
    一个自定义的数据集类，用于加载猫脸图片。
    它会返回原始图片和其水平翻转版本。
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 应用基础变换
            original_img_tensor = self.transform(img)
            # 对原始PIL Image进行翻转，再应用变换
            flipped_img_tensor = self.transform(F.hflip(img))
            
            # 返回一个包含两个tensor的元组和原始路径
            return (original_img_tensor, flipped_img_tensor), img_path
        except Exception as e:
            print(f"警告: 加载或处理图片失败 {img_path}: {e}")
            # 返回None，以便在后续处理中跳过
            return None, None

def collate_fn(batch):
    """
    自定义的collate_fn，用于过滤掉数据集中加载失败的项(None)。
    """
    # 过滤掉返回 (None, None) 的样本
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    # 将过滤后的数据解包并重新组合
    tensors, paths = zip(*batch)
    original_tensors, flipped_tensors = zip(*tensors)
    
    # 将tensor堆叠成批次
    original_batch = torch.stack(original_tensors)
    flipped_batch = torch.stack(flipped_tensors)
    
    return (original_batch, flipped_batch), paths


# --- 3. 核心功能函数 ---
def get_model(weights_path):
    """从本地路径加载模型"""
    print(f"正在从本地路径加载ResNet-50模型: {weights_path}")
    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load(weights_path))
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(DEVICE)
    model.eval()
    print("模型加载成功！")
    return model

def generate_embeddings_in_batches(model, data_loader):
    """使用批处理和TTA生成所有embeddings"""
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for (original_batch, flipped_batch), paths in tqdm(data_loader, desc="生成TTA特征 (批处理)"):
            if original_batch is None: continue # 跳过空的批次

            # 将原始图片和翻转图片拼接成一个更大的batch
            # [BATCH_SIZE, 3, 224, 224] + [BATCH_SIZE, 3, 224, 224] -> [2*BATCH_SIZE, 3, 224, 224]
            combined_batch = torch.cat([original_batch, flipped_batch]).to(DEVICE)

            # 一次性推理，得到所有embeddings
            # 输出形状: [2*BATCH_SIZE, 2048, 1, 1]
            combined_embeddings = model(combined_batch).squeeze() # -> [2*BATCH_SIZE, 2048]
            
            # 分割回原始和翻转的embeddings
            # original_embs: [BATCH_SIZE, 2048], flipped_embs: [BATCH_SIZE, 2048]
            original_embs, flipped_embs = torch.chunk(combined_embeddings, 2)
            
            # 对每对embedding取平均
            # 形状: [BATCH_SIZE, 2048]
            avg_embs = (original_embs + flipped_embs) / 2.0
            
            all_embeddings.append(avg_embs.cpu().numpy())
            all_filenames.extend(paths)
            
    # 将所有批次的结果合并成一个大的Numpy数组
    if not all_embeddings:
        return np.array([]), []
        
    return np.vstack(all_embeddings), all_filenames

def main():
    """主执行函数"""
    # 检查文件和目录
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"错误: 本地模型权重文件 '{MODEL_WEIGHTS_PATH}' 不存在。请先运行 `download_model.py`。")
        sys.exit(1)
    if not os.path.isdir(CROPPED_DIR) or not os.listdir(CROPPED_DIR):
        print(f"错误: 文件夹 '{CROPPED_DIR}' 不存在或为空。")
        sys.exit(1)

    # 加载模型
    model = get_model(MODEL_WEIGHTS_PATH)
    
    # 准备数据加载器
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_paths = glob(os.path.join(CROPPED_DIR, '*.[jp][pn]g')) + glob(os.path.join(CROPPED_DIR, '*.jpeg'))
    dataset = CatFaceDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # 生成Embeddings
    print(f"找到 {len(image_paths)} 张裁剪后的猫脸，开始生成Embeddings...")
    embeddings, filenames = generate_embeddings_in_batches(model, data_loader)
    
    if embeddings.size == 0:
        print("错误：未能生成任何有效的Embeddings。请检查图片文件和路径。")
        sys.exit(1)
        
    # 保存结果
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'filenames': filenames}, f)
        
    print(f"\nTTA Embeddings生成完毕，并保存到 {EMBEDDINGS_FILE}")
    print(f"总共生成了 {embeddings.shape[0]} 个向量，每个向量维度为 {embeddings.shape[1]}")

# --- 4. 主程序入口 ---
if __name__ == '__main__':
    # 在Windows上使用多进程DataLoader需要这个入口保护
    main()