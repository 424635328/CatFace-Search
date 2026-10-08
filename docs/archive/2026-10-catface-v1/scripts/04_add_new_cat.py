# 自动扫描并找出 cropped_faces 文件夹中所有新添加的、未被处理的猫脸图片。
# 从本地文件加载ResNet-50模型，并使用TTA（测试时增强）技术，以批处理的方式高效地为所有新图片生成鲁棒的特征向量。
# 将这些新生成的特征向量无缝追加到现有的数据库文件 (embeddings_tta.pkl) 中，使其保持最新。

# 04_update_new_cat.py
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
CROPPED_DIR = 'cat_retrieval/cropped_faces'
# 确保我们操作的是使用TTA技术的数据库文件
EMBEDDINGS_FILE = 'embeddings_tta.pkl'
# 本地模型权重路径
MODEL_WEIGHTS_PATH = 'pretrained_models/resnet50-weights.pth'
# 批处理大小，与02号脚本保持一致
BATCH_SIZE = 16
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}, 批处理大小: {BATCH_SIZE}, CPU核心数: {NUM_WORKERS}")


# --- 2. 复用核心类与函数 ---

# 复用在02号优化脚本中定义的Dataset类
class CatFaceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            original_img_tensor = self.transform(img)
            flipped_img_tensor = self.transform(F.hflip(img))
            return (original_img_tensor, flipped_img_tensor), img_path
        except Exception as e:
            print(f"警告: 加载或处理图片失败 {img_path}: {e}")
            return None, None

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    tensors, paths = zip(*batch)
    original_tensors, flipped_tensors = zip(*tensors)
    original_batch = torch.stack(original_tensors)
    flipped_batch = torch.stack(flipped_tensors)
    return (original_batch, flipped_batch), paths

# 复用在02号优化脚本中的批处理生成函数
def generate_embeddings_in_batches(model, data_loader):
    all_embeddings, all_filenames = [], []
    with torch.no_grad():
        for (original_batch, flipped_batch), paths in tqdm(data_loader, desc="为新图片生成TTA特征"):
            if original_batch is None: continue
            combined_batch = torch.cat([original_batch, flipped_batch]).to(DEVICE)
            combined_embeddings = model(combined_batch).squeeze()
            original_embs, flipped_embs = torch.chunk(combined_embeddings, 2)
            avg_embs = (original_embs + flipped_embs) / 2.0
            all_embeddings.append(avg_embs.cpu().numpy())
            all_filenames.extend(paths)
    if not all_embeddings:
        return np.array([]), []
    return np.vstack(all_embeddings), all_filenames

def get_model(weights_path):
    """从本地路径加载模型"""
    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load(weights_path))
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. 主执行函数 ---
def main():
    # 检查必要文件和目录
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"错误: 本地模型权重文件 '{MODEL_WEIGHTS_PATH}' 不存在。请先运行 `download_model.py`。")
        sys.exit(1)
    if not os.path.isdir(CROPPED_DIR):
        print(f"警告: 文件夹 '{CROPPED_DIR}' 不存在。将视为空白数据库进行处理。")
        os.makedirs(CROPPED_DIR)

    # 1. 加载现有数据库，找出已处理的文件
    processed_files = set()
    existing_data = {'embeddings': np.array([]), 'filenames': []}
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"正在加载现有数据库: {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            existing_data = pickle.load(f)
            processed_files = set(existing_data['filenames'])
        print(f"数据库中已有 {len(processed_files)} 个已处理的文件。")
    else:
        print("未找到现有数据库，将创建新文件。")

    # 2. 找出所有待处理的新文件
    all_face_files = set(glob(os.path.join(CROPPED_DIR, '*.[jp][pn]g')))
    new_files_to_process = sorted(list(all_face_files - processed_files))

    if not new_files_to_process:
        print("数据库已是最新，没有新的猫脸照片需要处理。")
        sys.exit(0)
    
    print(f"发现 {len(new_files_to_process)} 个新文件需要处理。")

    # 3. 加载模型并为新文件生成Embeddings
    model = get_model(MODEL_WEIGHTS_PATH)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    new_dataset = CatFaceDataset(new_files_to_process, transform=transform)
    new_data_loader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    new_embeddings, new_filenames = generate_embeddings_in_batches(model, new_data_loader)
    
    if new_embeddings.size == 0:
        print("未能为任何新文件生成有效的Embeddings。数据库未更新。")
        sys.exit(0)

    # 4. 将新数据追加到数据库
    if existing_data['embeddings'].size == 0:
        updated_embeddings = new_embeddings
    else:
        updated_embeddings = np.vstack([existing_data['embeddings'], new_embeddings])
    
    updated_filenames = existing_data['filenames'] + new_filenames

    # 5. 保存更新后的数据库
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': updated_embeddings, 'filenames': updated_filenames}, f)
        
    print("\n--- 数据库更新成功 ---")
    print(f"成功添加了 {len(new_filenames)} 条新记录。")
    print(f"数据库现在总共有 {len(updated_filenames)} 条记录。")

# --- 4. 主程序入口 ---
if __name__ == '__main__':
    main()