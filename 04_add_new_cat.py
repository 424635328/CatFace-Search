# 扫描 cropped_faces 文件夹，并与现有的数据库 (embeddings.pkl) 进行对比。
# 找出所有新添加的、尚未被处理过的猫脸图片。
# 只为这些新图片生成特征向量（Embeddings）。
# 将新生成的特征向量追加到现有数据库的末尾，并保存。

# 04_update_database.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import sys

# --- 1. 配置 ---
CROPPED_DIR = 'cat_retrieval/cropped_faces'
EMBEDDINGS_FILE = 'embeddings.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")

# --- 2. 加载现有数据库，找出已处理的文件 ---
processed_files = set()
if os.path.exists(EMBEDDINGS_FILE):
    print(f"正在加载现有数据库: {EMBEDDINGS_FILE}")
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
        # 假设 filenames 列表与 embeddings 数组一一对应
        processed_files = set(data['filenames'])
    print(f"数据库中已有 {len(processed_files)} 个已处理的文件。")
else:
    print("未找到现有数据库，将创建新文件。")
    data = {'embeddings': np.array([]), 'filenames': []}


# --- 3. 找出所有待处理的新文件 ---
all_face_files = set(glob(os.path.join(CROPPED_DIR, '*.[jp][pn]g')))
new_files_to_process = sorted(list(all_face_files - processed_files)) # 排序以保证顺序

if not new_files_to_process:
    print("数据库已是最新，没有新的猫脸照片需要处理。")
    sys.exit(0)

print(f"发现 {len(new_files_to_process)} 个新文件需要处理。")

# --- 4. 加载模型和预处理器 (与02号脚本完全相同) ---
print("加载特征提取模型...")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 5. 为新文件生成Embeddings ---
new_embeddings = []
processed_new_files = [] # 只保存成功处理的文件名

with torch.no_grad():
    for img_path in tqdm(new_files_to_process, desc="生成新特征"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            embedding = model(img_tensor)
            embedding_flat = embedding.squeeze().cpu().numpy()
            new_embeddings.append(embedding_flat)
            processed_new_files.append(img_path)
        except Exception as e:
            print(f"\n处理图片 {img_path} 时出错: {e}, 跳过此文件。")

if not new_embeddings:
    print("未能为任何新文件生成特征向量。")
    sys.exit(0)

# --- 6. 将新数据追加到数据库 ---
new_embeddings_np = np.array(new_embeddings, dtype='float32')

# 如果现有数据库为空，则直接使用新数据
if data['embeddings'].size == 0:
    updated_embeddings = new_embeddings_np
else:
    updated_embeddings = np.vstack([data['embeddings'], new_embeddings_np])

updated_filenames = data['filenames'] + processed_new_files

# --- 7. 保存更新后的数据库 ---
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump({'embeddings': updated_embeddings, 'filenames': updated_filenames}, f)

print("\n--- 数据库更新成功 ---")
print(f"成功添加了 {len(new_embeddings)} 条新记录。")
print(f"数据库现在总共有 {len(updated_filenames)} 条记录。")
print("你可以重新运行 03_search_similar.py 来使用更新后的数据库进行搜索。")