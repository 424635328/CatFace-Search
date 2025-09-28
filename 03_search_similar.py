# 接收一张你指定的查询猫脸图片。
# 利用预先建立好的猫脸特征数据库 (embeddings.pkl) 和超快速的FAISS索引。
# 在数据库中瞬间找出与查询图片在视觉上最相似的几张猫脸。
# 最后，将你的查询图片和找到的相似图片结果一起可视化地展示出来。

# 03_search_similar.py
import faiss
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# --- 配置 ---
EMBEDDINGS_FILE = 'embeddings.pkl'
QUERY_IMAGE_PATH = 'query\\p1.jpg' # 把你要查询的猫脸图片放在这里！
TOP_K = 5 # 找出最相似的5张图片

# --- 加载Embeddings和文件名 ---
print("加载Embeddings...")
with open(EMBEDDINGS_FILE, 'rb') as f:
    data = pickle.load(f)
embeddings = data['embeddings']
filenames = data['filenames']

# --- 建立FAISS索引 ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) # 使用L2距离
# 或者使用余弦相似度: faiss.IndexFlatIP(dimension)，但需要先对向量进行归一化
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"FAISS索引建立完毕，包含 {index.ntotal} 个向量。")

# --- 加载并处理查询图片 ---
# 复用和生成时完全一样的模型和预处理流程
import torch
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

print(f"处理查询图片: {QUERY_IMAGE_PATH}")
query_img = Image.open(QUERY_IMAGE_PATH).convert('RGB')
query_tensor = preprocess(query_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    query_embedding = model(query_tensor).squeeze().cpu().numpy()

query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
faiss.normalize_L2(query_embedding) # 同样需要归一化

# --- 执行搜索 ---
distances, indices = index.search(query_embedding, TOP_K)

# --- 展示结果 ---
print(f"查询结果 (Top {TOP_K}):")
results_paths = [filenames[i] for i in indices[0]]

# 显示查询图片
plt.figure(figsize=(15, 5))
plt.subplot(1, TOP_K + 1, 1)
plt.imshow(query_img)
plt.title("Query Image")
plt.axis('off')

# 显示搜索结果
for i, (path, dist) in enumerate(zip(results_paths, distances[0])):
    print(f"{i+1}. 路径: {path}, 距离: {dist:.4f}")
    result_img = Image.open(path)
    plt.subplot(1, TOP_K + 1, i + 2)
    plt.imshow(result_img)
    plt.title(f"Result {i+1}\nDist: {dist:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()