# 从本地文件一次性加载预训练好的ResNet-50模型，并封装成一个高效的特征提取器。
# 加载使用TTA（测试时增强）技术创建的高鲁棒性特征数据库，并用FAISS构建快速索引。
# 接收一张查询图片，同样使用TTA技术提取其特征，以确保比较的公平性。
# 在数据库中快速找出并可视化显示与查询图片最相似的猫脸。

# 03_search_similar.py
import faiss
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import sys

# --- 1. 配置 ---
# 确保加载的是使用TTA生成的数据库文件
EMBEDDINGS_FILE = 'embeddings_tta.pkl'
# 本地模型权重路径
MODEL_WEIGHTS_PATH = 'pretrained_models/resnet50-weights.pth'
# 把你要查询的猫脸图片放在这里！
QUERY_IMAGE_PATH = 'query\\8dc365c9-a34b-4e2f-807b-9e75be81fd21.png'
TOP_K = 10 # 找出最相似的5张图片

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 核心功能函数 ---

class FeatureExtractor:
    """一个封装了模型加载和特征提取的类"""
    def __init__(self, weights_path):
        self.device = DEVICE
        print(f"正在从本地路径加载ResNet-50模型: {weights_path}")
        self.model = self._load_model(weights_path)
        self.preprocess = self._get_preprocessor()
        print("特征提取器准备就绪！")

    def _load_model(self, weights_path):
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(weights_path))
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.to(self.device)
        model.eval()
        return model

    def _get_preprocessor(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embedding_tta(self, image):
        """对单张图片应用TTA并提取特征"""
        original_tensor = self.preprocess(image)
        flipped_tensor = self.preprocess(F.hflip(image))
        batch_tensor = torch.stack([original_tensor, flipped_tensor]).to(self.device)
        batch_embeddings = self.model(batch_tensor)
        final_embedding = batch_embeddings.squeeze().mean(dim=0).cpu().numpy()
        return final_embedding

def build_faiss_index(embeddings_path):
    """加载数据库并构建FAISS索引"""
    print(f"从 {embeddings_path} 加载Embeddings...")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings'].astype('float32')
    filenames = data['filenames']
    
    print("建立FAISS索引...")
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS索引建立完毕，包含 {index.ntotal} 个向量。")
    return index, filenames

def display_results(query_img, results_paths, similarities, top_k):
    """可视化显示查询结果"""
    print(f"\n查询结果 (Top {top_k}):")
    plt.figure(figsize=(15, 5))
    
    # 显示查询图片
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    # 显示搜索结果
    for i, (path, sim) in enumerate(zip(results_paths, similarities)):
        print(f"{i+1}. 路径: {path}, 相似度: {sim:.4f}")
        ax = plt.subplot(1, top_k + 1, i + 2)
        try:
            result_img = Image.open(path)
            plt.imshow(result_img)
            plt.title(f"Result {i+1}\nSim: {sim:.2f}")
        except FileNotFoundError:
            print(f"  -> 警告: 找不到结果图片文件: {path}")
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            plt.title(f"Result {i+1}\n(Not Found)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    """主执行函数"""
    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [EMBEDDINGS_FILE, MODEL_WEIGHTS_PATH, QUERY_IMAGE_PATH]):
        print("错误: 缺少必要的文件。请检查以下文件是否存在：")
        print(f"  - 数据库: {EMBEDDINGS_FILE}")
        print(f"  - 模型权重: {MODEL_WEIGHTS_PATH}")
        print(f"  - 查询图片: {QUERY_IMAGE_PATH}")
        sys.exit(1)

    # 1. 初始化特征提取器 (一次性加载模型)
    extractor = FeatureExtractor(MODEL_WEIGHTS_PATH)
    
    # 2. 构建FAISS索引
    index, filenames = build_faiss_index(EMBEDDINGS_FILE)
    
    # 3. 处理查询图片
    print(f"处理查询图片: {QUERY_IMAGE_PATH}")
    query_img = Image.open(QUERY_IMAGE_PATH).convert('RGB')
    query_embedding = extractor.get_embedding_tta(query_img)
    
    # 准备FAISS查询
    query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # 4. 执行搜索
    similarities, indices = index.search(query_embedding, TOP_K)
    
    # 5. 展示结果
    results_paths = [filenames[i] for i in indices[0]]
    display_results(query_img, results_paths, similarities[0], TOP_K)

# --- 3. 主程序入口 ---
if __name__ == '__main__':
    main()