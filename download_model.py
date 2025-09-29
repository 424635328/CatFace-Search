# download_model.py
import torchvision.models as models
import torch
import os

# --- 配置 ---
# 定义你想要保存模型权重的文件路径
MODEL_SAVE_DIR = 'pretrained_models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'resnet50-weights.pth')

# --- 创建保存目录 ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def download_and_save_model():
    """
    下载ResNet-50的预训练权重并保存到本地。
    """
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"模型文件已存在于: {MODEL_SAVE_PATH}")
        print("无需重复下载。")
        return

    print("正在从torchvision下载ResNet-50预训练权重...")
    
    # 使用推荐的API加载模型，这将自动下载权重
    # weights=models.ResNet50_Weights.DEFAULT 会获取最新的、最好的权重
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    print("下载完成。")
    
    # 获取模型的状态字典（即所有可学习的参数）
    weights = model.state_dict()
    
    # 将权重保存到文件
    torch.save(weights, MODEL_SAVE_PATH)
    
    print(f"模型权重已成功保存到: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    download_and_save_model()