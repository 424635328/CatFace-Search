# CatFace Search: 猫脸视觉相似性搜索引擎

## 效果展示
<img width="2459" height="1234" alt="bd2030c4-9fd0-4aac-baba-d775c89ac6a7" src="https://github.com/user-attachments/assets/f523cffa-118a-4a43-b38b-4e6b89a2bb9b" />
<img width="2335" height="1172" alt="cb94c72e-e5c7-4efb-b708-8f915f452c9e" src="https://github.com/user-attachments/assets/1375236d-1b0e-4623-a1d2-638778da4a73" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Tool](https://img.shields.io/badge/Tool-YOLOv9-red.svg)
![Tool](https://img.shields.io/badge/Tool-FAISS-green.svg)

这是一个基于深度学习的“以图搜图”引擎，专门用于在大型数据集中快速查找视觉上相似的猫脸。项目核心特点在于其**可扩展性**：当有新的猫咪照片加入时，**无需重新训练任何模型**，只需更新索引即可。

## 🧠 核心原理：从“识别”到“测量相似度”

传统的身份识别系统通常需要“监督学习”，即用大量标注好身份（例如，“猫 A”、“猫 B”）的数据去训练一个分类模型。这种方法的缺点是每当有新猫加入，就需要重新训练整个模型。

本项目巧妙地绕开了这个问题，其核心思想是**将“识别问题”转化为“在高维空间中测量相似度”的数学问题**：

1.  **我们不“训练”模型去认识猫，而是“利用”一个“视觉专家”**：
    我们使用了一个在 ImageNet（包含数百万张图片）上预训练好的`ResNet-50`模型。这个模型已经学会了如何理解图像的通用视觉特征——从简单的边缘、纹理到复杂的部件（眼睛、鼻子、毛皮图案）。它就像一个通用的“视觉翻译官”。

2.  **将图片“翻译”成“面部指纹”**：
    我们利用这个“翻译官”将每一张猫脸图片，转换（或“翻译”）成一个 2048 维的数字向量，即**特征向量 (Embedding)**。这个向量可以被看作是这张脸独一无二的“面部指纹”。

3.  **在“特征空间”中寻找最近邻**：
    这个 2048 维的向量可以被想象成一个点在拥有 2048 个坐标轴的超高维“特征空间”中的位置。这个空间有一个神奇的特性：

    - **视觉上相似的图片，其对应的点在空间中会非常接近。**
    - **视觉上差异大的图片，其对应的点会相距遥远。**

4.  **“识别”效果的涌现**：
    系统本身并不知道“身份”的概念。但基于一个强假设——**“同一只猫的不同照片，其视觉特征的相似度远高于它和其它猫的相似度”**——当我们用一张猫的照片去搜索时，系统通过`FAISS`库在这个高维空间中快速找到距离最近的点。这些“最近邻”在绝大多数情况下都恰好是同一只猫的其他照片。
    - **对用户来说**：系统“认出”了这只猫。
    - **对系统来说**：它只是完成了一次纯粹的数学距离计算。

通过这种方式，我们构建了一个无需身份标注、可无限扩展的视觉搜索引擎，完美地解决了新成员加入需要重训练的难题。

## ✨ 项目特点

- **🚀 自动化数据预处理**: 使用先进的 **YOLOv9** 模型自动从原始图片中检测并裁剪出猫脸。
- **🧠 深度特征提取**: 利用在 ImageNet 上预训练的 **ResNet-50** 模型，将每张猫脸图片转换为一个高维的特征向量（Embedding），即“面部指- 纹”。
- **⚡️ 闪电般快速的搜索**: 集成 Facebook AI 的 **FAISS** 库，为数万甚至数百万的特征向量建立高效索引，实现毫秒级的相似性搜索。
- **📈 强大的可扩展性**: 无需重新训练，即可轻松向数据库中添加新的猫脸图片，系统能够智能地进行增量更新。
- **🛠️ 易于维护**: 提供独立的工具脚本用于数据库的创建、更新和清理。

## 🏛️ 系统架构（工作流程）

本系统采用现代的度量学习和内容检索思想，而非传统的分类模型。整个流程分为三个阶段：

1.  **猫脸检测与裁剪 (`01_crop_faces.py`)**:

    - 输入：包含各种猫咪照片的文件夹 (`raw_images/`)。
    - 过程：YOLOv9 模型扫描每张图片，精确定位猫的位置，并裁剪出标准化的猫脸图片。
    - 输出：一个只包含干净、对齐的猫脸照片的文件夹 (`cropped_faces/`)。

2.  **特征提取与建库 (`02_generate_embeddings.py`)**:

    - 输入：裁剪好的猫脸照片 (`cropped_faces/`)。
    - 过程：一个去掉了分类层的 ResNet-50 模型作为通用的特征提取器，将每张猫脸图片“翻译”成一个 2048 维的数字向量（Embedding）。
    - 输出：一个数据库文件 (`embeddings.pkl`)，其中包含了所有猫脸的特征向量及其对应的文件名。

3.  **索引与搜索 (`03_search_similar.py`)**:
    - 输入：一张待查询的猫脸图片。
    - 过程：
      1.  加载数据库文件，并使用 FAISS 为所有特征向量建立一个超快速的搜索索引。
      2.  将查询图片通过同样的特征提取流程，得到其特征向量。
      3.  在 FAISS 索引中，以近乎瞬时的速度找出与查询向量最接近的 N 个向量。
    - 输出：可视化地展示查询图片和找到的最相似的猫脸图片。

---

## 📂 项目结构

```
cat-face-search/
├── yolov9/                   # YOLOv9 官方仓库代码
│   ├── yolov9-e.pt           # YOLOv9 预训练权重 (需自行下载)
│   └── ...
├── raw_images/               # 存放原始猫咪图片
├── cropped_faces/            # 存放由YOLOv9裁剪出的猫脸
├── query/                    # 存放用于搜索的查询图片
│
├── 01_crop_faces.py          # 脚本：自动化裁剪猫脸
├── 02_generate_embeddings.py # 脚本：从头创建特征数据库
├── 03_search_similar.py      # 脚本：执行一次相似性搜索
├── 04_update_database.py     # 脚本：智能地增量更新数据库
├── clean_database.py         # 脚本：清理数据库中的无效记录
│
├── embeddings.pkl            # 生成的特征数据库文件
└── README.md                 # 本说明文件
```

---

## 🚀 安装与配置

请按照以下步骤配置项目运行环境。推荐使用 `Anaconda` 或 `Miniconda` 创建独立的虚拟环境。

### 1. 先决条件

- Git
- Python (3.9 或更高版本)
- Anaconda 或 Miniconda

### 2. 克隆仓库并设置环境

```bash
# 1. 克隆本项目和YOLOv9仓库
git clone https://github.com/424635328/cat-face-search.git
cd cat-face-search
git clone https://github.com/WongKinYiu/yolov9.git

# 2. 创建并激活Conda虚拟环境
conda create --name cat-vision python=3.10 -y
conda activate cat-vision

# 3. 安装PyTorch
# 访问 PyTorch官网 (https://pytorch.org/get-started/locally/)
# 获取适合你系统的安装命令。例如，使用CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装其他核心依赖
# 重要：由于兼容性问题，我们需手动指定NumPy版本<2.0
pip install "numpy<2.0"
pip install opencv-python tqdm matplotlib pillow

# 5. 安装FAISS
# 如果你有NVIDIA GPU (推荐)
conda install -c pytorch faiss-gpu
# 如果你只有CPU
# conda install -c pytorch faiss-cpu

# 6. 安装YOLOv9的依赖
cd yolov9
pip install -r requirements.txt
cd ..
```

### 3. 下载预训练模型

从 [YOLOv9 Release 页面](https://github.com/WongKinYiu/yolov9/releases) 下载一个预训练权重文件，例如 `yolov9-e.pt`，并将其放入 `yolov9/` 文件夹内。

---

## ⚙️ 使用方法

按照以下流程来构建和使用你的猫脸搜索引擎。

### 第 1 步: 准备原始图片

将你收集的所有猫咪图片（文件名随意，格式为 jpg/png/jpeg）放入 `raw_images/` 文件夹。

### 第 2 步: 裁剪所有猫脸

运行 `01_crop_faces.py` 脚本，它会自动处理 `raw_images/` 中的所有图片，并将裁剪出的猫脸保存到 `cropped_faces/`。

```bash
python 01_crop_faces.py
```

### 第 3 步: 建立初始特征数据库

运行 `02_generate_embeddings.py` 脚本。它会为 `cropped_faces/` 文件夹中的**所有**图片生成特征向量，并创建一个新的 `embeddings.pkl` 文件。

```bash
python 02_generate_embeddings.py
```

### 第 4 步: 执行相似性搜索

1.  将一张你想要搜索的猫脸图片放入 `query/` 文件夹。
2.  修改 `03_search_similar.py` 脚本中的 `QUERY_IMAGE_PATH` 变量，使其指向你的查询图片。
3.  运行脚本，稍等片刻，结果将会以图片形式展示。

```bash
python 03_search_similar.py
```

---

## 🔄 如何添加新的猫咪图片 (工作流)

这是本项目的核心优势所在。当你有新的猫咪照片时，请遵循以下**增量更新**流程：

1.  **添加新图片**: 将新的原始照片放入 `raw_images/` 文件夹。
2.  **裁剪新面孔**: 再次运行裁剪脚本。这个优化过的脚本会跳过已处理的图片，只裁剪新加入的图片。
    ```bash
    python 01_crop_faces.py
    ```
3.  **智能更新数据库**: 运行 `04_update_database.py` 脚本。它会自动对比并**只为新加入的猫脸**生成特征向量，然后追加到现有的 `embeddings.pkl` 文件中。
    ```bash
    python 04_update_database.py
    ```
    现在，你的搜索引擎就已经包含了这些新猫的信息，可以进行搜索了！

---

## 🛠️ 维护工具

如果因为手动删除图片文件等原因导致数据库记录与实际文件不符，可以运行 `clean_database.py` 脚本来清理无效数据。

```bash
python clean_database.py
```

## 💡 未来可改进的方向

- **Web 界面**: 使用 Flask 或 Gradio 为项目创建一个用户友好的 Web 界面。
- **优化猫脸检测**: 微调一个 YOLO 模型，专门用于猫脸检测而非全身检测，以提高裁剪精度。
- **替换特征提取器**: 尝试更先进的预训练模型（如 EfficientNetV2）或专门为人脸识别设计的模型架构和损失函数（如 ArcFace），以提升特征的区分度。
- **数据库后端**: 对于超大规模数据集，可以考虑将特征向量存储在专门的向量数据库中（如 Milvus, Weaviate）。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 授权。
