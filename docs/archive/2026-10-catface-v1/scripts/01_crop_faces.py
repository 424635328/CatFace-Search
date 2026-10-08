# 读取 raw_images 文件夹里的原始、未经处理的图片。
# 使用YOLOv9模型在每张图片中自动定位猫的位置和边界。
# 根据定位结果，智能地裁剪出只包含猫脸的区域。
# 将所有裁剪好的、标准化的猫脸图片保存到 cropped_faces 文件夹中。

# 01_crop_faces.py
import torch
import cv2
import os
from glob import glob
from tqdm import tqdm
import sys

# --- 1. 配置区域 ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

YOLO_DIR = os.path.join(SCRIPT_DIR, 'yolov9')
if not os.path.isdir(YOLO_DIR):
    print(f"错误: YOLOv9目录未在以下路径找到: {YOLO_DIR}")
    sys.exit(1)

WEIGHTS_PATH = os.path.join(YOLO_DIR, 'yolov9-e.pt') 
if not os.path.exists(WEIGHTS_PATH):
    print(f"错误: 权重文件未找到: {WEIGHTS_PATH}")
    sys.exit(1)

# 注意：这里的路径可能需要根据你的实际结构调整
RAW_IMG_DIR = os.path.join(SCRIPT_DIR, 'cat_retrieval/cropped_faces')
CROPPED_DIR = os.path.join(SCRIPT_DIR, 'cat_retrieval/sec_cropped')

CONF_THRESHOLD = 0.3
IMG_SIZE = 640
CAT_CLASS_ID = 15

# --- 2. 初始化 ---
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(RAW_IMG_DIR, exist_ok=True)

# --- 3. 加载YOLOv9模型 ---
print("正在加载YOLOv9模型...")
try:
    model = torch.hub.load(YOLO_DIR, 'custom', path=WEIGHTS_PATH, source='local', trust_repo=True)
    model.conf = CONF_THRESHOLD
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型时发生错误: {e}")
    sys.exit(1)

# --- 4. 遍历图片并处理 ---
image_paths = glob(os.path.join(RAW_IMG_DIR, '*.[jp][pn]g')) + glob(os.path.join(RAW_IMG_DIR, '*.jpeg'))

if not image_paths:
    print(f"警告: 在 '{RAW_IMG_DIR}' 目录中没有找到任何图片。")
    sys.exit(0)

print(f"在 '{RAW_IMG_DIR}' 中找到 {len(image_paths)} 张图片，开始处理...")

total_faces_cropped = 0
total_faces_skipped = 0
for img_path in tqdm(image_paths, desc="裁剪猫脸"):
    try:
        # --- 提前构建潜在的输出文件名来进行检查 ---
        # 我们先检查一下，如果一张图片只有一个猫脸，对应的文件是否已存在。
        # 这是一个初步的快速跳过，对于单猫图片很有效。
        base_filename = os.path.basename(img_path)
        filename, ext = os.path.splitext(base_filename)
        # 检查最常见的情况：_face_0
        potential_save_path = os.path.join(CROPPED_DIR, f"{filename}_face_0{ext}")
        if os.path.exists(potential_save_path):
             # 检查这张图片是否还有其他猫脸（_face_1, _face_2...）
             # 如果没有更复杂的命名，我们可以简单跳过
             has_multiple_faces = any(glob(os.path.join(CROPPED_DIR, f"{filename}_face_[1-9]*.{ext}")))
             if not has_multiple_faces:
                total_faces_skipped += 1
                continue # 跳过这张已经处理过的单猫图片

        # 对于未跳过的图片，正常加载和处理
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, size=IMG_SIZE)
        predictions = results.xyxy[0]
        cat_detections = predictions[predictions[:, 5] == CAT_CLASS_ID]
        
        if len(cat_detections) == 0 and os.path.exists(potential_save_path):
            # 如果模型这次没检测到猫，但之前处理过，也算作跳过
            total_faces_skipped += 1
            continue

        for i, det in enumerate(cat_detections):
            # 1. 构建将要保存的文件名
            base_filename = os.path.basename(img_path)
            filename, ext = os.path.splitext(base_filename)
            save_path = os.path.join(CROPPED_DIR, f"{filename}_face_{i}{ext}")
            
            # 2. 检查这个文件是否已经存在
            if os.path.exists(save_path):
                total_faces_skipped += 1
                continue # 如果存在，就跳过这个猫脸，处理下一个

            # 3. 如果不存在，才执行裁剪和保存
            x1, y1, x2, y2 = map(int, det[:4])
            box_width = x2 - x1
            center_x = x1 + box_width // 2
            head_center_y = y1 + (y2 - y1) // 3
            half_width = box_width // 2
            
            head_x1 = max(0, center_x - half_width)
            head_y1 = max(0, head_center_y - half_width)
            head_x2 = min(img.shape[1], center_x + half_width)
            head_y2 = min(img.shape[0], head_center_y + half_width)
            
            cat_head = img[head_y1:head_y2, head_x1:head_x2]

            if cat_head.size > 0:
                cv2.imwrite(save_path, cat_head)
                total_faces_cropped += 1
            
    except Exception as e:
        print(f"\n处理图片 {img_path} 时发生未知错误: {e}")

print("\n--- 处理完成 ---")
print(f"总共检查了 {len(image_paths)} 张原始图片。")
print(f"跳过了 {total_faces_skipped} 个已经存在的猫脸。")
print(f"新增了 {total_faces_cropped} 张猫脸到 '{CROPPED_DIR}' 目录。")
