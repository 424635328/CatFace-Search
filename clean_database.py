# 备份当前的特征数据库 (embeddings.pkl) 以防意外。
# 检查数据库中的每一条记录，验证其对应的图片文件路径是否真实存在于磁盘上。
# 自动移除所有无效的记录（例如，路径不存在的文件名或非文件路径的ID）。
# 保存一个清理过的、只包含有效记录的干净数据库。

# clean_database.py
import pickle
import numpy as np
import os

# --- 配置 ---
EMBEDDINGS_FILE = 'embeddings.pkl'
BACKUP_FILE = 'embeddings.pkl.bak' # 创建一个备份以防万一

# --- 检查文件是否存在 ---
if not os.path.exists(EMBEDDINGS_FILE):
    print(f"数据库文件 {EMBEDDINGS_FILE} 不存在，无需清理。")
    exit()

# --- 1. 创建备份 ---
print(f"正在备份当前数据库到 {BACKUP_FILE}...")
import shutil
shutil.copy(EMBEDDINGS_FILE, BACKUP_FILE)

# --- 2. 加载数据库 ---
print("正在加载数据库...")
with open(EMBEDDINGS_FILE, 'rb') as f:
    data = pickle.load(f)
embeddings = data['embeddings']
filenames = data['filenames']
print(f"加载了 {len(filenames)} 条记录。")

# --- 3. 找出并移除无效记录 ---
# 我们要移除所有不是有效文件路径的记录
indices_to_remove = []
cleaned_filenames = []
for i, path in enumerate(filenames):
    # 如果路径不是文件，或者它就是那个特定的ID，就标记为待删除
    if not os.path.exists(path):
        print(f"发现无效记录: '{path}' (索引 {i})，将予以移除。")
        indices_to_remove.append(i)

# 如果没有找到要移除的记录
if not indices_to_remove:
    print("数据库很干净，没有找到无效记录。")
    os.remove(BACKUP_FILE) # 删除多余的备份
    exit()

# --- 4. 创建新的干净数据 ---
# np.delete 可以根据索引移除数组的行
cleaned_embeddings = np.delete(embeddings, indices_to_remove, axis=0)

# 从文件名列表中移除元素
# 我们倒序遍历，这样删除元素时不会影响前面元素的索引
for index in sorted(indices_to_remove, reverse=True):
    del filenames[index]
cleaned_filenames = filenames

# --- 5. 保存清理后的数据库 ---
print(f"正在保存清理后的数据库...")
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump({'embeddings': cleaned_embeddings, 'filenames': cleaned_filenames}, f)

print("\n--- 清理完成 ---")
print(f"移除了 {len(indices_to_remove)} 条无效记录。")
print(f"数据库现在包含 {len(cleaned_filenames)} 条有效记录。")
print(f"原始数据库已备份为 {BACKUP_FILE}。")