# 自动备份当前使用的特征数据库 (embeddings_tta.pkl)。
# 高效地扫描数据库，精确找出并统计所有记录中对应的图片文件已不存在的“无效”条目。
# 一次性地移除所有这些无效条目及其对应的特征向量。
# 保存一个清理过的、100%健康的干净数据库，并保留原始备份以供恢复。

# clean_database.py
import pickle
import numpy as np
import os
import shutil
import sys

# --- 1. 配置 ---
# 同步目标文件为使用TTA技术的数据库
EMBEDDINGS_FILE = 'embeddings_tta.pkl'
# 备份文件名
BACKUP_FILE = 'embeddings_tta.pkl.bak'

# --- 2. 核心功能函数 ---

def clean_database(db_path, backup_path):
    """
    检查并清理数据库文件，移除所有指向不存在文件的记录。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(db_path):
        print(f"数据库文件 '{db_path}' 不存在，无需清理。")
        return

    # 2. 创建备份
    try:
        print(f"正在备份当前数据库到 '{backup_path}'...")
        shutil.copy(db_path, backup_path)
    except Exception as e:
        print(f"错误：创建备份失败: {e}")
        return

    # 3. 加载数据库
    print("正在加载数据库...")
    try:
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
        embeddings = data['embeddings']
        filenames = data['filenames']
        original_count = len(filenames)
        print(f"加载了 {original_count} 条记录。")
    except Exception as e:
        print(f"错误：加载数据库文件失败: {e}")
        return

    # 4. 高效地找出所有有效的记录
    # 使用列表推导式和enumerate来同时获取索引和路径
    valid_indices = [i for i, path in enumerate(filenames) if os.path.exists(path)]
    
    invalid_count = original_count - len(valid_indices)

    if invalid_count == 0:
        print("数据库很干净，没有找到无效记录。")
        os.remove(backup_path)  # 删除多余的备份
        print(f"已删除备份文件 '{backup_path}'。")
        return
        
    print(f"发现 {invalid_count} 条无效记录，将予以移除。")

    # 5. 使用布尔/高级索引创建新的干净数据，这比np.delete更高效
    cleaned_embeddings = embeddings[valid_indices]
    cleaned_filenames = [filenames[i] for i in valid_indices]

    # 6. 保存清理后的数据库
    print("正在保存清理后的数据库...")
    try:
        with open(db_path, 'wb') as f:
            pickle.dump({'embeddings': cleaned_embeddings, 'filenames': cleaned_filenames}, f)
    except Exception as e:
        print(f"错误：保存清理后的数据库失败: {e}")
        # 如果保存失败，可以考虑从备份中恢复
        print("操作失败，请检查错误并可从备份文件中恢复。")
        return

    print("\n--- 清理完成 ---")
    print(f"移除了 {invalid_count} 条无效记录。")
    print(f"数据库现在包含 {len(cleaned_filenames)} 条有效记录。")
    print(f"原始数据库已备份为 '{backup_path}'。")

# --- 3. 主程序入口 ---
def main():
    """主执行函数"""
    clean_database(EMBEDDINGS_FILE, BACKUP_FILE)

if __name__ == '__main__':
    main()