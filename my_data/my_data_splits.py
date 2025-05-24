import os
import shutil
import random


def remove_files_in_dir(dir_path: str):
    """
    删除指定目录下的所有文件和子目录。
    """
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

def find_traj_folders(data_dir: str):
    """
    递归查找包含 'traj_data.pkl' 文件的文件夹，并返回相对于 data_dir 的路径。
    """
    traj_folders = []
    for root, dirs, files in os.walk(data_dir):
        if "traj_data.pkl" in files:
            traj_folders.append(os.path.relpath(root, data_dir))  # 使用相对路径
    return traj_folders

def process_data(data_dir: str, dataset_name: str, data_splits_dir: str, split: float = 0.8):
    """
    从指定目录提取数据，分割成训练集和测试集，并保存到指定的目标文件夹中。

    参数:
    - data_dir: 包含数据子目录的目录路径
    - dataset_name: 数据集名称，用于创建目标文件夹
    - split: 训练集/测试集的划分比例，默认80%训练集，20%测试集
    - data_splits_dir: 数据分割后保存的目标目录
    """
    # 获取包含 'traj_data.pkl' 文件的文件夹名称
    folder_names = find_traj_folders(data_dir)

    # 随机打乱文件夹名称
    random.shuffle(folder_names)

    # 按照 split 比例划分训练集和测试集
    split_index = int(split * len(folder_names))
    train_folder_names = folder_names[:split_index]
    test_folder_names = folder_names[split_index:]

    # 创建训练集和测试集目录
    train_dir = os.path.join(data_splits_dir, dataset_name, "train")
    test_dir = os.path.join(data_splits_dir, dataset_name, "test")

    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    # 保存训练集和测试集的文件夹名称到 traj_names.txt
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name + "\n")

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")

    print("Data processing complete!")

# 设置固定路径
data_dir = '/home/yzc/CRUEL/data'
dataset_name = 'data_splits'  # 自定义数据集名称
data_splits_dir = '/home/yzc/CRUEL'  # 目标目录
split = 0.8  # 训练集和测试集的划分比例，默认80%训练，20%测试

# 调用处理数据函数
process_data(data_dir, dataset_name, data_splits_dir, split)

print("Done")