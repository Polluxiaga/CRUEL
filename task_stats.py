import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, isnan
from pathlib import Path

ROOT_FOLDER = "/home/yzc/CRUEL/data"  # 修改为你的数据文件夹路径

def calculate_speed(csv_path):
    """计算速度并返回(速度, 坐标数组)"""
    try:
        df = pd.read_csv(csv_path, header=None)
        if len(df) < 2:
            return None, None
        
        if df.shape[1] != 2 or not all(df.apply(pd.to_numeric, errors='coerce').notna().all()):
            return None, None
        
        coords = df.values
        first = coords[0]
        last = coords[-1]
        distance = sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
        
        if distance == 0:
            return 0.0, coords
        
        speed = distance * 5 / len(df)
        if isnan(speed):
            return None, None
        return speed, coords
    except Exception as e:
        print(f"处理 {csv_path} 出错: {e}")
        return None, None

def process_folder(root_folder):
    """处理文件夹并返回分类数据，并打印_f_和_s_中速度过低的文件路径"""
    categories = {
        '_f_': {'speeds': [], 'coords': [], 'paths': []},
        '_s_': {'speeds': [], 'coords': [], 'paths': []},
        '_x_': {'speeds': [], 'coords': [], 'paths': []}
    }

    for subdir in Path(root_folder).rglob('*'):
        if subdir.is_dir():
            folder_name = subdir.name
            csv_path = subdir / 'traj_data.csv'
            
            if csv_path.exists():
                speed, coords = calculate_speed(csv_path)
                
                if speed is not None:
                    if '_f_' in folder_name:
                        cat = '_f_'
                    elif '_s_' in folder_name:
                        cat = '_s_'
                    elif '_x_' in folder_name:
                        cat = '_x_'
                    else:
                        continue
                    
                    categories[cat]['speeds'].append(speed)
                    categories[cat]['coords'].append(coords)
                    categories[cat]['paths'].append(str(csv_path))

    return categories

def plot_results(categories):
    """绘制速度比较图和轨迹图"""
    plt.figure(figsize=(15, 6))
    
    # 速度比较图
    plt.subplot(1, 2, 1)
    colors = {'_f_': 'blue', '_s_': 'green', '_x_': 'red'}
    labels = {'_f_': 'Fast', '_s_': 'Slow', '_x_': 'Mixed'}
    
    for cat, data in categories.items():
        speeds = data['speeds']
        if speeds:
            speeds = [s for s in speeds if not np.isnan(s)]
            if not speeds:
                continue
            mean = np.mean(speeds)
            std = np.std(speeds)
            
            x = np.random.normal(1 + list(categories.keys()).index(cat), 0.04, len(speeds))
            plt.scatter(x, speeds, alpha=0.5, color=colors[cat], label=labels[cat])
            plt.axhline(mean, color=colors[cat], linestyle='--')
            
            print(f"{labels[cat]} 类别: 平均速度 = {mean:.4f}, 标准差 = {std:.4f}, 样本数 = {len(speeds)}")
    
    plt.xticks([1, 2, 3], [labels[k] for k in categories.keys()])
    plt.title('speed')
    plt.legend()
    plt.grid(True)
    
    # 轨迹图
    plt.subplot(1, 2, 2)
    for cat, data in categories.items():
        if data['coords']:
            for coords in data['coords']:
                if coords is not None:
                    plt.plot(coords[:, 0], coords[:, 1], 
                             color=colors[cat], alpha=0.3, linewidth=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('trajectory')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"开始分析文件夹: {ROOT_FOLDER}")
    categories = process_folder(ROOT_FOLDER)
    plot_results(categories)