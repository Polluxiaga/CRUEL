import pandas as pd
import pickle

def view_pkl_file(file_path):
    """查看pkl文件内容和类型"""
    try:
        # 尝试用pandas读取
        try:
            data = pd.read_pickle(file_path)
            print("=== 数据类型（pandas）===")
            print(type(data))
            
            if isinstance(data, pd.DataFrame):
                print("\n=== 数据概览 ===")
                print(data.head())
                print("\n=== 数据形状 ===")
                print(data.shape)
            else:
                print("\n=== 完整内容 ===")
                print(data)
        
        # 如果pandas读取失败，用原生pickle读取
        except:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                print("=== 数据类型（pickle）===")
                print(type(data))
                print("\n=== 完整内容 ===")
                print(data)
                
    except Exception as e:
        print(f"读取失败: {str(e)}")

# 使用示例
view_pkl_file("/home/yzc/CRUEL/data/fzm1_f_1/person_ids.pkl")  # 替换为你的pkl文件路径