import os
import rosbag
import pickle
import numpy as np
from PIL import Image
import io
import torchvision.transforms.functional as TF

# 配置常量
IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3
FRAME_INTERVAL = 6  # 每0.2s提取一次RGB图片
ODOMETER_SAMPLE_INTERVAL = 10  # 每0.2s提取一条Odometry数据

def extract_images_from_bag(bag_path, output_dir):
    """
    从 .bag 文件中提取图像并保存到指定文件夹，并返回第一张图像的时间戳。
    
    参数:
    - bag_path: .bag 文件路径
    - output_dir: 输出文件夹路径
    
    返回:
    - 第一张图像的时间戳
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    first_timestamp = None  # 用于记录第一张图像的时间戳
    img_index = 0  # 从0开始命名
    frame_counter = 0  # 计数器，用来记录已经提取的帧数

    # 打开 .bag 文件
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera/rgb/image_raw/compressed']):
            frame_counter += 1  # 增加帧计数器
            if frame_counter <=99:
                continue
            img_data = msg.data  # msg.data 包含了图像的原始压缩数据
            img_timestamp = t.to_sec()  # 获取图像的时间戳

            # 将压缩数据解码为图像
            img = Image.open(io.BytesIO(img_data))  # 转换为图像
            w, h = img.size
            img = TF.center_crop(img, (h, int(h * IMAGE_ASPECT_RATIO)))
            img = img.resize(IMAGE_SIZE)

            # 每隔 FRAME_INTERVAL 帧提取一次图像
            if (frame_counter-1) % FRAME_INTERVAL == 0:
                if first_timestamp is None:
                    first_timestamp = img_timestamp
                # print(img_timestamp)
                img_filename = f"{img_index}.jpg"  # 使用 img_index 作为文件名（从0开始）
                img.save(os.path.join(output_dir, img_filename))  # 保存图像
                img_index += 1  # 更新计数器


    print(f"提取完成，共提取了 {img_index} 张图像！")
    return first_timestamp  # 返回第一张图像的时间戳


def extract_odometry_from_bag(bag_path, output_dir, first_image_timestamp):
    """
    从 .bag 文件中提取 Odometry 数据并保存到 PKL 文件，
    并确保提取的 Odometry 数据的时间戳从第一张图像的时间戳开始。
    
    参数:
    - bag_path: .bag 文件路径
    - output_dir: 输出文件夹路径
    - first_image_timestamp: 第一张图像的时间戳
    
    返回:
    - None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建 PKL 文件并写入数据
    pkl_filename = os.path.join(output_dir, "traj_data.pkl")
    odom_data_list = []  # 用于存储位置数据的列表

    odom_index = 0  # 用于计数 odometry 数据
    frame_counter = 0  # 用于计数每 10 条数据的间隔
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/jackal_velocity_controller/odom']):
            odom_timestamp = t.to_sec()

            # 如果 Odometry 数据的时间戳大于等于第一张图像的时间戳，才保存
            if odom_timestamp > first_image_timestamp:
                # 每隔 ODOMETER_SAMPLE_INTERVAL 条数据提取一条
                if frame_counter % ODOMETER_SAMPLE_INTERVAL == 0:
                    # 提取位置数据并转换为 numpy array
                    position_x = msg.pose.pose.position.x
                    position_y = msg.pose.pose.position.y
                    position = np.array([position_x, position_y])  # 保存为一个位置数组

                    # 存储数据到列表
                    odom_data = {
                        'timestamp': odom_timestamp,
                        'position': position
                    }
                    odom_data_list.append(odom_data)
                    odom_index += 1  # 更新计数器

                frame_counter += 1  # 增加帧计数器

    # 使用 pickle 保存数据
    with open(pkl_filename, 'wb') as pklfile:
        pickle.dump(odom_data_list, pklfile)

    print(f"提取完成，共提取了 {odom_index} 条 Odometry 数据，已保存为 {pkl_filename}！")


# 设置文件路径
bag_path = '/Users/pollux/Documents/Robot_Navigation/SCAND/B_Jackal_Stadium_Sanjac_Sat_Nov_13_90.bag'
output_dir = '/Users/pollux/Documents/Robot_Navigation/SCAND/B_Jackal_Stadium_Sanjac_Sat_Nov_13_90'

# 提取图像并获取第一张图像的时间戳
first_image_timestamp = extract_images_from_bag(bag_path, output_dir)

# 提取 Odometry 数据，且从第一张图像的时间戳开始提取
extract_odometry_from_bag(bag_path, output_dir, first_image_timestamp)