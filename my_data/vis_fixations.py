import os
import cv2
import pandas as pd

# 矩形尺寸和颜色设定
RECT_SIZE = 15  # 15x15像素的矩形
COLOR = (255, 255, 255)  # 白色（BGR格式）
BORDER_WIDTH = 1      # 边框宽度

# 根目录路径（修改为你的路径）
root_folder = r"./data_vis"  # <- 修改为实际路径

for subdir, _, files in os.walk(root_folder):
    if 'fixations.csv' in files:
        csv_path = os.path.join(subdir, 'fixations.csv')
        try:
            # 读取fixations.csv，假设无表头
            df = pd.read_csv(csv_path, header=None)

            for index, row in df.iterrows():
                x, y = int(row[0]), int(row[1])
                image_name = f"{index}.jpg"
                image_path = os.path.join(subdir, image_name)

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"读取失败: {image_path}")
                        continue

                    # 获取图像尺寸
                    height, width = image.shape[:2]
                    
                    # 计算矩形边界（确保不超出图像范围）
                    half_size = RECT_SIZE // 2
                    
                    # 计算初始矩形坐标
                    left = x - half_size
                    top = y - half_size
                    right = x + half_size
                    bottom = y + half_size
                    
                    # 调整超出边界的坐标
                    if left < 0:
                        right -= left  # 向右移动
                        left = 0
                    if right >= width:
                        left -= (right - width + 1)
                        right = width - 1
                    if top < 0:
                        bottom -= top  # 向下移动
                        top = 0
                    if bottom >= height:
                        top -= (bottom - height + 1)
                        bottom = height - 1
                    
                    # 确保矩形至少保留BORDER_WIDTH的可见部分
                    if (right - left) >= BORDER_WIDTH*2 and (bottom - top) >= BORDER_WIDTH*2:
                        # 在图像上画矩形边框
                        cv2.rectangle(image, (left, top), (right, bottom), COLOR, BORDER_WIDTH)
                        
                        # 覆盖保存图像
                        cv2.imwrite(image_path, image)
                        print(f"写入安全矩形边框: {image_path} (中心: {x},{y} 调整后: [{left},{top}]-[{right},{bottom}]")
                    else:
                        print(f"跳过过小矩形: {image_path} (中心: {x},{y} 可用空间不足)")

                else:
                    print(f"图像不存在: {image_path}")

        except Exception as e:
            print(f"处理失败: {csv_path}，原因: {e}")