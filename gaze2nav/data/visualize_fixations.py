"""Draw fixation boxes on frame images for quick visual inspection."""

import argparse
import os

import cv2
import pandas as pd


RECT_SIZE = 15
COLOR = (255, 255, 255)  # BGR white.
BORDER_WIDTH = 1


def draw_fixation_boxes(root_folder: str) -> None:
    """Overlay one bounded rectangle per fixation point in each trajectory folder."""
    for subdir, _, files in os.walk(root_folder):
        if "fixations.csv" not in files:
            continue

        csv_path = os.path.join(subdir, "fixations.csv")
        try:
            df = pd.read_csv(csv_path, header=None)

            for index, row in df.iterrows():
                x, y = int(row[0]), int(row[1])
                image_path = os.path.join(subdir, f"{index}.jpg")

                if not os.path.exists(image_path):
                    print(f"图像不存在: {image_path}")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"读取失败: {image_path}")
                    continue

                height, width = image.shape[:2]
                half_size = RECT_SIZE // 2

                left = x - half_size
                top = y - half_size
                right = x + half_size
                bottom = y + half_size

                if left < 0:
                    right -= left
                    left = 0
                if right >= width:
                    left -= (right - width + 1)
                    right = width - 1
                if top < 0:
                    bottom -= top
                    top = 0
                if bottom >= height:
                    top -= (bottom - height + 1)
                    bottom = height - 1

                if (right - left) >= BORDER_WIDTH * 2 and (bottom - top) >= BORDER_WIDTH * 2:
                    cv2.rectangle(image, (left, top), (right, bottom), COLOR, BORDER_WIDTH)
                    cv2.imwrite(image_path, image)
                    print(f"写入安全矩形边框: {image_path} (中心: {x},{y} 调整后: [{left},{top}]-[{right},{bottom}])")
                else:
                    print(f"跳过过小矩形: {image_path} (中心: {x},{y} 可用空间不足)")

        except Exception as e:
            print(f"处理失败: {csv_path}，原因: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw fixation boxes on image frames.")
    parser.add_argument("--root", default="./data_vis", help="Root folder containing trajectory subfolders.")
    args = parser.parse_args()

    draw_fixation_boxes(args.root)
