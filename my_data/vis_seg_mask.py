import os
import cv2
import numpy as np
import csv

def overlay_mask_on_image(image, mask, color=(0, 0, 0), alpha=0.5):
    """将二值mask叠加到图像上"""
    overlay = image.copy()
    mask_bool = mask.astype(bool)

    # 只对掩码区域叠加颜色
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)

    return overlay.astype(np.uint8)

def visualize_csv_masks(folder_path, save_visualized=False):
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if not file.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file)
            image_name = os.path.splitext(file)[0] + ".jpg"
            image_path = os.path.join(root, image_name)

            if not os.path.exists(image_path):
                print(f"Warning: image {image_name} not found for {file}")
                continue

            # Read image
            image = cv2.imread(image_path)
            H, W = image.shape[:2]

            # Read CSV: skip first row (ID header), read the rest as 0/1
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)  # skip header
                except StopIteration:
                    print(f"Warning: CSV {file} is empty, skipping.")
                    continue  # skip empty CSV

                data = np.array([[int(val) for val in row] for row in reader], dtype=np.uint8)

            # Now data shape is (H*W, N), where N is number of targets
            if data.shape[0] != H * W:
                print(f"Error: CSV shape {data.shape} does not match image size {H}x{W}")
                continue

            # Sum all masks along columns and reshape
            combined_mask = data.sum(axis=1)
            combined_mask = (combined_mask > 0).astype(np.uint8).reshape(H, W)  # binarize

            # Overlay
            result = overlay_mask_on_image(image, combined_mask, color=(0, 0, 255), alpha=0.5)

            if save_visualized:
                out_path = os.path.join(root, f"{os.path.splitext(file)[0]}.jpg")
                cv2.imwrite(out_path, result)
                print(f"Saved visualization to {out_path}")
            else:
                cv2.imshow("Mask Overlay", result)
                key = cv2.waitKey(0)
                if key == 27:  # ESC to exit early
                    cv2.destroyAllWindows()
                    return


if __name__ == "__main__":
    visualize_csv_masks("/home/yzc/CRUEL/data_vis", save_visualized=True)
