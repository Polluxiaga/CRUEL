"""Convert generated gaze maps into semantic salient-person IDs.

This is the Semantic Saliency Matching stage described in the paper: predicted
gaze peaks are matched against tracked pedestrian masks, then accumulated in a
short sliding window to produce 2phase/2phaseplus labels for the action model.
"""

import os
import argparse
import pandas as pd
import re
from collections import deque, Counter
import torch

# --- Configuration (保持不变) ---
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 128
SLIDING_WINDOW_SIZE = 15
# --- End Configuration ---

def extract_peak_fixations_from_gaze_maps(pt_file_path: str) -> dict:
    """
    读取 generate_gaze 函数生成的 .pt 文件，并为每个 gaze map 提取其峰值 (x, y) 坐标。

    Args:
        pt_file_path (str): generate_gaze 函数生成的 .pt 文件的完整路径。

    Returns:
        dict: 一个字典，键为 (traj_name, curr_time)，值为 (x_coord, y_coord) 峰值坐标。
              x_coord 对应图像宽度，y_coord 对应图像高度。
    """
    if not os.path.exists(pt_file_path):
        print(f"Error: The file '{pt_file_path}' does not exist.")
        return {}

    print(f"Loading gaze maps from '{pt_file_path}'...")
    try:
        loaded_gaze_maps_individual = torch.load(pt_file_path, map_location='cpu')
        print(f"Successfully loaded {len(loaded_gaze_maps_individual)} gaze maps from '{os.path.basename(pt_file_path)}'.")
    except Exception as e:
        print(f"Error loading .pt file '{pt_file_path}': {e}")
        return {}

    extracted_fixations = {}
    print(f"Extracting peak (x, y) coordinates for each gaze map from '{os.path.basename(pt_file_path)}'...")

    for key, gaze_map_tensor in loaded_gaze_maps_individual.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            print(f"Warning: Key '{key}' has unexpected format in '{os.path.basename(pt_file_path)}'. Skipping.")
            continue

        gaze_map_2d = gaze_map_tensor.squeeze()

        if gaze_map_2d.dim() != 2:
            print(f"Warning: Gaze map for key {key} in '{os.path.basename(pt_file_path)}' has unexpected dimensions {gaze_map_2d.shape}. Skipping peak extraction.")
            extracted_fixations[key] = (0, 0)
            continue

        max_idx_flat = torch.argmax(gaze_map_2d.view(-1)).item()

        H_map, W_map = gaze_map_2d.shape
        y_coord = max_idx_flat // W_map
        x_coord = max_idx_flat % W_map

        extracted_fixations[key] = (x_coord, y_coord)

    print(f"Finished extracting peak fixations from '{os.path.basename(pt_file_path)}'.")
    return extracted_fixations


def get_pixel_coords_from_flat_index(flat_index, image_width):
    """Convert a flattened mask index into (row, col) pixel coordinates."""
    y = flat_index // image_width
    x = flat_index % image_width
    return y, x

def extract_and_save_all_selected_ids(root_folder: str,
                                        peak_fixations_data: dict,
                                        output_pt_file_path: str):
    """Match gaze peaks to pedestrian masks and save selected IDs by frame."""
    print(f"Starting ID selection with sliding window in: {root_folder}\n")

    all_selected_ids = {}

    output_dir = os.path.dirname(output_pt_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Results for this run will be saved to: {output_pt_file_path}")

    # --- 提取当前 peak_fixations_data 中包含的所有轨迹名称 ---
    relevant_traj_names_in_gaze_data = set()
    for key in peak_fixations_data.keys():
        if isinstance(key, tuple) and len(key) == 2:
            relevant_traj_names_in_gaze_data.add(key[0])

    if not relevant_traj_names_in_gaze_data:
        print(f"  Info: No relevant trajectory names found in provided peak_fixations_data for this split. No IDs will be extracted.")
        # 即使没有相关轨迹，也要尝试保存一个空的 .pt 文件，以避免后续流程找不到文件
        try:
            torch.save({}, output_pt_file_path)
            print(f"  Saved empty .pt file to {output_pt_file_path}.")
        except Exception as e:
            print(f"  Error saving empty .pt file to {output_pt_file_path}: {e}")
        return # 直接返回，不遍历文件系统

    # 遍历根目录下的所有子文件夹
    for subdir, dirs, files in os.walk(root_folder):
        # 确保只处理包含 CSV 文件的实际轨迹子文件夹
        if not any(f.endswith('.csv') for f in files):
            continue

        traj_name = os.path.basename(subdir.rstrip(os.sep))

        # --- 只处理在当前 gaze.pt 中有数据的轨迹 ---
        if traj_name not in relevant_traj_names_in_gaze_data:
            continue # 跳过当前文件夹，处理下一个

        print(f"\n--- Processing folder: {subdir} ({traj_name}) ---")

        numbered_csv_files = []
        for f in files:
            match = re.fullmatch(r'(\d+)\.csv', f)
            if match:
                file_number = int(match.group(1))
                numbered_csv_files.append((file_number, f))
        numbered_csv_files.sort()

        if not numbered_csv_files:
            print(f"  No numbered CSV files (e.g., 0.csv, 1.csv) found in {subdir}. Skipping ID selection for this folder.")
            continue

        neighbor_offsets = [(dx, dy) for dx in range(-7, 8) for dy in range(-7, 8)]

        previous_frames_selected_ids_window = deque(maxlen=SLIDING_WINDOW_SIZE - 1)

        for frame_idx, (file_num, filename) in enumerate(numbered_csv_files):
            current_frame_key = (traj_name, frame_idx)
            csv_file_path = os.path.join(subdir, filename)

            # --- 新增：处理 CSV 读取错误，特别是 EmptyDataError，使其行为与“没有人”一致 ---
            mask_df = None # 初始化 mask_df
            try:
                mask_df = pd.read_csv(csv_file_path, header=None)
            except pd.errors.EmptyDataError:
                # 如果文件为空，用户希望将其视为“没有人的 mask”，并继续正常的滑动窗口逻辑。
                # 因此，我们创建一个空的 DataFrame 来模拟这种情况。
                mask_df = pd.DataFrame()
                print(f"  Info: File '{filename}' is empty or malformed. Treating as no person IDs found for this frame.")
            except FileNotFoundError:
                print(f"  Error: {filename} not found. Skipping this frame.")
                # 文件未找到属于严重错误，直接跳过此帧，不参与滑动窗口和最终保存。
                all_selected_ids[current_frame_key] = []
                previous_frames_selected_ids_window.append(set())
                continue
            except Exception as e:
                print(f"  An unexpected error occurred while reading {filename}: {e}. Skipping this frame.")
                # 其他读取错误，也直接跳过此帧。
                all_selected_ids[current_frame_key] = []
                previous_frames_selected_ids_window.append(set())
                continue

            # 如果 gaze.pt 没有此帧的 fixation 数据，跳过保存，但仍维护窗口
            if current_frame_key not in peak_fixations_data:
                previous_frames_selected_ids_window.append(set())
                continue

            fix_x, fix_y = peak_fixations_data[current_frame_key]
            current_frame_pixel_selected_ids = set()

            # 如果 fixation 坐标是 (0,0)，也视为没有有效注视点，并继续处理窗口
            if fix_x == 0 and fix_y == 0:
                # all_selected_ids[current_frame_key] = [] # 不再直接设置
                # previous_frames_selected_ids_window.append(set()) # 也不再直接设置，让下面统一处理
                current_frame_pixel_selected_ids = set() # 确保是空集
                # continue # 不再直接跳过，让下面统一添加到窗口
            # 如果 mask_df 为空 (无论是原始为空还是由 EmptyDataError 模拟的)，则此帧不贡献新的 ID
            elif mask_df.empty:
                current_frame_pixel_selected_ids = set()
            else:
                person_ids = mask_df.iloc[0, :].tolist()
                masks_data = mask_df.iloc[1:, :].values

                coords_to_check = []
                for dy, dx in neighbor_offsets:
                    check_y, check_x = fix_y + dy, fix_x + dx
                    if 0 <= check_y < IMAGE_HEIGHT and 0 <= check_x < IMAGE_WIDTH:
                        coords_to_check.append((check_y, check_x))

                flat_indices_to_check = [y * IMAGE_WIDTH + x for y, x in coords_to_check]

                for col_idx, person_id in enumerate(person_ids):
                    if col_idx < masks_data.shape[1]:
                        person_mask_column = masks_data[:, col_idx]
                        for flat_idx in flat_indices_to_check:
                            if 0 <= flat_idx < len(person_mask_column):
                                if person_mask_column[flat_idx] == 1:
                                    current_frame_pixel_selected_ids.add(person_id)
                                    break

            # --- 统一将当前帧（无论是否贡献 ID）添加到滑动窗口并计算最终 ID ---
            previous_frames_selected_ids_window.append(current_frame_pixel_selected_ids)

            final_selected_ids_for_frame = set(current_frame_pixel_selected_ids)
            for past_ids_set in previous_frames_selected_ids_window:
                final_selected_ids_for_frame.update(past_ids_set)

            all_selected_ids[current_frame_key] = sorted(list(final_selected_ids_for_frame))

    if all_selected_ids:
        try:
            torch.save(all_selected_ids, output_pt_file_path)
            print(f"\nSuccessfully saved all selected IDs to: {output_pt_file_path}")

            print("\nContent of the final .pt file (first 5 entries):")
            count = 0
            sorted_all_selected_ids_list = sorted(all_selected_ids.items(), key=lambda item: (item[0][0], item[0][1]))
            for key, ids in sorted_all_selected_ids_list:
                print(f"  Key: {key}, Selected IDs: {ids}")
                count += 1
                if count >= 5:
                    break
            print(f"Total entries in the final .pt file: {len(all_selected_ids)}")

        except Exception as e:
            print(f"Error saving the final .pt file to {output_pt_file_path}: {e}")
    else:
        try:
            torch.save({}, output_pt_file_path)
            print(f"No selected IDs were generated. Saved empty .pt file to {output_pt_file_path}.")
        except Exception as e:
            print(f"No selected IDs were generated and empty .pt save failed for {output_pt_file_path}: {e}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2phase/2phaseplus salient-person IDs from gaze maps.")
    parser.add_argument("--data_root", default="data", help="Root data directory containing trajectory folders.")
    parser.add_argument("--train_gaze", default="data_splits/train/gazeplus.pt", help="Input train gaze-map .pt file.")
    parser.add_argument("--train_output", default="data_splits/train/2phaseplus.pt", help="Output train salient-ID .pt file.")
    parser.add_argument("--test_gaze", default="data_splits/test/gazeplus.pt", help="Input test gaze-map .pt file.")
    parser.add_argument("--test_output", default="data_splits/test/2phaseplus.pt", help="Output test salient-ID .pt file.")
    args = parser.parse_args()

    root_directory = args.data_root
    split_paths = [
        (args.test_gaze, args.test_output),
        (args.train_gaze, args.train_output),
    ]

    print("Starting processing for each data split...")

    # 获取所有期望的轨迹名称 (从文件系统，用于最终的全局验证报告)
    expected_traj_names_global = set()
    print(f"Scanning '{root_directory}' for all expected trajectory names (for overall verification)...")
    for subdir_path in os.listdir(root_directory):
        full_subdir_path = os.path.join(root_directory, subdir_path)
        if os.path.isdir(full_subdir_path):
            if any(f.endswith('.csv') for f in os.listdir(full_subdir_path)):
                expected_traj_names_global.add(subdir_path)
    print(f"Found {len(expected_traj_names_global)} total expected trajectory folders with CSVs in '{root_directory}'.")


    all_extracted_traj_names_across_splits = set() # 用于汇总所有 split 的提取轨迹，供全局报告使用
    all_extracted_frame_counts_across_splits = Counter() # 用于汇总所有 split 的帧数，供全局报告使用

    for gaze_input_path, phase_output_path in split_paths:
        split_name = os.path.basename(os.path.dirname(gaze_input_path)) # 'test' or 'train'
        print(f"\n--- Processing split: {split_name} ---")

        # Step 1: Extract peak fixations for the current split
        current_split_gaze_peak_fixations = extract_peak_fixations_from_gaze_maps(gaze_input_path)

        if not current_split_gaze_peak_fixations:
            print(f"Failed to extract peak fixations for {split_name} split. Skipping select_ids generation for this split.")
            # 即使没有提取到，也要尝试为该 split 保存一个空的 2phase.pt 文件，避免下游程序报错
            try:
                os.makedirs(os.path.dirname(phase_output_path), exist_ok=True)
                torch.save({}, phase_output_path)
                print(f"  Saved empty .pt file to {phase_output_path}.")
            except Exception as e:
                print(f"  Error saving empty .pt file to {phase_output_path}: {e}")
            continue # 跳过当前 split 的后续处理

        # 更新全局统计数据，用于最终的整体报告
        for key in current_split_gaze_peak_fixations.keys():
            if isinstance(key, tuple) and len(key) == 2:
                traj_name, frame_idx = key
                all_extracted_traj_names_across_splits.add(traj_name)
                all_extracted_frame_counts_across_splits[traj_name] += 1

        print(f"\nStep 2: Starting select_ids extraction process for {split_name} split ({len(current_split_gaze_peak_fixations)} peak fixations).")
        if not os.path.isdir(root_directory):
            print(f"Error: The specified root directory '{root_directory}' does not exist.")
            print("Please pass the correct data root with --data_root.")
        else:
            # 将当前 split 的注视点数据和对应的输出路径传递给函数
            extract_and_save_all_selected_ids(root_directory, current_split_gaze_peak_fixations, phase_output_path)
            print(f"Finished select_ids extraction and saving for {split_name} split.")

    # --- 最终的全局验证报告 ---
    print("\n\n--- Overall Verification Report Across All Processed Gaze.pt Files ---")

    missing_in_gaze_pt_overall = expected_traj_names_global - all_extracted_traj_names_across_splits
    extra_in_gaze_pt_overall = all_extracted_traj_names_across_splits - expected_traj_names_global

    if not missing_in_gaze_pt_overall and not extra_in_gaze_pt_overall:
        print("All expected trajectories found across all gaze.pt files, and no extra trajectories.")
    else:
        if missing_in_gaze_pt_overall:
            print(f"\n！！！ 警告：以下轨迹文件夹在 '{root_directory}' 中存在，但在所有 gaze.pt 文件（test/train）的合并提取结果中仍然完全缺失数据：")
            for traj in sorted(list(missing_in_gaze_pt_overall)):
                print(f"  - {traj}")

        if extra_in_gaze_pt_overall:
            print(f"\n！！！ 警告：以下轨迹在 gaze.pt 文件中存在，但在 '{root_directory}' 中没有找到对应的文件夹：")
            for traj in sorted(list(extra_in_gaze_pt_overall)):
                print(f"  - {traj} (提取到 {all_extracted_frame_counts_across_splits.get(traj, 0)} 帧)")

    print("\n--- 提取到的所有轨迹及帧数概览 (Top 10) ---")
    for traj, count in all_extracted_frame_counts_across_splits.most_common(10):
        print(f"  - '{traj}': 提取到 {count} 帧")
    if len(all_extracted_frame_counts_across_splits) > 10:
         print(f"  ... (共 {len(all_extracted_frame_counts_across_splits)} 个轨迹)")

    print("\n--- End Overall Verification Report ---")
    print("\nAll processing complete for specified data splits.")
