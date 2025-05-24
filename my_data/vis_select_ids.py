import pickle
import sys
import os
import numpy as np
from PIL import Image
import pandas as pd

# --- Configuration ---
# IMPORTANT: Set your image dimensions here for correct pixel mapping.
# If a 160x120 image was flattened, IMAGE_WIDTH should be 160, IMAGE_HEIGHT should be 120.
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
# --- End Configuration ---

# Importance safety warning: Do not load pickle files from unknown or untrusted sources!
# This could lead to arbitrary code execution.
# Also, ensure you trust the source of your CSV and image files.

def apply_mask_overlay(image_path, mask_array, output_path, overlay_color=(0, 255, 0)):
    """
    Applies a binary mask overlay to a JPG image and saves it.

    Args:
        image_path (str): Path to the original JPG image.
        mask_array (np.ndarray): 2D binary mask array (0 or 1) with shape (height, width).
        output_path (str): Path to save the masked image.
        overlay_color (tuple): RGB color tuple for the overlay (e.g., (255, 0, 0) for red).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is in RGB format
        img_width, img_height = img.size

        # Ensure mask dimensions match image dimensions
        if mask_array.shape[0] != img_height or mask_array.shape[1] != img_width:
            print(f"  WARNING: Image {os.path.basename(image_path)} ({img_width}x{img_height}) and mask ({mask_array.shape[1]}x{mask_array.shape[0]}) dimensions do not match. Skipping overlay.")
            return False

        # Create a color overlay layer of the same size as the image
        color_overlay = Image.new('RGB', img.size, color=overlay_color)

        # Convert the binary mask (0/1) to a PIL-compatible L-mode (grayscale) mask (0/255)
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')

        # Use the mask to composite the color overlay onto the original image
        # Image.composite(image1, image2, mask) => uses image1 where mask is non-zero, image2 where mask is zero
        masked_img = Image.composite(color_overlay, img, mask_img)

        masked_img.save(output_path)
        return True # Successfully saved

    except FileNotFoundError:
        # File not found errors are handled by the calling function.
        return False
    except Exception as e:
        print(f"  ERROR applying mask overlay for {image_path}: {e}")
        return False


def process_pickle_and_apply_masks(filepath, data_folder, image_width, image_height):
    """
    Loads a pickle file (assumed to be a list of lists), processes its data structure,
    finds corresponding CSVs and JPGs, extracts matching masks, and overlays them
    onto the JPG images, overwriting the originals.
    If no matching mask is found or an error occurs, the original image is retained.

    Args:
        filepath (str): Path to the select_ids.pkl file.
        data_folder (str): Path to the folder containing corresponding CSV and JPG files.
        image_width (int): Width of the images (in pixels).
        image_height (int): Height of the images (in pixels).
    """
    print(f"Attempting to load pickle file: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

            print("\n  Successfully loaded pickle data.")

            # --- Corrected processing logic: Assume data is directly a list of lists ---
            if isinstance(data, list):
                print(f"  Detected top-level is a list. Processing its elements as rows of selected IDs.")

                processed_rows = [] # Stores each processed row (deduplicated list)

                try:
                    # 'data' is already the list of lists we want to iterate over
                    for row_index, row_list in enumerate(data):
                        # The 'row_list' from select_ids.pkl is already the list of person IDs
                        # It should already be unique and sorted as per the generation script,
                        # but we can deduplicate again defensively if needed.
                        unique_row_list = list(set(row_list)) # Deduplicate just in case
                        
                        processed_rows.append(unique_row_list)

                    print(f"  Processing complete. {len(processed_rows)} rows of data to process.")

                except Exception as e:
                    # If an error occurs during row processing (e.g., unexpected element type)
                    print(f"  ERROR: An error occurred while processing rows from the pickle list: {e}")
                    print("  Cannot proceed with file processing for this pickle file.")
                    return # Stop processing for this pickle file

                # --- File search, mask extraction, and image overlay/copy ---
                print(f"\n  Starting file search and image processing in folder '{data_folder}'...")

                masked_count = 0
                skipped_count = 0 # Count items skipped due to missing files, CSV errors, or image processing errors

                # Iterate through each row of processed data and its index
                for index, unique_row_list in enumerate(processed_rows):
                    csv_filename = f'{index}.csv'
                    jpg_filename = f'{index}.jpg'
                    csv_path = os.path.join(data_folder, csv_filename)
                    jpg_path = os.path.join(data_folder, jpg_filename)
                    # Output path is the same as the original JPG path for overwriting
                    output_path = jpg_path

                    print(f"\n  Processing index {index}: (corresponding files {csv_filename}, {jpg_filename})")

                    # Check if corresponding CSV and JPG files exist
                    csv_exists = os.path.exists(csv_path)
                    jpg_exists = os.path.exists(jpg_path)

                    if not csv_exists or not jpg_exists:
                        print(f"  SKIPPING: Missing file ({'CSV' if not csv_exists else ''}{', ' if not csv_exists and not jpg_exists else ''}{'JPG' if not jpg_exists else ''}) in '{data_folder}'.")
                        skipped_count += 1
                        continue # Skip to the next index

                    # Read CSV file and extract mask data
                    masks_for_this_image = [] # Stores all mask arrays to be overlaid for the current image

                    try:
                        # Use pandas to read CSV, assuming first row is person IDs, subsequent rows are flattened mask data
                        # We need to read the first row as header to get person IDs, then access data
                        # To correctly match the previous script's behavior where 0.csv had IDs on first row and data below
                        # we'll read with no header, then extract row 0 as IDs, and row 1 onwards as data.
                        df_raw = pd.read_csv(csv_path, header=None)

                        if df_raw.empty:
                            print(f"  WARNING: CSV file '{csv_path}' is empty. No masks to apply.")
                            # masks_for_this_image remains empty, leading to retention of original image
                        else:
                            # Person IDs are in the first row (index 0)
                            csv_person_ids = df_raw.iloc[0, :].tolist()
                            # Mask data starts from the second row (index 1)
                            masks_data_flat = df_raw.iloc[1:, :].values

                            # Iterate through the person IDs extracted from the CSV's first row
                            for col_idx, csv_id in enumerate(csv_person_ids):
                                # Check if this CSV's person ID is in the current frame's unique_row_list from pickle
                                if csv_id in unique_row_list:
                                    print(f"    CSV Person ID '{csv_id}' matched a selected ID. Attempting to extract mask data...")

                                    if col_idx < masks_data_flat.shape[1]:
                                        col_data = masks_data_flat[:, col_idx] # This is the flattened mask for this person

                                        # Check if data length matches the flattened image size
                                        expected_length = image_width * image_height
                                        if len(col_data) != expected_length:
                                            print(f"    WARNING: Mask data for ID '{csv_id}' has length ({len(col_data)}) that doesn't match expected size ({expected_length}). Skipping this mask.")
                                            continue

                                        try:
                                            # Convert data to numpy array, ensure it's uint8 (0 or 1)
                                            mask_flat = np.array(col_data).astype(np.uint8)

                                            # Check if array contains only 0s and 1s
                                            if not np.all(np.isin(mask_flat, [0, 1])):
                                                print(f"    WARNING: Mask data for ID '{csv_id}' contains non-0/1 values. Skipping this mask.")
                                                continue

                                            # Reshape the flattened mask to 2D image dimensions
                                            mask_2d = mask_flat.reshape((image_height, image_width))

                                            masks_for_this_image.append(mask_2d)
                                            print(f"    Successfully extracted and processed mask for ID '{csv_id}'.")

                                        except (ValueError, Exception) as mask_err:
                                            print(f"    ERROR: An error occurred while processing mask data for ID '{csv_id}': {mask_err}")
                                            continue
                                    else:
                                        print(f"    WARNING: Column index {col_idx} for ID '{csv_id}' is out of bounds for mask data. Skipping.")

                    except (pd.errors.EmptyDataError, FileNotFoundError, Exception) as csv_err:
                        print(f"  ERROR: An error occurred while reading or processing CSV file '{csv_path}': {csv_err}.")
                        print(f"  Due to CSV error, cannot apply mask. Original image '{jpg_filename}' will be retained.")
                        masks_for_this_image = [] # Ensure masks_for_this_image is empty, forcing original image retention
                        # Do not continue here, let the code proceed to the decision point below.


                    # --- Decision Point: Apply mask overlay or retain original image ---
                    if masks_for_this_image:
                        print(f"  Found {len(masks_for_this_image)} masks. Merging and applying overlay...")
                        try:
                            # Combine all masks (logical OR operation)
                            combined_mask_array = np.logical_or.reduce(masks_for_this_image)
                            # Apply the combined mask to the corresponding JPG image, overwriting the original
                            success = apply_mask_overlay(jpg_path, combined_mask_array, output_path)
                            if success:
                                 masked_count += 1
                                 print(f"  Successfully applied mask and OVERWROTE {os.path.basename(output_path)}")
                            else:
                                 # apply_mask_overlay will print specific errors
                                 print(f"  Failed to apply mask overlay to '{jpg_path}'. Original image retained.")
                                 skipped_count += 1

                        except Exception as overlay_err:
                            print(f"  ERROR: An error occurred while merging or applying mask overlay to '{jpg_path}' when processing {filepath}: {overlay_err}.")
                            skipped_count += 1 # Count as skipped due to overlay error

                    else:
                        # No matching masks found (includes cases where CSV error led to empty masks_for_this_image)
                        print(f"  No matching column headers found for masks. Original image '{jpg_filename}' will be retained.")
                        # No operation needed here, as the original image is simply not overwritten.
                        # We increment skipped_count because it indicates a frame where masks were not applied as desired.
                        skipped_count += 1


                print(f"\n  --- File Processing Summary for {os.path.basename(filepath)} ---")
                print(f"  Total frames processed based on pickle data: {len(processed_rows)}")
                print(f"  Successfully applied mask and OVERWROTE images: {masked_count}")
                print(f"  Original images retained (no matching masks or errors): {skipped_count}")
                print("  ------------------------------------------------")

            # --- Fallback logic: If not the expected specific structure ---
            else:
                print(f"  Loaded data from {filepath} is not the expected list of lists. Cannot proceed with file processing for this pickle file.")


    except FileNotFoundError:
        print(f"ERROR: Pickle file '{filepath}' not found. Please ensure the file path is correct.")
    except pickle.UnpicklingError:
        print(f"ERROR: Could not load data from '{filepath}'. It might not be a valid pickle file, or the file is corrupted.")
    except Exception as e:
        print(f"ERROR: An unknown error occurred while loading or processing pickle file '{filepath}': {e}")


if __name__ == "__main__":
    # === CONFIGURE YOUR ROOT DIRECTORY HERE ===
    # This should be the main folder containing all your subfolders (e.g., 't/subfolder1', 't/subfolder2')
    root_directory = '/home/yzc/CRUEL/data_vis' # <<< DOUBLE-CHECK AND CHANGE THIS PATH!

    print("--- ⚠️⚠️⚠️ WARNING: THIS SCRIPT WILL OVERWRITE YOUR ORIGINAL JPG FILES. ⚠️⚠️⚠️ ---")
    print("--- PLEASE BACK UP YOUR DATA BEFORE PROCEEDING. ---")
    input("Press Enter to continue, or Ctrl+C to abort.")

    if not os.path.isdir(root_directory):
        print(f"ERROR: The specified root directory '{root_directory}' does not exist or is not a valid directory. Please check the path.")
        sys.exit(1)

    found_pickle_files = []
    # Walk through all subdirectories to find select_ids.pkl files
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if 'select_ids.pkl' in filenames:
            pickle_path = os.path.join(dirpath, 'select_ids.pkl')
            found_pickle_files.append((pickle_path, dirpath)) # Store (pickle_path, data_folder)

    if not found_pickle_files:
        print(f"No 'select_ids.pkl' files found in '{root_directory}' or its subdirectories. No masks will be applied.")
        sys.exit(0)

    print(f"\nFound {len(found_pickle_files)} 'select_ids.pkl' files. Starting processing...")
    for pkl_path, data_folder_for_pkl in found_pickle_files:
        print(f"\nProcessing data for: {data_folder_for_pkl}")
        process_pickle_and_apply_masks(
            pkl_path,
            data_folder_for_pkl,
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
        print("\n--- Processing complete for this subfolder. ---")

    print("\nAll mask application and overwriting processes complete.")
    print("\n--- ⚠️⚠️⚠️ REMINDER: YOUR ORIGINAL IMAGES MAY HAVE BEEN OVERWRITTEN. ⚠️⚠️⚠️ ---")