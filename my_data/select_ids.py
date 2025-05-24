import os
import pandas as pd
import pickle
import re
import numpy as np
from collections import deque

# --- Configuration ---
# IMPORTANT: You MUST set your image dimensions here for correct pixel mapping.
# These values determine how flattened mask data (1D column) maps back to 2D (y, x) coordinates.
# If a 160x120 image was flattened, IMAGE_WIDTH should be 160, IMAGE_HEIGHT should be 120.
IMAGE_WIDTH = 160  # Corrected example from your input
IMAGE_HEIGHT = 120 # Corrected example from your input

# Sliding window configuration
# SLIDING_WINDOW_SIZE = 10 means: current frame + past 9 frames = 10 frames total.
SLIDING_WINDOW_SIZE = 15
# --- End Configuration ---

def get_pixel_coords_from_flat_index(flat_index, image_width):
    """Converts a flattened 1D pixel index to (y, x) (row, col) coordinates."""
    y = flat_index // image_width
    x = flat_index % image_width
    return y, x

def extract_selected_ids_based_on_fixations(root_folder):
    """
    Traverses subfolders to:
    1. Read fixation points from 'fixations.csv' (assuming (X, Y) per row, no header).
    2. For each fixation, calculate a neighborhood of pixels.
    3. IMPORTANT: If fixation coordinates are (0,0), it will skip mask matching for that frame.
    4. Identify which person_ids (from corresponding numbered CSVs' first row)
       have a mask pixel (value 1) within that neighborhood.
       (Numbered CSVs assumed: no header, first row is IDs, subsequent rows are flattened mask data.)
    5. Incorporate unique IDs from a SLIDING_WINDOW_SIZE number of previous frames.
    6. Compile these selected IDs into a list of lists, saved as 'select_ids.pkl'.
    7. Prints debugging information for fixation coordinates and the 1D flattened indices (CSV row numbers).
    """
    print(f"Starting ID selection with sliding window in: {root_folder}\n")

    for subdir, dirs, files in os.walk(root_folder):
        print(f"\n--- Processing folder: {subdir} ---")

        fixations_file_path = os.path.join(subdir, 'fixations.csv')
        select_ids_for_subdir = [] # This will be the list of lists for select_ids.pkl

        # --- 1. Load Fixation Data ---
        if not os.path.exists(fixations_file_path):
            print(f"  'fixations.csv' not found in {subdir}. Skipping this folder.")
            continue

        try:
            fixations_df = pd.read_csv(fixations_file_path, header=None)
            fixation_points = fixations_df.values # Convert to NumPy array for efficient access
            print(f"  Loaded {len(fixation_points)} fixation points from 'fixations.csv'.")

        except pd.errors.EmptyDataError:
            print(f"  'fixations.csv' in {subdir} is empty. Skipping this folder.")
            continue
        except Exception as e:
            print(f"  Error loading 'fixations.csv' in {subdir}: {e}. Skipping.")
            continue

        # --- 2. Get and Sort Numbered CSV Files ---
        numbered_csv_files = []
        for f in files:
            # Match files like '0.csv', '1.csv', '10.csv'
            match = re.fullmatch(r'(\d+)\.csv', f)
            if match:
                file_number = int(match.group(1))
                numbered_csv_files.append((file_number, f))
        numbered_csv_files.sort() # Sort by number to ensure correct frame order

        if not numbered_csv_files:
            print(f"  No numbered CSV files (e.g., 0.csv, 1.csv) found in {subdir}. Skipping ID selection.")
            continue

        # --- 3. Define Neighborhood Offsets ---
        # 11x11 square around fixation point (from -5 to +5 in both x and y)
        # Note: You changed this to -7 to 8 range, keeping that for now.
        neighbor_offsets = [(dx, dy) for dx in range(-7, 8) for dy in range(-7, 8)]

        # --- 4. Initialize Sliding Window ---
        # deque stores sets of IDs from previous frames, maxlen is SLIDING_WINDOW_SIZE - 1
        previous_frames_selected_ids_window = deque(maxlen=SLIDING_WINDOW_SIZE - 1)

        # --- 5. Iterate Through Each Frame (CSV File) ---
        for frame_idx, (file_num, filename) in enumerate(numbered_csv_files):
            # Ensure a corresponding fixation point exists for this frame
            if frame_idx >= len(fixation_points):
                print(f"  Warning: No corresponding fixation point for {filename} (frame {frame_idx}). Appending empty IDs.")
                previous_frames_selected_ids_window.append(set()) # Add empty to window to maintain length
                select_ids_for_subdir.append([])
                continue

            csv_file_path = os.path.join(subdir, filename)
            
            # This set will hold IDs selected based purely on pixel overlap in the current frame
            current_frame_pixel_selected_ids = set()

            try:
                # Get the fixation point for the current frame.
                # Coordinates are (X, Y) as per your clarification
                fix_x, fix_y = int(fixation_points[frame_idx, 0]), int(fixation_points[frame_idx, 1])

                # --- Debugging Prints for Fixation and 1D Flattened Indices ---
                print(f"\n    --- Frame {frame_idx} ({filename}) ---")
                print(f"    Fixation coordinates (X, Y): ({fix_x}, {fix_y})") # Print (X, Y) for clarity

                # --- New Logic: Skip mask matching if fixation is (0,0) ---
                if fix_x == 0 and fix_y == 0:
                    print(f"    Fixation is (0,0). Skipping mask matching for this frame.")
                    previous_frames_selected_ids_window.append(set()) # Add empty to window
                    select_ids_for_subdir.append([])
                    continue # Skip to the next frame

                # Read the numbered CSV file (mask data) without a header
                mask_df = pd.read_csv(csv_file_path, header=None)

                if mask_df.empty:
                    print(f"    {filename} is empty. Skipping mask processing for this frame.")
                    previous_frames_selected_ids_window.append(set()) # Add empty to window
                    select_ids_for_subdir.append([])
                    continue

                # Extract person IDs from the very first row (header row)
                person_ids = mask_df.iloc[0, :].tolist()
                
                # Extract mask data: all rows from the second row onwards, all columns.
                # This `masks_data` contains the flattened mask for each person.
                masks_data = mask_df.iloc[1:, :].values
                
                # Calculate coordinates to check around the fixation point
                coords_to_check = [] # This will store (y, x) coordinates for pixels
                for dy, dx in neighbor_offsets:
                    check_y, check_x = fix_y + dy, fix_x + dx # Use fix_y for Y-axis, fix_x for X-axis
                    # Ensure coordinates are within image bounds before adding
                    if 0 <= check_y < IMAGE_HEIGHT and 0 <= check_x < IMAGE_WIDTH:
                        coords_to_check.append((check_y, check_x)) # Store as (y, x)

                # Convert 2D (y, x) coords to flattened 1D indices for mask_data lookup.
                # This is the list of CSV row numbers (after header) that will be checked.
                # Formula remains y * IMAGE_WIDTH + x for row-major flattening
                flat_indices_to_check = [y * IMAGE_WIDTH + x for y, x in coords_to_check]

                print(f"    Number of 1D flattened indices to check: {len(flat_indices_to_check)}")
                # For brevity, print a sample of the indices or the full list if small
                if len(flat_indices_to_check) <= 20: # Adjust this threshold as needed
                    print(f"    1D Flattened Indices (CSV row numbers): {flat_indices_to_check}")
                else:
                    print(f"    1D Flattened Indices (CSV row numbers - first 10, last 10): {flat_indices_to_check[:10]} ... {flat_indices_to_check[-10:]}")

                if not flat_indices_to_check:
                    print(f"    Neighborhood is empty (fixation coordinates out of image bounds or invalid).")
                # --- End Debugging Prints ---

                # Iterate through each person_id and its corresponding mask column
                for col_idx, person_id in enumerate(person_ids):
                    # Ensure the column index exists in the masks_data
                    if col_idx < masks_data.shape[1]:
                        person_mask_column = masks_data[:, col_idx] # Get the flattened mask for this person

                        # Check if any pixel in the neighborhood has a mask value of 1 for this person
                        for flat_idx in flat_indices_to_check:
                            # Ensure the flat_idx is within the bounds of the person's mask column length
                            if 0 <= flat_idx < len(person_mask_column):
                                if person_mask_column[flat_idx] == 1: # If the mask pixel is 1
                                    current_frame_pixel_selected_ids.add(person_id) # Add this person's ID
                                    break # Move to the next person_id, no need to check more pixels for this one

                # --- 6. Apply Sliding Window Logic ---
                # Add the IDs selected in the current frame (purely pixel-based) to the window
                previous_frames_selected_ids_window.append(current_frame_pixel_selected_ids)

                # Combine current frame's IDs with all unique IDs from the sliding window
                final_selected_ids_for_frame = set(current_frame_pixel_selected_ids)
                for past_ids_set in previous_frames_selected_ids_window:
                    final_selected_ids_for_frame.update(past_ids_set)
                
                # Convert to sorted list and append to the main results list for this subfolder
                select_ids_for_subdir.append(sorted(list(final_selected_ids_for_frame)))
                
                print(f"    Final IDs (with window) for Frame {frame_idx}: {sorted(list(final_selected_ids_for_frame))}")

            except pd.errors.EmptyDataError:
                print(f"    Skipping {filename}: File is empty or malformed.")
                previous_frames_selected_ids_window.append(set()) # Add empty to window to maintain length
                select_ids_for_subdir.append([]) # Append empty list for skipped frame
            except FileNotFoundError:
                print(f"    Error: {filename} not found.")
                previous_frames_selected_ids_window.append(set()) # Add empty to window
                select_ids_for_subdir.append([])
            except Exception as e:
                print(f"    An unexpected error occurred while processing {filename}: {e}")
                previous_frames_selected_ids_window.append(set()) # Add empty to window
                select_ids_for_subdir.append([])

        # --- 7. Save Results as .pkl ---
        if select_ids_for_subdir:
            output_pkl_path = os.path.join(subdir, 'select_ids.pkl')
            try:
                with open(output_pkl_path, 'wb') as f:
                    pickle.dump(select_ids_for_subdir, f)
                print(f"\n  Successfully saved selected IDs to: {output_pkl_path}")

                # Print the content for verification in the desired list-of-lists format
                print("\n  Content of select_ids.pkl:")
                with open(output_pkl_path, 'rb') as f:
                    loaded_selected_ids = pickle.load(f)
                    print(loaded_selected_ids) # Print the raw list of lists
                    print(f"  Total frames processed: {len(loaded_selected_ids)}")

            except Exception as e:
                print(f"  Error saving or loading {output_pkl_path}: {e}")
        else:
            print(f"  No selected IDs for {subdir}. 'select_ids.pkl' not created.")

# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Set your root directory here! This is the main folder containing your subfolders.
    # Example for Windows: root_directory = 'C:\\Users\\YourUser\\MyProjectData'
    # Example for macOS/Linux: root_directory = '/Users/YourUser/Documents/ExperimentData'
    root_directory = '/home/yzc/CRUEL/data' # <<< DOUBLE-CHECK AND CHANGE THIS PATH if needed!

    if not os.path.isdir(root_directory):
        print(f"Error: The specified root directory '{root_directory}' does not exist.")
        print("Please update 'root_directory' in the script to your actual folder path.")
    else:
        print(f"Starting select_ids extraction process with sliding window in: {root_directory}")
        extract_selected_ids_based_on_fixations(root_directory)
        print("\nAll select_ids extraction and saving complete.")