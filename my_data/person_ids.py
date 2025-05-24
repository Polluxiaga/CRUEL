import os
import pandas as pd
import pickle
import re

def extract_and_save_person_ids(root_folder):
    """
    Traverses subfolders, extracts ALL values from the first row of sequentially
    numbered CSV files as sublists, compiles these sublists into a larger list
    per subfolder, and saves the list as 'person_ids.pkl'.
    It then prints the content for verification.
    """
    for subdir, dirs, files in os.walk(root_folder):
        print(f"\n--- Processing folder: {subdir} ---")

        # List to store lists of IDs for the current subfolder.
        # Each element in this list will be a sublist of IDs from one CSV file's first row.
        all_person_ids_for_subdir = []

        # Filter for CSV files named with digits and sort them numerically
        numbered_csv_files = []
        for f in files:
            # Check for files like '0.csv', '1.csv', '10.csv'
            match = re.fullmatch(r'(\d+)\.csv', f)
            if match:
                file_number = int(match.group(1))
                numbered_csv_files.append((file_number, f))

        # Sort files numerically based on their name
        numbered_csv_files.sort()

        if not numbered_csv_files:
            print(f"  No numbered CSV files found in {subdir}. Skipping.")
            continue

        for file_num, filename in numbered_csv_files:
            file_path = os.path.join(subdir, filename)
            ids_from_first_row = [] # 初始化为空列表，无论成功与否，都会尝试添加到 all_person_ids_for_subdir

            try:
                # Read the CSV file. Assume no header.
                df = pd.read_csv(file_path, header=None)

                if not df.empty:
                    # Get the entire first row (index 0) and convert it to a list.
                    ids_from_first_row = df.iloc[0, :].tolist()
                    print(f"  Extracted IDs {ids_from_first_row} from {filename}")
                else:
                    print(f"  {filename} is empty. Appending an empty list.")
                    # ids_from_first_row 已经为空列表，无需再次赋值

            except pd.errors.EmptyDataError:
                print(f"  Skipping {filename}: File is empty or malformed. Appending an empty list.")
                # ids_from_first_row 已经为空列表，无需再次赋值
            except FileNotFoundError:
                print(f"  Error: {filename} not found. Appending an empty list.")
                # ids_from_first_row 已经为空列表，无需再次赋值
            except Exception as e:
                print(f"  An error occurred while processing {filename}: {e}. Appending an empty list.")
                # ids_from_first_row 已经为空列表，无需再次赋值
            finally:
                # 无论try/except结果如何，都将当前文件的IDs（可能是空列表）添加到总列表中
                all_person_ids_for_subdir.append(ids_from_first_row)


        # Save the accumulated list of ID sublists to a .pkl file in the current subfolder
        if all_person_ids_for_subdir: # 即使包含空子列表，只要总列表不为空，就保存
            output_pkl_path = os.path.join(subdir, 'person_ids.pkl')
            try:
                with open(output_pkl_path, 'wb') as f:
                    pickle.dump(all_person_ids_for_subdir, f)
                print(f"\n  Successfully saved IDs to: {output_pkl_path}")

                # Print the content for verification
                print("\n  Content of person_ids.pkl:")
                with open(output_pkl_path, 'rb') as f:
                    loaded_ids = pickle.load(f)
                    print(f"  {loaded_ids}")
                    print(f"  Total CSV files processed for IDs (including empty ones): {len(loaded_ids)}")

            except Exception as e:
                print(f"  Error saving or loading {output_pkl_path}: {e}")
        else:
            # 这种情况通常发生在 `numbered_csv_files` 为空（已经处理过）或者所有文件处理都失败了
            print(f"  No IDs extracted or all files were skipped (resulting in an empty list). 'person_ids.pkl' not created.")

# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Replace './your_root_folder' with the actual path to your target folder!
    # Example for Windows: root_directory = 'C:\\Users\\YourUser\\YourProjectData'
    # Example for macOS/Linux: root_directory = '/Users/YourUser/YourProjectData'
    root_directory = '/home/yzc/CRUEL/data' # This path is from your previous input.

    if not os.path.isdir(root_directory):
        print(f"Error: The specified root directory '{root_directory}' does not exist.")
        print("Please update 'root_directory' in the script to your actual folder path.")
    else:
        print(f"Starting ID extraction in: {root_directory}")
        extract_and_save_person_ids(root_directory)
        print("\nAll ID extraction and saving complete.")