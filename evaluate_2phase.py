import os
import pickle
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score # Accuracy is less direct here
from typing import List, Tuple, Set

# --- Configuration Paths ---
# Your root directory containing all trajectory subfolders (each with person_ids.pkl and select_ids.pkl)
DATA_ROOT_FOLDER = '/home/yzc/CRUEL/data' 

# Path to your generated select_ids file (typically 2phase.pt)
# Adjust this path based on whether you're evaluating train or test set
GENERATED_SELECT_IDS_PATH = '/home/yzc/CRUEL/data_splits/train/2phaseplus.pt' 
# --- End Configuration ---

# --- Evaluation Display Settings ---
NUM_TRAJS_TO_PRINT_SUMMARY = 1 # Number of lowest F1-score trajectories to display in the summary
NUM_TRAJS_TO_PRINT_DETAIL = 5   # Number of lowest F1-score trajectories for detailed frame-by-frame labels
# Minimum number of unique person IDs for a trajectory to be included in per-trajectory metric calculation
MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS = 5 
# --- End Evaluation Display Settings ---


class SelectIDComparer:
    def __init__(self, data_root_folder: str, generated_select_ids_path: str):
        self.data_root_folder = data_root_folder
        self.generated_select_ids_path = generated_select_ids_path
        
        # Cache for original person_ids.pkl and select_ids.pkl (Ground Truth)
        self.person_ids_cache = {}
        self.gt_select_ids_cache = {} 

        # Store per-frame data for detailed printing (for lowest F1 trajectories)
        # Structure: {traj_name: [(frame_idx, gt_labels_list, pred_labels_list, person_ids_list), ...]}
        self.per_trajectory_data_frames = defaultdict(list) 

        # Store unique collected IDs per trajectory for the new metric calculation
        # Structure: {traj_name: {'gt_ids': set(), 'pred_ids': set()}}
        self.trajectory_unique_ids = defaultdict(lambda: {'gt_ids': set(), 'pred_ids': set()})

        # Load your generated select_ids data (predictions)
        print(f"Loading generated select_ids from: {self.generated_select_ids_path}")
        try:
            self.predicted_select_ids = torch.load(self.generated_select_ids_path, map_location='cpu')
            print(f"Successfully loaded {len(self.predicted_select_ids)} predicted select_ids entries.")
        except FileNotFoundError:
            print(f"Error: Generated select_ids file not found at {self.generated_select_ids_path}")
            self.predicted_select_ids = {}
        except Exception as e:
            print(f"Error loading generated select_ids: {e}")
            self.predicted_select_ids = {}

    def _load_gt_select_ids(self, trajectory_name: str) -> List[List[int]]:
        """
        Loads GT select_ids for a given trajectory from select_ids.pkl.
        """
        if trajectory_name not in self.gt_select_ids_cache:
            file_path = os.path.join(self.data_root_folder, trajectory_name, "select_ids.pkl")
            if not os.path.exists(file_path):
                print(f"Warning: GT select_ids.pkl not found for {trajectory_name}. Returning empty list.")
                self.gt_select_ids_cache[trajectory_name] = []
                return []
            try:
                with open(file_path, "rb") as f:
                    select_ids_list = pickle.load(f)
                self.gt_select_ids_cache[trajectory_name] = select_ids_list
            except Exception as e:
                print(f"Error loading GT select_ids.pkl for {trajectory_name}: {e}. Returning empty list.")
                self.gt_select_ids_cache[trajectory_name] = []
        return self.gt_select_ids_cache[trajectory_name]

    def _load_person_ids(self, trajectory_name: str) -> List[List[int]]:
        """
        Loads person_ids for a given trajectory from person_ids.pkl.
        """
        if trajectory_name not in self.person_ids_cache:
            file_path = os.path.join(self.data_root_folder, trajectory_name, "person_ids.pkl")
            if not os.path.exists(file_path):
                print(f"Warning: person_ids.pkl not found for {trajectory_name}. Returning empty list.")
                self.person_ids_cache[trajectory_name] = []
                return []
            try:
                with open(file_path, "rb") as f:
                    person_ids_list = pickle.load(f)
                self.person_ids_cache[trajectory_name] = person_ids_list
            except Exception as e:
                print(f"Error loading person_ids.pkl for {trajectory_name}: {e}. Returning empty list.")
                self.person_ids_cache[trajectory_name] = []
        return self.person_ids_cache[trajectory_name]

    def _get_gt_and_pred_labels_for_frame(self, trajectory_name: str, curr_time: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Gets person_ids, GT labels, and Pred labels for the current frame.
        Applies filtering based on person_ids_for_frame.
        Returns: (person_ids_for_frame, gt_labels_for_frame, pred_labels_for_frame)
        """
        all_person_ids = self._load_person_ids(trajectory_name)
        all_gt_select_ids = self._load_gt_select_ids(trajectory_name)

        if curr_time >= len(all_person_ids) or curr_time >= len(all_gt_select_ids):
            return [], [], []

        person_ids_for_frame = all_person_ids[curr_time]
        gt_select_ids_for_frame_raw = all_gt_select_ids[curr_time]
        
        key = (trajectory_name, curr_time)
        predicted_select_ids_for_frame_raw = self.predicted_select_ids.get(key, [])

        # Filter out IDs not in person_ids_for_frame for both GT and Pred
        gt_select_ids_filtered = [tid for tid in gt_select_ids_for_frame_raw if tid in person_ids_for_frame]
        pred_select_ids_filtered = [tid for tid in predicted_select_ids_for_frame_raw if tid in person_ids_for_frame]

        # Convert to binary labels based on filtered IDs
        gt_labels_for_frame = [1 if tid in gt_select_ids_filtered else 0 for tid in person_ids_for_frame]
        pred_labels_for_frame = [1 if tid in pred_select_ids_filtered else 0 for tid in person_ids_for_frame]

        return person_ids_for_frame, gt_labels_for_frame, pred_labels_for_frame

    def calculate_set_metrics(self, gt_set: Set[int], pred_set: Set[int]) -> Tuple[float, float, float]:
        """
        Calculates precision, recall, and f1_score based on two sets of unique IDs.
        """
        tp = len(gt_set.intersection(pred_set))
        fn = len(gt_set - pred_set) # IDs in GT but not in Pred
        fp = len(pred_set - gt_set) # IDs in Pred but not in GT

        # Handle division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1, tp, fn, fp

    def compare_and_evaluate(self):
        evaluated_frames_count = 0
        total_unique_traj_names = len(set(key[0] for key in self.predicted_select_ids.keys()))

        print(f"\nStarting evaluation across {total_unique_traj_names} trajectories...")

        sorted_keys = sorted(self.predicted_select_ids.keys(), key=lambda x: (x[0], x[1]))

        for key_idx, (traj_name, curr_time) in enumerate(sorted_keys):
            if (key_idx + 1) % 5000 == 0: 
                print(f"  Processed {key_idx + 1}/{len(sorted_keys)} frames...")

            person_ids_for_frame, gt_labels_for_frame, pred_labels_for_frame = \
                self._get_gt_and_pred_labels_for_frame(traj_name, curr_time)
            
            # Only consider frames with actual persons for evaluation
            if not person_ids_for_frame:
                continue 

            # The filtering ensures label lengths match by design now
            # if len(gt_labels_for_frame) != len(pred_labels_for_frame):
            #     print(f"Warning: Label length mismatch for {traj_name} at time {curr_time}. GT: {len(gt_labels_for_frame)}, Pred: {len(pred_labels_for_frame)}. Skipping this frame.")
            #     continue

            evaluated_frames_count += 1

            # Store for detailed frame-by-frame printing
            self.per_trajectory_data_frames[traj_name].append((curr_time, gt_labels_for_frame, pred_labels_for_frame, person_ids_for_frame))

            # Accumulate unique GT and Pred IDs for the entire trajectory
            for i, person_id in enumerate(person_ids_for_frame):
                if gt_labels_for_frame[i] == 1:
                    self.trajectory_unique_ids[traj_name]['gt_ids'].add(person_id)
                if pred_labels_for_frame[i] == 1:
                    self.trajectory_unique_ids[traj_name]['pred_ids'].add(person_id)

        if not self.trajectory_unique_ids:
            print("\nNo unique person IDs found across any trajectory for evaluation. Please check your data paths and content.")
            return

        # --- Calculate Overall Evaluation Results (Based on Unique Person IDs per Trajectory) ---
        # Flatten all unique GT and Pred IDs from all trajectories for global metrics
        all_unique_gt_ids_overall = set()
        all_unique_pred_ids_overall = set()

        for traj_name, ids_data in self.trajectory_unique_ids.items():
            all_unique_gt_ids_overall.update(ids_data['gt_ids'])
            all_unique_pred_ids_overall.update(ids_data['pred_ids'])

        # Calculate overall metrics using the combined sets
        overall_precision, overall_recall, overall_f1, overall_tp, overall_fn, overall_fp = \
            self.calculate_set_metrics(all_unique_gt_ids_overall, all_unique_pred_ids_overall)

        print(f"\n--- Overall Evaluation Results (Unique Person IDs across ALL Trajectories) ---")
        print(f"Total Unique GT IDs: {len(all_unique_gt_ids_overall)}")
        print(f"Total Unique Pred IDs: {len(all_unique_pred_ids_overall)}")
        print(f"TP: {overall_tp}, FN: {overall_fn}, FP: {overall_fp}")
        print(f"Precision: {overall_precision:.4f}")
        print(f"Recall:    {overall_recall:.4f}")
        print(f"F1-score:  {overall_f1:.4f}")
        print(f"--------------------------")

        # --- Per-Trajectory Evaluation (Summary, Based on Unique Person IDs) ---
        print("\n--- Per-Trajectory Evaluation (Summary, Based on Unique Person IDs) ---")
        per_traj_metrics = {}

        for traj_name, ids_data in self.trajectory_unique_ids.items():
            gt_ids = ids_data['gt_ids']
            pred_ids = ids_data['pred_ids']

            # Filter out trajectories with too few unique persons for meaningful metrics
            if len(gt_ids.union(pred_ids)) < MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS:
                 continue
            
            precision, recall, f1, tp, fn, fp = self.calculate_set_metrics(gt_ids, pred_ids)
            
            per_traj_metrics[traj_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'total_gt_ids': len(gt_ids),
                'total_pred_ids': len(pred_ids),
                'total_unique_ids_in_frame': len(gt_ids.union(pred_ids)) # union of all seen GT/Pred IDs
            }
        
        # Sort by F1-score ascending
        sorted_traj_by_f1 = sorted(per_traj_metrics.items(), key=lambda item: item[1]['f1'])

        if not sorted_traj_by_f1:
            print(f"No trajectories with at least {MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS} unique persons found for per-trajectory evaluation summary.")
        else:
            print(f"Top {NUM_TRAJS_TO_PRINT_SUMMARY} trajectories with the lowest F1-score:")
            for i, (traj_name, metrics) in enumerate(sorted_traj_by_f1):
                if i >= NUM_TRAJS_TO_PRINT_SUMMARY:
                    break
                print(f"  - Trajectory: {traj_name}")
                print(f"    F1-score: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                print(f"    TP: {metrics['tp']}, FN: {metrics['fn']}, FP: {metrics['fp']}")
                print(f"    Total Unique GT IDs: {metrics['total_gt_ids']}, Total Unique Pred IDs: {metrics['total_pred_ids']}")
                print("    ---")
        
        # --- Detailed Frame-by-Frame Labels (for Top Lowest F1-score Trajectories) ---
        print(f"\n--- Detailed Frame-by-Frame Labels for Top {NUM_TRAJS_TO_PRINT_DETAIL} Lowest F1-score Trajectories ---")
        detailed_printed_count = 0
        for traj_name, metrics in sorted_traj_by_f1:
            if detailed_printed_count >= NUM_TRAJS_TO_PRINT_DETAIL:
                break
            
            # Ensure this trajectory had frame data collected and meets the unique person threshold
            if traj_name not in self.per_trajectory_data_frames or \
               len(self.trajectory_unique_ids[traj_name]['gt_ids'].union(self.trajectory_unique_ids[traj_name]['pred_ids'])) < MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS:
                continue

            print(f"\nTrajectory: {traj_name} (F1-score: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f})")
            print(f"  Overall TP: {metrics['tp']}, FN: {metrics['fn']}, FP: {metrics['fp']}")
            print(f"  Unique GT IDs for trajectory: {sorted(list(self.trajectory_unique_ids[traj_name]['gt_ids']))}")
            print(f"  Unique Pred IDs for trajectory: {sorted(list(self.trajectory_unique_ids[traj_name]['pred_ids']))}")
            print(f"  ------|------------------|------------------|------------")
            print(f"  Frame | Person IDs       | GT Labels        | Pred Labels")
            print(f"  ------|------------------|------------------|------------")
            
            # Get and sort all frame data for this trajectory
            sorted_frames_data = sorted(self.per_trajectory_data_frames[traj_name], key=lambda x: x[0])

            for frame_idx, gt_labels, pred_labels, person_ids in sorted_frames_data:
                # Pad strings for alignment
                person_ids_str = str(person_ids).ljust(16)
                gt_str = str(gt_labels).ljust(16) 
                pred_str = str(pred_labels)
                print(f"  {frame_idx:<5} | {person_ids_str} | {gt_str} | {pred_str}")
            
            detailed_printed_count += 1
            print("------------------------------------------------------------------") 

        if detailed_printed_count == 0 and len(sorted_traj_by_f1) > 0:
            print(f"No trajectories met the criteria for detailed printing (e.g., MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS not met or no frames data).")
        elif detailed_printed_count == 0 and len(sorted_traj_by_f1) == 0:
             print(f"No trajectories with calculable metrics found to display details for (check MIN_UNIQUE_PERSONS_FOR_TRAJ_METRICS).")

        print("\n--- End Evaluation Report ---")

if __name__ == "__main__":
    comparer = SelectIDComparer(DATA_ROOT_FOLDER, GENERATED_SELECT_IDS_PATH)
    comparer.compare_and_evaluate()