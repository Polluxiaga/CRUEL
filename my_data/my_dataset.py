import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset

from my_data.my_data_utils import (
    img_path_to_data,
    get_data_path,
)


class gaze_dataset(Dataset):
    # Class-level flag to track if caches are built
    _caches_built = {}

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        len_traj_pred: int,
        context_size: int,
        obs_type: str = "image",
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            context_size (int): Number of previous observations to use as context
            normalize (bool): Whether to normalize the distances or actions
            obs_type (str): What data type to use for the observation
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.len_traj_pred = len_traj_pred

        self.context_size = context_size
        self.obs_type = obs_type


        self.trajectory_cache = {}
        self.fixations_cache = {}
        self.person_ids_cache = {}
        self.select_ids_cache = {}
        self._load_index()
        
        # Use cache path as unique key for tracking built status
        # We'll use two separate LMDBs for images and masks for clarity and modularity
        self._image_cache_path = os.path.join(
            self.data_split_folder,
            f"{self.dataset_name}_images.lmdb",
        )
        self._mask_cache_path = os.path.join( # New cache path for masks
            self.data_split_folder,
            f"{self.dataset_name}_masks.lmdb",
        )
        
        # Only build caches if not already built for this cache path
        if self._image_cache_path not in self._caches_built: # Check image cache build status
            self._build_caches()
            self._caches_built[self._image_cache_path] = True
        
        # Always open the LMDB environment for both
        self._open_cache()

    def _open_cache(self):
        """Open the LMDB environment(s) in read-only mode"""
        self._image_cache = lmdb.open(self._image_cache_path, readonly=True, max_readers=256, lock=False)
        self._mask_cache = lmdb.open(self._mask_cache_path, readonly=True, max_readers=256, lock=False) # Open mask cache

    def __del__(self):
        """Clean up LMDB resources"""
        if hasattr(self, '_image_cache'):
            self._image_cache.close()
        if hasattr(self, '_mask_cache'): # Close mask cache
            self._mask_cache.close()

    def __getstate__(self):
        """Handle pickling"""
        state = self.__dict__.copy()
        state["_image_cache"] = None
        state["_mask_cache"] = None # Ensure mask cache is also set to None
        return state
    
    def __setstate__(self, state):
        """Handle unpickling"""
        self.__dict__ = state
        self._open_cache()


    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data)

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size
            end_time = traj_len - self.len_traj_pred * 3
            for curr_time in range(begin_time, end_time):
                samples_index.append((traj_name, curr_time))

        return samples_index, goals_index


    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, obs_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"{self.dataset_name}_index_to_data.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.samples_index, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.samples_index, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.samples_index, self.goals_index), f)
        
        # Validate all trajectories
        for traj_name in self.traj_names:
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data)
            
            # Load and verify person_ids length
            with open(os.path.join(self.data_folder, traj_name, "person_ids.pkl"), "rb") as f:
                person_ids_list = pickle.load(f)
            if len(person_ids_list) != traj_len:
                print(f"Warning: Trajectory {traj_name} has mismatched lengths:")
                print(f"traj_data length: {traj_len}")
                print(f"person_ids length: {len(person_ids_list)}")


    def _build_caches(self, use_tqdm: bool = True):
        """扩展缓存以包含所有需要的数据"""
        # Check if both caches exist, if so, return
        if os.path.exists(self._image_cache_path) and os.path.exists(self._mask_cache_path):
            return

        tqdm_iterator = tqdm.tqdm(
            self.goals_index,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Building LMDB caches for {self.dataset_name}"
        )

        # Open both LMDB environments for writing
        with lmdb.open(self._image_cache_path, map_size=2**40) as img_cache, \
             lmdb.open(self._mask_cache_path, map_size=2**40) as mask_cache:
            with img_cache.begin(write=True) as img_txn, \
                 mask_cache.begin(write=True) as mask_txn:
                for traj_name, time in tqdm_iterator:
                    # Cache images (existing logic)
                    image_path = get_data_path(self.data_folder, traj_name, time)
                    with open(image_path, "rb") as f:
                        img_txn.put(image_path.encode(), f.read())

                    # --- New: Cache person masks ---
                    # First, load person_ids for the current frame
                    person_ids, _ = self._load_persons(traj_name, time) # Use internal _load_persons

                    # Prepare a dictionary to store all masks for the current frame
                    frame_masks = {}
                    H, W = self.image_size
                    dummy_mask_tensor = torch.zeros((H, W), dtype=torch.bool)

                    # Iterate through each person_id and load/process their mask
                    for person_id in person_ids:
                        mask_path = os.path.join(self.data_folder, traj_name, f'{time}.csv')
                        current_person_mask = dummy_mask_tensor # Start with dummy

                        try:
                            df = pd.read_csv(mask_path)
                            if not df.empty:
                                col_name = str(person_id)
                                if col_name in df.columns:
                                    col = df[col_name].to_numpy(dtype=np.uint8)
                                    padding_needed = H * W - col.size
                                    col = np.pad(col, (0, padding_needed), 'constant', constant_values=0)
                                    current_person_mask = torch.from_numpy(col.reshape(H, W).astype(bool))
                        except pd.errors.EmptyDataError:
                            pass # File is empty, current_person_mask remains dummy
                        except FileNotFoundError:
                            pass # CSV file not found, current_person_mask remains dummy
                        except Exception as e:
                            print(f"Warning: Error processing mask for {person_id} in {traj_name} at time {time}: {e}. Using dummy mask.")
                            pass

                        frame_masks[str(person_id)] = current_person_mask.numpy().tobytes() # Store as bytes

                    # Store the pickled dictionary of masks for this frame
                    # The key should uniquely identify the frame and mask type, e.g., "traj_name_time_masks"
                    mask_key = f"{traj_name}_{time}_masks".encode()
                    mask_txn.put(mask_key, pickle.dumps(frame_masks))


    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            # Consider returning a dummy black image or raising an error if image is critical
            # For robustness, returning a dummy image (e.g., zeros) might be better than crashing
            H, W = self.image_size
            return torch.zeros((3, H, W), dtype=torch.float32)


    def _get_selected(self, trajectory_name, curr_time):
        if trajectory_name not in self.select_ids_cache:
            with open(os.path.join(self.data_folder, trajectory_name, "select_ids.pkl"), "rb") as f:
                select_ids_list = pickle.load(f)
            self.select_ids_cache[trajectory_name] = select_ids_list  # 直接存储列表
        select_ids = self.select_ids_cache[trajectory_name][curr_time]  # 获取当前时间帧的select_ids
        return select_ids


    def _load_persons(self, trajectory_name, curr_time):
        """
        Load person IDs for the current frame
        Args:
            trajectory_name: name of trajectory
            curr_time: current frame index
        Returns:
            tuple: (person_ids, labels)
        """
        if trajectory_name not in self.person_ids_cache:
            with open(os.path.join(self.data_folder, trajectory_name, "person_ids.pkl"), "rb") as f:
                person_ids_list = pickle.load(f)
            self.person_ids_cache[trajectory_name] = person_ids_list
        
        try:
            person_ids = self.person_ids_cache[trajectory_name][curr_time]
        except IndexError:
            print(f"ERROR in trajectory: {trajectory_name}")
            print(f"Current time: {curr_time}")
            print(f"List length: {len(self.person_ids_cache[trajectory_name])}")
            print(f"Content preview: {self.person_ids_cache[trajectory_name][:10]}")
            # If this is called during _build_caches, it might mask the root issue.
            # If it's called during __getitem__, it might be a data consistency issue.
            # Deciding whether to raise or return empty depends on expected behavior.
            return [], [] # Returning empty lists for person_ids and labels
        
        select_ids = self._get_selected(trajectory_name, curr_time)
        labels = [1 if tid in select_ids else 0 for tid in person_ids]

        return person_ids, labels
    

    def _load_person_mask(self, trajectory_name, time, person_id):
        """
        Load mask for person_id at given frame from LMDB cache, return tensor HxW bool.
        """
        H, W = self.image_size
        dummy_mask = torch.zeros((H, W), dtype=torch.bool)
        
        mask_key = f"{trajectory_name}_{time}_masks".encode()
        
        try:
            with self._mask_cache.begin() as txn:
                frame_masks_bytes = txn.get(mask_key)
                if frame_masks_bytes is None:
                    # Key not found, indicating no masks or an issue during caching
                    return dummy_mask
                
                frame_masks_dict = pickle.loads(frame_masks_bytes)
                
                person_id_str = str(person_id)
                if person_id_str in frame_masks_dict:
                    mask_bytes = frame_masks_dict[person_id_str]
                    mask_np = np.frombuffer(mask_bytes, dtype=bool).reshape(H, W)
                    return torch.from_numpy(mask_np.copy())
                else:
                    # Person ID not found in the dictionary for this frame
                    return dummy_mask
        except Exception as e:
            print(f"Failed to load mask for person {person_id} in {trajectory_name} at time {time} from cache: {e}")
            return dummy_mask


    def _load_fixations(self, trajectory_name, curr_time):
        """
        Load fixation data and create a Gaussian attention map
        Returns:
            torch.Tensor: Gaussian attention map of shape (1, H, W)
        """
        if trajectory_name not in self.fixations_cache:
            with open(os.path.join(self.data_folder, trajectory_name, "fixations.pkl"), "rb") as f:
                fixations_data = pickle.load(f)
            self.fixations_cache[trajectory_name] = fixations_data
            
        fixations_df = self.fixations_cache[trajectory_name]
        
        H, W = self.image_size
        dummy_attention_map = torch.zeros((1, H, W), dtype=torch.float32)

        try:
            fx, fy = 0, 0 
            
            # Check current time's fixation
            if curr_time < len(fixations_df):
                current_fx, current_fy = fixations_df.iloc[curr_time][0], fixations_df.iloc[curr_time][1]
                if not (current_fx == 0 and current_fy == 0):
                    fx, fy = current_fx, current_fy
            
            # If current fixation is (0,0), search for the nearest non-(0,0) by alternating
            if fx == 0 and fy == 0:
                found_fixation = False
                max_offset = max(curr_time, len(fixations_df) - 1 - curr_time)

                for offset in range(1, max_offset + 1):
                    # Check backward
                    idx_b = curr_time - offset
                    if idx_b >= 0:
                        temp_fx_b, temp_fy_b = fixations_df.iloc[idx_b][0], fixations_df.iloc[idx_b][1]
                        if not (temp_fx_b == 0 and temp_fy_b == 0):
                            fx, fy = temp_fx_b, temp_fy_b
                            found_fixation = True
                            break
                    
                    # Check forward
                    idx_f = curr_time + offset
                    if idx_f < len(fixations_df):
                        temp_fx_f, temp_fy_f = fixations_df.iloc[idx_f][0], fixations_df.iloc[idx_f][1]
                        if not (temp_fx_f == 0 and temp_fy_f == 0):
                            fx, fy = temp_fx_f, temp_fy_f
                            found_fixation = True
                            break
                
                # If still (0,0) after searching, return dummy map
                if not found_fixation:
                    print(f"No non-(0,0) fixations found for {trajectory_name} around time {curr_time}. Returning dummy map.")
                    return dummy_attention_map
            
            fixations = [(fx, fy)] 

        except IndexError: # 确保处理超出索引的情况
            print(f"Error loading fixations for {trajectory_name} at time {curr_time}: Index out of bounds. Returning dummy map.")
            return dummy_attention_map
        except Exception as e:
            print(f"Error loading fixations for {trajectory_name} at time {curr_time}: {e}. Returning dummy map.")
            return dummy_attention_map
        
        
        # Create empty attention map
        attention_map = torch.zeros((1, H, W), dtype=torch.float32)
        
        # Generate Gaussian kernel
        sigma = 10.0  # Standard deviation in pixels
        x = torch.arange(0, W)
        y = torch.arange(0, H)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # Add Gaussian for each fixation point
        for fx, fy in fixations:
            # Convert fixation coordinates to integers
            fx = int(fx)
            fy = int(fy)
            
            # Skip if fixation is outside image bounds
            if fx < 0 or fx >= W or fy < 0 or fy >= H:
                continue
                
            # Generate 2D Gaussian centered at fixation
            gaussian = torch.exp(-((x - fx)**2 + (y - fy)**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.max()  # Normalize to [0,1]
            
            # Add to attention map
            attention_map[0] = torch.maximum(attention_map[0], gaussian)
        
        return attention_map
            

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data


    def _compute_actions(self, traj_data, curr_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * 3 + 1

        pos = traj_data.iloc[start_index:end_index:3, :2].to_numpy(dtype=np.float64)

        if pos.shape != (self.len_traj_pred + 1, 2):
            print(f"{pos.shape} and {(self.len_traj_pred + 1, 2)} should be equal")

        waypoints = pos - pos[0]
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        actions = waypoints[1:]

        return actions
    

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        """
        traj_name, curr_time = self.samples_index[i]
        context = [curr_time + i for i in range(-self.context_size, 1)]

        # Load images and gaze data
        obs_images = torch.cat([self._load_image(traj_name, t) for t in context])
        gaze_maps = torch.cat([self._load_fixations(traj_name, t) for t in context])

        # Load person IDs, labels and masks
        person_ids, select_labels = self._load_persons(traj_name, curr_time)
        
        # Handle empty person case
        if len(person_ids) == 0:
            H, W = self.image_size
            # Create dummy tensors with correct dimensions
            person_masks = torch.zeros((1, self.context_size + 1, H, W), dtype=torch.bool)
            select_labels = torch.zeros((1,), dtype=torch.bool)
        else:
            # Load masks from LMDB cache
            person_masks_list = []
            for tid in person_ids:
                seq = [self._load_person_mask(traj_name, t, tid) for t in context]
                person_masks_list.append(torch.stack(seq, dim=0))  # (context_size+1, H, W)
            person_masks = torch.stack(person_masks_list, dim=0).bool()  # (num_persons, context_size+1, H, W)
            select_labels = torch.tensor(select_labels, dtype=torch.bool) # (num_persons,)

        # Load trajectory data
        curr_traj_data = self._get_trajectory(traj_name)
        curr_traj_len = len(curr_traj_data)
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Compute actions
        actions = self._compute_actions(curr_traj_data, curr_time)

        
        return (
            torch.as_tensor(obs_images, dtype=torch.float32),
            torch.as_tensor(gaze_maps, dtype=torch.float32),
            torch.as_tensor(person_masks, dtype=torch.bool),
            torch.as_tensor(select_labels, dtype=torch.bool),
            torch.as_tensor(actions, dtype=torch.float32)
        )