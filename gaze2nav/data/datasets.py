"""Dataset loaders for Gaze2Nav training and artifact generation.

``ObsDataset`` feeds gaze and salient-person prediction models. ``ActDataset``
feeds motion planners with RGB history plus gaze maps or selected-person masks.
"""

import numpy as np
import pandas as pd
import os
import pickle
import tqdm
import io
import lmdb
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset

from gaze2nav.data.utils import (
    img_path_to_data,
    get_data_path,
)


class BaseDataset(Dataset):
    # Class-level flag avoids rebuilding the same LMDB cache repeatedly.
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
        super().__init__()
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

        # Small in-memory caches for trajectory-local metadata.
        self.trajectory_cache = {}
        self.fixations_cache = {}
        self.person_ids_cache = {}
        self.select_ids_cache = {} # This will store the select_ids for winner_labels

        self._load_index() # Load or build samples_index and goals_index

        # Use cache path as unique key for tracking built status
        # Use two separate LMDBs for images and masks for clarity and modularity
        self._image_cache_path = os.path.join(
            self.data_split_folder,
            f"{self.dataset_name}_images.lmdb",
        )
        self._mask_cache_path = os.path.join( # New cache path for masks
            self.data_split_folder,
            f"{self.dataset_name}_masks_v2.lmdb",
        )

        # Only build caches if not already built for this cache path
        if self._image_cache_path not in self._caches_built: # Check image cache build status
            self._build_caches()
            self._caches_built[self._image_cache_path] = True

        # Always open the LMDB environment for both images and masks
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
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # if the index_to_data file doesn't exist, create it
            self.samples_index, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.samples_index, self.goals_index), f)



    def _build_caches(self, use_tqdm: bool = True):
        """Build LMDB caches for RGB frames and per-person binary masks."""
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
                    W, H = self.image_size
                    dummy_mask_tensor = torch.zeros((H, W), dtype=torch.bool)

                    # Iterate through each person_id and load/process their mask
                    for person_id in person_ids:
                        mask_path = os.path.join(self.data_folder, traj_name, f'{time}.csv')
                        current_person_mask = dummy_mask_tensor # Start with dummy

                        try:
                            df = pd.read_csv(mask_path, header=None)
                            if not df.empty:
                                csv_person_ids = df.iloc[0, :].tolist()
                                csv_person_ids_as_str = [str(pid) for pid in csv_person_ids]
                                person_id_str = str(person_id)

                                if person_id in csv_person_ids:
                                    col_idx = csv_person_ids.index(person_id)
                                elif person_id_str in csv_person_ids_as_str:
                                    col_idx = csv_person_ids_as_str.index(person_id_str)
                                else:
                                    col_idx = None

                                if col_idx is not None and col_idx < df.shape[1]:
                                    col = df.iloc[1:, col_idx].to_numpy(dtype=np.uint8)
                                    expected_size = H * W
                                    if col.size < expected_size:
                                        col = np.pad(col, (0, expected_size - col.size), 'constant', constant_values=0)
                                    elif col.size > expected_size:
                                        col = col[:expected_size]
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
        """Load one cached RGB frame as a float tensor."""
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
            W, H = self.image_size
            return torch.zeros((3, H, W), dtype=torch.float32)


    def _get_selected(self, trajectory_name, curr_time):
        """Load ground-truth salient person IDs for a frame."""
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
            # This should ideally not happen if _build_index handles short trajectories correctly
            print(f"ERROR: Person IDs index out of bounds for {trajectory_name} at time {curr_time}. Returning empty.")
            return [], [] # Returning empty lists for person_ids and labels

        select_ids = self._get_selected(trajectory_name, curr_time)
        gt_labels = [1 if tid in select_ids else 0 for tid in person_ids]

        return person_ids, gt_labels


    def _load_person_mask(self, trajectory_name, time, person_id):
        """
        Load mask for person_id at given frame from LMDB cache, return tensor HxW bool.
        """
        W, H = self.image_size
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
        Load fixation data for the given trajectory and time
        Args:
            trajectory_name (str): Name of the trajectory
            curr_time (int): Current time index
        Returns:
            list: List of tuples containing (x,y) fixation coordinates
        """
        if trajectory_name not in self.fixations_cache:
            with open(os.path.join(self.data_folder, trajectory_name, "fixations.pkl"), "rb") as f:
                fixations_data = pickle.load(f)
            self.fixations_cache[trajectory_name] = fixations_data

        fixations_df = self.fixations_cache[trajectory_name]

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

                # If still (0,0) after searching, return empty list
                if not found_fixation:
                    print(f"No non-(0,0) fixations found for {trajectory_name} around time {curr_time}. Returning empty list.")
                    return []

            return [(fx, fy)]

        except IndexError:
            print(f"Error loading fixations for {trajectory_name} at time {curr_time}: Index out of bounds. Returning empty list.")
            return []
        except Exception as e:
            print(f"Error loading fixations for {trajectory_name} at time {curr_time}: {e}. Returning empty list.")
            return []

    def _load_gazemaps(self, trajectory_name, curr_time):
        """
        Create a Gaussian attention map
        Returns:
            torch.Tensor: Gaussian attention map of shape (1, H, W)
        """
        W, H = self.image_size
        dummy_attention_map = torch.zeros((1, H, W), dtype=torch.float32)

        # Get fixation points from _load_fixations
        fixations = self._load_fixations(trajectory_name, curr_time)
        if not fixations:
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
        """Load and cache trajectory positions for a trajectory."""
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

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError("BaseDataset does not implement __getitem__. Use a specific dataset.")


class ObsDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        len_traj_pred: int,
        context_size: int,
        obs_type: str = "image",
        use_generated_attnmaps: bool = False,
        generated_attnmaps_path: Optional[str] = None,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name,
            image_size, len_traj_pred, context_size, obs_type)

        self.use_generated_attnmaps = use_generated_attnmaps
        self._cached_generated_attnmaps : Optional[List[torch.Tensor]]=None

        # 如果在初始化时指定了路径，尝试从磁盘加载预生成的标签
        if self.use_generated_attnmaps and generated_attnmaps_path and os.path.exists(generated_attnmaps_path):
            print(f"Loading generated attnmaps from disk: {generated_attnmaps_path}...")
            try:
                loaded_preds_list = torch.load(generated_attnmaps_path)
                if not isinstance(loaded_preds_list, list) or not all(isinstance(p, torch.Tensor) for p in loaded_preds_list):
                     raise TypeError("Generated attnmaps file should contain a list of tensors.")

                # 存储为列表，通过索引访问
                self._cached_generated_attnmaps = loaded_preds_list
                print(f"Loaded {len(self._cached_generated_attnmaps)} generated attnmaps samples from disk.")
            except Exception as e:
                print(f"Warning: Failed to load generated attnmaps from {generated_attnmaps_path}: {e}. Proceeding without them.")
                self.use_generated_attnmaps = False # 加载失败则禁用
                self._cached_generated_attnmaps = None
        elif self.use_generated_attnmaps and not generated_attnmaps_path:
             print("Warning: use_generated_attnmaps is True but no generated_attnmaps_path provided.")


    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        """
        traj_name, curr_time = self.samples_index[i]
        context = [curr_time + i for i in range(-self.context_size, 1)]

        # Load images
        obs_images = torch.cat([self._load_image(traj_name, t) for t in context])
        fixation_tensors = []
        for t in context:
            frame_fixations = self._load_fixations(traj_name, t)
            fixation_xy = frame_fixations[0] if frame_fixations else (0, 0)
            fixation_tensors.append(torch.tensor(fixation_xy, dtype=torch.float32))
        fixations = torch.stack(fixation_tensors)

        # Load person IDs, labels and masks
        person_ids, gt_winner_labels_list = self._load_persons(traj_name, curr_time)
        gt_winner_labels_tensor = torch.tensor(gt_winner_labels_list, dtype=torch.bool)

        # Handle empty person case
        if len(person_ids) == 0:
            W, H = self.image_size
            # Create dummy tensors with correct dimensions
            person_masks = torch.zeros((1, self.context_size + 1, H, W), dtype=torch.bool)
            winner_labels_to_return = torch.zeros((1,), dtype=torch.bool)
        else:
            # Load masks from LMDB cache
            person_masks_list = []
            for tid in person_ids:
                seq = [self._load_person_mask(traj_name, t, tid) for t in context]
                person_masks_list.append(torch.stack(seq, dim=0))  # (context_size+1, H, W)
            person_masks = torch.stack(person_masks_list, dim=0).bool()  # (num_persons, context_size+1, H, W)

            # Determine which winner_labels to return: GT or generated
            winner_labels_to_return = gt_winner_labels_tensor # ObsModel always trains on GT labels

        # Get generated attention map if enabled, otherwise use a dummy/None
        # The WinnerSelector will consume this generated attention map as an input
        generated_attention_map_to_use = None

        if self.use_generated_attnmaps: # Check if flag is true
            if self._cached_generated_attnmaps is not None:
                # 获取缓存中存储的注意力图 (N * spatial_flatten_len)
                generated_attention_map_from_cache = self._cached_generated_attnmaps[i]

                if generated_attention_map_from_cache.ndim != 1:
                    print(f"Warning: Cached attention map for index {i} has unexpected dimensions: {generated_attention_map_from_cache.shape}. Expected 1D (N*spatial_flatten_len).")

                generated_attention_map_to_use = generated_attention_map_from_cache
            else:
                # 如果 use_generated_attnmaps 为 True 但 _cached_generated_attnmaps 为 None，
                # 说明加载失败或未设置，此时应返回一个指示，或者一个全零/占位符。
                # 返回 None 让 collate_fn 处理，或者根据需要返回一个零张量。
                print(f"Warning: Generated attention map for index {i} not found in cache despite use_generated_attnmaps being True. Returning None.")
                generated_attention_map_to_use = None # 显式设置为 None
        else:
            generated_attention_map_to_use = torch.zeros(120, dtype=torch.float32)

        # Return order for ObsModel: obs_images, person_masks, winner_labels, invalid_flags, original_index, generated_attention_map
        return (
            obs_images,
            fixations,
            person_masks,
            generated_attention_map_to_use,
            winner_labels_to_return, # GT winner labels
            torch.tensor(i, dtype=torch.long), # Original dataset index
        )


class ActDataset(BaseDataset):

    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        len_traj_pred: int,
        context_size: int,
        obs_type: str = "image",
        use_generated_labels: bool = False,
        generated_labels_path: Optional[str]=None,
    ):
        super().__init__(
            data_folder, data_split_folder, dataset_name,
            image_size, len_traj_pred, context_size, obs_type
        )

        self.use_generated_labels = use_generated_labels
        self._cached_generated_winner_labels = None
        self._cached_generated_gaze_maps_individual = None

        self.generated_labels_type = None  # winner_labels or gaze_maps

        # 尝试从磁盘加载生成标签，如果 use_generated_labels 为 True 且提供了路径
        if self.use_generated_labels and generated_labels_path and os.path.exists(generated_labels_path):
            print(f"Loading generated labels from disk: {generated_labels_path}...")
            try:
                loaded_data = torch.load(generated_labels_path)
                # 判断加载的数据类型
                if "phase" in generated_labels_path:
                    # 加载的是 winner_labels
                    if not isinstance(loaded_data, list) or not all(isinstance(p, torch.Tensor) for p in loaded_data):
                        raise TypeError("Generated winner labels file should contain a list of tensors.")
                    self._cached_generated_winner_labels = loaded_data
                    self.generated_labels_type = "winner_labels"
                    print(f"Loaded {len(self._cached_generated_winner_labels)} generated winner label samples from disk.")

                elif "gaze" in generated_labels_path:
                    # 加载的是 gaze_maps
                    if not isinstance(loaded_data, dict) or not all(isinstance(k, tuple) and isinstance(v, torch.Tensor) for k, v in loaded_data.items()):
                        raise TypeError("Generated gaze maps file should contain a list of tensors.")
                    self._cached_generated_gaze_maps_individual = loaded_data
                    self.generated_labels_type = "gaze_maps"
                    print(f"Loaded {len(self._cached_generated_gaze_maps_individual)} generated gaze map samples from disk.")
                else:
                    # 路径中不包含 "phase" 也不包含 "gaze"，无法判断类型
                    print(f"Warning: generated_labels_path '{generated_labels_path}' does not contain 'phase' or 'gaze'. Unable to determine type of generated labels. Proceeding without them.")
                    self.use_generated_labels = False # 禁用生成标签
                    self._cached_generated_winner_labels = None
                    self._cached_generated_gaze_maps_individual = None

            except Exception as e:
                print(f"Warning: Failed to load generated labels from {generated_labels_path}: {e}. Proceeding without them.")
                self.use_generated_labels = False # 加载失败则禁用
                self._cached_generated_winner_labels = None
                self._cached_generated_gaze_maps_individual = None
        elif self.use_generated_labels and not generated_labels_path:
             print("Warning: use_generated_labels is True but no generated_labels_path provided.")
             self.use_generated_labels = False


    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        traj_name, curr_time = self.samples_index[i]
        context = [curr_time + i for i in range(-self.context_size, 1)]

        # Load images and gaze data
        obs_images = torch.cat([self._load_image(traj_name, t) for t in context])

        # Determine which gaze maps to return: GT or generated
        gaze_maps_to_return = None
        if self.use_generated_labels and self.generated_labels_type == "gaze_maps":
            if self._cached_generated_gaze_maps_individual is not None:
                extracted_gaze_maps_list: List[torch.Tensor] = []
                for t in context:
                    # 从缓存的字典中获取单个预测的 gaze map
                    key = (traj_name, t)
                    predicted_gaze_map_for_frame = self._cached_generated_gaze_maps_individual.get(key)

                    if predicted_gaze_map_for_frame is not None:
                        extracted_gaze_maps_list.append(predicted_gaze_map_for_frame.unsqueeze(0)) # 确保形状是 (1, H, W)
                    else:
                        print(f"ActDataset Warning: Individual predicted gaze map for {key} not found in cache. Using GT gaze map.")
                        extracted_gaze_maps_list.append(self._load_gazemaps(traj_name, t))

                # 将所有获取到的 gaze maps 拼接起来形成上下文序列
                gaze_maps_to_return = torch.cat(extracted_gaze_maps_list, dim=0) # 结果形状 (context_size + 1, H, W)
            else:
                print(f"Error: Generated gaze maps for index {i} not found in cache. Returning GT gaze maps.")
                gaze_maps_to_return = torch.cat([self._load_gazemaps(traj_name, t) for t in context])
        else:
            # 使用 GT gaze maps
            gaze_maps_to_return = torch.cat([self._load_gazemaps(traj_name, t) for t in context])

        # Load person IDs, labels and masks
        person_ids, gt_winner_labels_list = self._load_persons(traj_name, curr_time)
        gt_winner_labels_tensor = torch.tensor(gt_winner_labels_list, dtype=torch.bool)

        # Handle empty person case
        if len(person_ids) == 0:
            W, H = self.image_size
            # Create dummy tensors with correct dimensions
            person_masks = torch.zeros((1, self.context_size + 1, H, W), dtype=torch.bool)
            winner_labels_to_return = torch.zeros((1,), dtype=torch.bool)
        else:
            # Load masks from LMDB cache
            person_masks_list = []
            for tid in person_ids:
                seq = [self._load_person_mask(traj_name, t, tid) for t in context]
                person_masks_list.append(torch.stack(seq, dim=0))  # (context_size+1, H, W)
            person_masks = torch.stack(person_masks_list, dim=0).bool()  # (num_persons, context_size+1, H, W)

            # Determine which winner_labels to return: GT or generated
            winner_labels_to_return = None
            if self.use_generated_labels and self.generated_labels_type == "winner_labels":
                if self.use_generated_labels and self._cached_generated_winner_labels is not None:
                    # 获取缓存中存储的原始未填充预测标签 (P_actual,)
                    generated_labels_from_cache = self._cached_generated_winner_labels[i]

                    # 检查加载的预测标签长度是否与当前样本的实际人数匹配
                    if generated_labels_from_cache.shape[0] != len(person_ids):
                        print(f"Warning: Generated labels shape mismatch for index {i}. Expected {len(person_ids)}, got {generated_labels_from_cache.shape[0]}. Using dummy labels.")
                        winner_labels_to_return = torch.zeros((len(person_ids),), dtype=torch.bool)
                    else:
                        winner_labels_to_return = generated_labels_from_cache
                else:
                    print(f"Error: Generated labels for index {i} not found in cache. Returning dummy labels.")
                    winner_labels_to_return = torch.zeros((len(person_ids),), dtype=torch.bool)
            else:
                # 使用 GT 标签
                winner_labels_to_return = gt_winner_labels_tensor


        # Load trajectory data
        curr_traj_data = self._get_trajectory(traj_name)
        curr_traj_len = len(curr_traj_data)
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Compute actions
        actions = self._compute_actions(curr_traj_data, curr_time)


        return (
            torch.as_tensor(obs_images, dtype=torch.float32),
            torch.as_tensor(gaze_maps_to_return, dtype=torch.float32),
            torch.as_tensor(person_masks, dtype=torch.bool),
            torch.as_tensor(winner_labels_to_return, dtype=torch.bool),
            torch.as_tensor(actions, dtype=torch.float32),
            torch.tensor(i, dtype=torch.long),  # Original dataset index (供 collate_fn 或 generate_selector_predictions 使用)
            traj_name
        )
