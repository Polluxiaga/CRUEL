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
        self._load_index()
        self._build_caches()


    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()


    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["pos_x(m)"])

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


    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"{self.dataset_name}_images.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)


    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")


    def _load_gaze(self, trajectory_name, time):
        """
        Load gaze attention data from pkl file
        """
        pkl_path = get_data_path(self.data_folder, trajectory_name, time).replace('.jpg', '.pkl')
        try:
            # 使用 pandas 正确读取 DataFrame 类型的 pkl
            df = pd.read_pickle(pkl_path)
            gaze_data = torch.tensor(df.to_numpy(), dtype=torch.float32)
            return gaze_data
        except Exception as e:
            print(f"Failed to load gaze data {pkl_path}: {e}")
            return None


    def _compute_actions(self, traj_data, curr_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * 3 + 1
        pos = traj_data[["pos_x(m)", "pos_y(m)"]][start_index:end_index:3].to_numpy(dtype=np.float64)
        pos = np.stack([np.array(item) for item in pos], axis=0)

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
        Returns:
            Tuple of tensors containing the context, observation, gaze attention and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                gaze_attention (torch.Tensor): tensor of shape [N] containing the gaze attention data
                action_label (torch.Tensor): tensor of shape (3, 2) containing the action labels
        """
        f_curr, curr_time = self.samples_index[i]

        # Load images and gaze data
        obs_times = [curr_time + i for i in range(-self.context_size, 1)]
        obs_context = [(f_curr, t) for t in obs_times]

        obs_image = torch.cat([self._load_image(f, t) for f, t in obs_context])
        gaze_attention = torch.cat([self._load_gaze(f, t) for f, t in obs_context])

        # Load trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["pos_x(m)"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # Compute actions
        actions = self._compute_actions(curr_traj_data, curr_time)
        
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(gaze_attention, dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.float32)
        )