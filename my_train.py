import os
from typing import Optional, Dict, List, Tuple
import torch.nn.functional as F
import argparse
import tqdm
import yaml
import wandb
import numpy as np
import time

# 解析命令行参数
parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
parser.add_argument("--config", "-c", default="configs/vint_config.yaml", help="Path to config file")
args = parser.parse_args()

# 加载配置文件
with open("configs/vint_config.yaml", "r") as f:
    default_config = yaml.safe_load(f)
config = default_config.copy()
with open(args.config, "r") as f:
    user_config = yaml.safe_load(f)
config.update(user_config)

# 强制设置GPU可见性（不检查CUDA是否可用）
if "gpu_id" in config:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
    print(f"强制将物理GPU {config['gpu_id']} 映射为逻辑GPU 0")

# 现在导入PyTorch及相关模块
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from my_data.my_dataset import ObsDataset, ActDataset
from my_model.act_models import vint_model, channel_model, catoken_model, gnm_model, gnmchannel_model
from my_model.observe_models import WinnerSelectorPlus, GazePredictorPlus
from my_training.my_train_utils import act_person_collate_fn, obs_person_collate_fn, act_base_collate_fn, obs_base_collate_fn, render_fixations_to_gaze_maps
from my_training.my_train_eval_loop import train_eval_loop, load_model, count_parameters


def generate_attnmap(
    model: nn.Module,
    dataloader: DataLoader, # This DataLoader MUST provide (obs_img, attention, ...) compatible with catoken_model
    device: torch.device,
    save_path: str
):
    """
    Generate the RGB attention maps from the act_model (catoken_model) for supervision
    and save them to disk.

    Args:
        model (nn.Module): The act_model (catoken_model) to generate attention maps from.
        dataloader (DataLoader): DataLoader providing (obs_img, attention, ...) compatible with catoken_model.
        device (torch.device): The device (CPU or GPU) to run the model on.
        save_path (str): The file path where the list of attention maps will be saved.
                         Example: "data_splits/train/genattnmap.pt"
    """

    model.eval() # Set model to evaluation mode
    
    total_samples = len(dataloader.dataset)
    # Initialize list to store attention maps in original dataset order
    # Each element will be a tensor of shape (N * spatial_flatten_len)
    all_unpadded_rgb_attention_maps = [None] * total_samples 
    
    # We need to determine H_feature * W_feature from the model's structure
    # A safer way is to infer it or pass it. Let's infer during the first batch.
    H_feature_val = None
    W_feature_val = None
    spatial_flatten_len_val = None

    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(dataloader, desc="Generating persontoken Attention Maps", dynamic_ncols=True)
        for batch_data in tqdm_iter:
            # Unpack the batched data, matching persontoken_train's data unpacking
            (
                obs_image,
                person_masks,
                select_labels,
                _, # Unused (e.g., action_label)
                invalid,
                original_indices_batch,
                _
            ) = batch_data

            # Move tensors to device
            obs_image = obs_image.to(device)
            person_masks = person_masks.to(device)
            select_labels = select_labels.to(device)
            invalid = invalid.to(device)
            original_indices_list = original_indices_batch.cpu().tolist()

            # --- Prepare person_attention input, exactly as in persontoken_train ---
            valid_masks = ~invalid
            select_mask = (select_labels == 1) & valid_masks
            select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            person_masks = person_masks * select_mask.float()
            person_attention = (person_masks.sum(dim=1) > 0).float() 

            # --- Infer spatial_flatten_len and N_frames_val if not already done ---
            if spatial_flatten_len_val is None:
                # Infer H_feature and W_feature using rgb_encoder's output for a single frame
                sample_rgb_for_feature_size = obs_image[0:1, 0:3].to(device) # Just one frame (3 channels)
                sample_features = model.rgb_encoder.extract_features(sample_rgb_for_feature_size)
                H_feature_val = sample_features.shape[2]
                W_feature_val = sample_features.shape[3]
                spatial_flatten_len_val = H_feature_val * W_feature_val
                
                # Infer N (number of frames) from input image channels / 3
                N_frames_val = obs_image.shape[1] // 3
                if N_frames_val != (model.context_size + 1):
                    print(f"Warning: Inferred N_frames ({N_frames_val}) does not match model's context_size+1 ({model.context_size+1}). Please ensure consistency.")

            # Call the act_model
            # attention_scores shape: (B, total_seq_len, total_seq_len)
            # where total_seq_len = N * spatial_flatten_len (RGB) + N * spatial_flatten_len (Attention)
            _, attention_scores_from_act_model = model(obs_image, person_attention)
            
            # --- Extract RGB-only Attention Map from the returned attention_scores ---
            # The length of the RGB token sequence in the decoder's input
            # The first half of the sequence for both queries and keys.
            rgb_token_segment_length = N_frames_val * spatial_flatten_len_val
            
            attn_to_rgb_tokens = attention_scores_from_act_model[:, :, :rgb_token_segment_length]  # Shape: (B, total_seq_len, rgb_token_segment_length)

            # Sum attention across the query dimension (dim=1) to get total attention received by each RGB key token.
            rgb_token_saliency = attn_to_rgb_tokens.sum(dim=1)  # (B, N * spatial_flatten_len)

            # Softmax normalize this sequence to get a probability distribution over the RGB tokens.
            normalized_rgb_attention_sequence = F.softmax(rgb_token_saliency, dim=1)
            
            # Iterate through each sample in the current batch
            for i_sample_in_batch in range(normalized_rgb_attention_sequence.shape[0]):
                original_dataset_idx = original_indices_list[i_sample_in_batch]
                
                single_sample_attn_map = normalized_rgb_attention_sequence[i_sample_in_batch].cpu()
                all_unpadded_rgb_attention_maps[original_dataset_idx] = single_sample_attn_map
    
    # Save the generated attention maps to disk
    output_name = "attnmap_used.pt"
    full_save_path = os.path.join(save_path, output_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
    torch.save(all_unpadded_rgb_attention_maps, full_save_path)
    print(f"Generated {output_name} saved to: {save_path}")


def generate_gaze(
        method: str,
        model: nn. Module,
        dataloader: DataLoader, # This DataLoader MUST use obs_base_collate_fn
        device: torch.device,
        save_path: str,
        H: int=128,
        W: int=160,
        context_size: int=5,
        sigma: float = 10.0
):
    """
    Generates gaze maps from predicted fixation data (obtained from the GazePredictor(plus))
    by applying Gaussian kernels and saves them to disk.

    Args:
        model (nn.Module): The Gaze Prediction Model that outputs fixations given obs_images.
        dataloader (DataLoader): DataLoader that provides batch_data including obs_images.
                                 It should return (obs_images, ..., original_indices_batch).
        device (torch.device): The device (CPU or GPU) to run the model and operations on.
        save_path (str): The directory path where the generated gaze maps will be saved.
        H (int): Height of the image/gaze map.
        W (int): Width of the image/gaze map.
        context_size (int): Number of context frames. Total frames will be context_size + 1.
        sigma (float): Standard deviation for the Gaussian kernel.
    """
    model.eval() # Set model to evaluation mode for prediction
    
    # Store generated gaze maps as { (traj_name, curr_time): single_predicted_gaze_map_tensor }
    # This will be used by ActDataset to look up individual predicted gaze maps.
    all_generated_gaze_maps_individual: Dict[Tuple[str, int], torch.Tensor] = {}

    # Pre-generate meshgrid for the image size, move to device
    x_coords = torch.arange(0, W, device=device)
    y_coords = torch.arange(0, H, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    tqdm_iter = tqdm.tqdm(dataloader, desc=f"Generating {method} Individual Gaze Maps", dynamic_ncols=True)
    
    # Access the dataset instance to get trajectory names and current times
    dataset_instance = dataloader.dataset
    
    # 检查是否是 ConcatDataset
    if isinstance(dataset_instance, torch.utils.data.ConcatDataset):
        print("Detected ConcatDataset. Building combined samples_index from constituent datasets.")
        combined_samples_index = []
        for ds in dataset_instance.datasets:
            # 确保每个子数据集都有 samples_index 属性
            if hasattr(ds, 'samples_index'):
                combined_samples_index.extend(ds.samples_index)
            else:
                raise AttributeError(
                    f"A dataset within ConcatDataset ({type(ds)}) is missing 'samples_index' attribute."
                    "Ensure all concatenated datasets are derived from BaseDataset or ActDataset."
                )
        # 使用组合后的索引列表进行查找
        effective_samples_index = combined_samples_index
        print(f"Combined samples_index length: {len(effective_samples_index)}")
    else:
        # 如果不是 ConcatDataset，则直接使用数据集自身的 samples_index
        if not hasattr(dataset_instance, 'samples_index'):
            raise AttributeError("Dataloader's dataset must have a 'samples_index' attribute (e.g., from BaseDataset).")
        effective_samples_index = dataset_instance.samples_index


    with torch.no_grad():
        for batch_data in tqdm_iter:
            # Unpack the batched data based on ActDataset __getitem__ return:
            # (obs_images_N_frames_stacked, gaze_maps_all_N_frames_gt, person_masks, winner_labels_to_return, actions, original_indices_batch)
            obs_images_N_frames_stacked, fixations_all_N_frames_gt, _, _, original_indices_batch = batch_data 
            
            # Move data to device
            obs_images_N_frames_stacked = obs_images_N_frames_stacked.to(device) # [B, (N_frames)*3, H, W]
            
            fixations_all_N_frames_gt = fixations_all_N_frames_gt.to(device)
            prev_fixation = fixations_all_N_frames_gt[:, :-1, :]  # [B, C, 2] - previous fixations

            prev_gaze_maps_for_model = render_fixations_to_gaze_maps(
                fixations_batch=prev_fixation,
                H=128, # 使用传入的图像高度
                W=160,  # 使用传入的图像宽度
                sigma=10.0, # 使用传入的高斯核标准差
                device=device
            )

            # --- Obtain predicted fixation for the CURRENT (last) frame from the model ---
            # model outputs (fixation_point_current_frame, attention_map)
            predicted_fixations_current_frame_batch, _ = model(obs_images_N_frames_stacked, prev_gaze_maps_for_model) # [B, 2]

            for i_sample_in_batch in range(predicted_fixations_current_frame_batch.shape[0]):
                original_dataset_idx = original_indices_batch[i_sample_in_batch].item() # Get as scalar int
                
                # Retrieve (traj_name, curr_time) for this specific sample
                traj_name, curr_time = effective_samples_index[original_dataset_idx]
                
                # Create empty gaze map for this specific current frame prediction
                single_predicted_gaze_map = torch.zeros((1, H, W), dtype=torch.float32, device=device)
                
                fx, fy = predicted_fixations_current_frame_batch[i_sample_in_batch].detach().cpu().numpy() 

                # Ensure fixation is valid numbers and within bounds
                if not (np.isnan(fx) or np.isnan(fy)):
                    fx_int = round(float(fx))
                    fy_int = round(float(fy))
                    
                    if 0 <= fx_int < W and 0 <= fy_int < H:
                        gaussian = torch.exp(-((x_grid - fx_int)**2 + (y_grid - fy_int)**2) / (2 * sigma**2))
                        if gaussian.max() > 0:
                            gaussian = gaussian / gaussian.max() # Normalize to [0, 1]
                        single_predicted_gaze_map[0] = gaussian # Assign to single-channel map

                # Store the generated single-frame gaze map with its (traj_name, curr_time) key
                all_generated_gaze_maps_individual[(traj_name, curr_time)] = single_predicted_gaze_map.cpu()
    
    underlying_datasets = dataloader.dataset.datasets if isinstance(dataloader.dataset, torch.utils.data.ConcatDataset) else [dataloader.dataset]

    print("\nPopulating initial context frames with GT gaze maps...")
    num_added_gt = 0
    
    # 使用 tqdm 包装外部循环，提供进度条
    for ds_idx, ds in enumerate(tqdm.tqdm(underlying_datasets, desc="Processing underlying datasets for GT fill")):
        if not hasattr(ds, 'samples_index') or not hasattr(ds, '_get_trajectory') or not hasattr(ds, '_load_gazemaps'):
            print(f"Warning: Underlying dataset {type(ds)} (index {ds_idx}) does not support required methods (_get_trajectory, _load_gazemaps) for GT fill. Skipping.")
            continue
        
        # 收集该数据集实例中所有唯一的轨迹名称
        unique_traj_names_in_ds = sorted(list(set(item[0] for item in ds.samples_index)))

        for traj_name in tqdm.tqdm(unique_traj_names_in_ds, desc=f"Filling GT for traj in DS {ds_idx}", leave=False):
            try:
                # 获取该轨迹的总长度，以确保不会访问越界的帧
                # 假设 _get_trajectory(traj_name) 能返回一个可获取长度的数据结构
                traj_data = ds._get_trajectory(traj_name) 
                traj_len = len(traj_data)
            except Exception as e:
                print(f"Warning: Could not get trajectory length for {traj_name} from dataset {type(ds)}. Error: {e}. Skipping GT fill for this trajectory.")
                continue

            for t in range(min(context_size, traj_len)): 
                key = (traj_name, t)
                
                # 检查该键是否已经存在于字典中（理论上不应存在，因为这些 t < context_size）
                if key not in all_generated_gaze_maps_individual:
                    try:
                        # 从该数据集实例加载地面真实注视图
                        gt_gaze_map = ds._load_gazemaps(traj_name, t).cpu() 
                        all_generated_gaze_maps_individual[key] = gt_gaze_map
                        num_added_gt += 1
                    except Exception as e:
                        print(f"Warning: Failed to load GT gaze map for {key}. Error: {e}. Skipping this frame.")
    
    print(f"Added {num_added_gt} GT gaze maps for initial context frames (t < {context_size}).")

    # Save the generated predictions to disk
    output_filename = f"{method}.pt" # Example: "gaze.pt"
    full_save_file_path = os.path.join(save_path, output_filename)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(all_generated_gaze_maps_individual, full_save_file_path)
    print(f"Individual predicted gaze maps saved to: {full_save_file_path}")

    if all_generated_gaze_maps_individual:
        # Print an example key-value pair
        example_key = next(iter(all_generated_gaze_maps_individual.keys()))
        print(f"Example cached gaze map entry: Key={example_key}, Shape={all_generated_gaze_maps_individual[example_key].shape}")


def generate_1phase_winners(
    method: str,
    model: nn.Module,
    dataloader: DataLoader, # This DataLoader MUST use obs_person_collate_fn
    device: torch.device,
    save_path: str
):
    """
    Generate selector predictions (binary masks) from the obs_model and save them to disk.

    Args:
        model (nn.Module): The obs_model (WinnerSelector/WinnerSelectorPlus) to generate predictions from.
        dataloader (DataLoader): DataLoader using obs_person_collate_fn.
        device (torch.device): The device (CPU or GPU) to run the model on.
        save_path (str): The file path where the list of predictions will be saved.
                         Example: "data_splits/train/1phase_winners.pt", "data_splits/train/1phaseplus_winners.pt"
    """
    model.eval() # Set model to evaluation mode for prediction
    
    total_samples = len(dataloader.dataset)
    all_unpadded_predictions = [None] * total_samples # Initialize list to store results in original dataset order
    
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(dataloader, desc=f"Generating {method} Winners", dynamic_ncols=True)
        for batch_data in tqdm_iter:
            # Unpack the batched data from obs_person_collate_fn
            obs_batch, mask_batch, attnmap_batch, _, invalid_flag_batch, original_indices_batch = batch_data
            
            # Move tensors to device
            obs_batch = obs_batch.to(device)
            mask_batch = mask_batch.to(device)
            attnmap_batch = attnmap_batch.to(device)
            invalid_flag_batch = invalid_flag_batch.to(device)
            original_indices_list = original_indices_batch.cpu().tolist() # Get original dataset indices

            logits, _ = model(obs_batch, mask_batch, invalid_flag_batch) # (B, P_max)

            batched_preds_bool = (logits > 0).cpu().bool() # Convert logits to boolean predictions (B, P_max)
            
            # Iterate through each sample in the current batch
            for i_sample_in_batch in range(batched_preds_bool.shape[0]):
                original_dataset_idx = original_indices_list[i_sample_in_batch]
                
                # Extract the unpadded prediction for this sample
                # Use invalid_flag_batch to mask out padded persons
                unpadded_pred = batched_preds_bool[i_sample_in_batch, ~invalid_flag_batch[i_sample_in_batch]] # (P_actual,)
                
                # Store the unpadded prediction in the master list at its original index
                all_unpadded_predictions[original_dataset_idx] = unpadded_pred
                
    # Save the generated predictions to disk
    output_name = f"{method}.pt"
    full_save_path = os.path.join(save_path, output_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
    torch.save(all_unpadded_predictions, full_save_path)
    print(f"Generated {output_name} saved to: {save_path}")
    if len(all_unpadded_predictions) > 0:
        print(f"Example cached prediction shape (first sample): {all_unpadded_predictions[0].shape}")


def create_model_and_optimizer(method_type, config, device, lr, model_type="act_model", model_instance=None):
    """Helper function to create a model and its optimizer/scheduler."""
    model = model_instance
    if model is None:
        if model_type == "obs_model":
            if method_type in ["gaze", "gazeplus"]:
                model = GazePredictorPlus(
                    context_size=config["context_size"],
                ).to(device)
            elif method_type in ["1phase", "1phaseplus"]:
                model = WinnerSelectorPlus(
                    context_size=config["context_size"],
                ).to(device)

        elif model_type == "act_model":
            if method_type in ["gazechannel", "personchannel"]:
                model = channel_model(
                    method=method_type,
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoder=config["obs_encoder"],
                    encoding_size=config["encoding_size"],
                    mha_num_attention_heads=config["mha_num_attention_heads"],
                    mha_num_attention_layers=config["mha_num_attention_layers"],
                    mha_ff_dim_factor=config["mha_ff_dim_factor"],
                ).to(device)
            elif method_type in ["gazetoken", "persontoken"]:
                model = catoken_model(
                    method=method_type,
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoder=config["obs_encoder"],
                    encoding_size=config["encoding_size"],
                    mha_num_attention_heads=config["mha_num_attention_heads"],
                    mha_num_attention_layers=config["mha_num_attention_layers"],
                    mha_ff_dim_factor=config["mha_ff_dim_factor"],
                ).to(device)
            elif method_type in ["vint", "cnnaux", "gazeaux", "personaux"]:
                 model = vint_model(
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoder=config["obs_encoder"],
                    encoding_size=config["encoding_size"],
                    mha_num_attention_heads=config["mha_num_attention_heads"],
                    mha_num_attention_layers=config["mha_num_attention_layers"],
                    mha_ff_dim_factor=config["mha_ff_dim_factor"],
                ).to(device)
            elif method_type in ["gnm", "gnmgazeaux", "gnmpersonaux"]:
                model = gnm_model( 
                    method=method_type,
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoding_size=config["encoding_size"],
                ).to(device)
            elif method_type in ["gnmgazechannel", "gnmpersonchannel"]:
                model = gnmchannel_model( 
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoding_size=config["encoding_size"],
                ).to(device)
            else:
                raise ValueError(f"Unknown method type for act_model: {method_type}")
    
    # 确保模型不为空，以便为其创建优化器和调度器
    if model is None:
        raise ValueError("Model instance is None and no new model was created. Cannot create optimizer.")

    # 梯度裁剪
    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    # 优化器创建
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    #调度器创建
    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")
        
        if config["warmup"] and config["warmup_epochs"] > 0:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )
    return model, optimizer, scheduler


def main(config):
    torch.set_num_threads(4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True


    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    current_epoch = 0

    print(f"\n--- Starting for method: {config['method']} ---")

    if config["method"] in ["1phase", "1phaseplus", "gaze", "gazeplus"]:

        use_generated_attnmaps = False if config["method"] in ["1phase", "gaze"] else True
        train_gen_attnmaps_path = None if config ["method"] in ["1phase", "gaze"] else "/home/yzc/CRUEL/data_splits/train/attnmap_used.pt"
        test_gen_attnmaps_path = None if config ["method"] in ["1phase", "gaze"] else "/home/yzc/CRUEL/data_splits/test/attnmap_used.pt"

        # Create datasets for obs_model training
        train_dataset_obs = ConcatDataset([
            ObsDataset(
                data_folder=config["datasets"]["data"]["data_folder"],
                data_split_folder=config["datasets"]["data"]["train"],
                dataset_name="data",
                image_size=config["image_size"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                obs_type=config["obs_type"],
                use_generated_attnmaps=use_generated_attnmaps,
                generated_attnmaps_path=train_gen_attnmaps_path
            )
        ])
        test_dataset_obs = ConcatDataset([
            ObsDataset(
                data_folder=config["datasets"]["data"]["data_folder"],
                data_split_folder=config["datasets"]["data"]["test"],
                dataset_name="data",
                image_size=config["image_size"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                obs_type=config["obs_type"],
                use_generated_attnmaps=use_generated_attnmaps,
                generated_attnmaps_path=test_gen_attnmaps_path
            )
        ])

        # Determine the collate_fn for the action model based on its method type
        if config["method"] in ["1phase","1phaseplus"]: # These use person_collate_fn
            obs_collate_fn = obs_person_collate_fn
        else: # Default for gaze or gazeplus or 2phase or plus
            obs_collate_fn = obs_base_collate_fn

        train_loader_obs = DataLoader(
            train_dataset_obs,
            batch_size=config.get("batch_size_obs_model", config["batch_size"]), # Use specific, fall back to general
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=obs_collate_fn # WinnerSelector needs obs_person_collate_fn
        )
        test_loader_obs = DataLoader(
            test_dataset_obs,
            batch_size=config.get("batch_size_obs_model", config["batch_size"]), # Use specific, fall back to general
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=obs_collate_fn # WinnerSelector needs obs_person_collate_fn
        )
        train_loader_use = train_loader_obs
        test_loader_use = test_loader_obs
        
    else:
        if config["wg_origin"]=="GT":
            use_generated_labels = False
            train_gen_labels_path = None
            test_gen_labels_path = None
        else:
            use_generated_labels = True
            train_gen_labels_path = f'/home/yzc/CRUEL/data_splits/train/{config["wg_origin"]}.pt'
            test_gen_labels_path = f'/home/yzc/CRUEL/data_splits/test/{config["wg_origin"]}.pt'

        # Create datasets for act_model training
        train_dataset_act = ConcatDataset([
            ActDataset(
                data_folder=config["datasets"]["data"]["data_folder"],
                data_split_folder=config["datasets"]["data"]["train"],
                dataset_name="data",
                image_size=config["image_size"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                obs_type=config["obs_type"],
                use_generated_labels=use_generated_labels,
                generated_labels_path=train_gen_labels_path
            )
        ])
        test_dataset_act = ConcatDataset([
            ActDataset(
                data_folder=config["datasets"]["data"]["data_folder"],
                data_split_folder=config["datasets"]["data"]["test"],
                dataset_name="data",
                image_size=config["image_size"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                obs_type=config["obs_type"],
                use_generated_labels=use_generated_labels,
                generated_labels_path=test_gen_labels_path
            )
        ])

        # Determine the collate_fn for the action model based on its method type
        if config["method"] in ["gnmpersonaux", "gnmpersonchannel", "personaux", "personchannel", "persontoken"]: # These use person_collate_fn
            act_collate_fn = act_person_collate_fn
        else: # Default for vint, cnnaux, gazeaux, gazechannel, gazetoken
            act_collate_fn = act_base_collate_fn

        train_loader_act = DataLoader(
            train_dataset_act,
            batch_size=config.get("batch_size_act_model", config["batch_size"]), # Use specific, fall back to general
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=act_collate_fn
        )
        test_loader_act = DataLoader(
            test_dataset_act,
            batch_size=config.get("batch_size_act_model", config["batch_size"]), # Use specific, fall back to general
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=act_collate_fn
        )
        train_loader_use = train_loader_act
        test_loader_use = test_loader_act


    # Generate Attention Map from stage0 or gazes/winners from stage1
    if config["ifgenerate"] == True:

        if config ["method"] == "persontoken":

            act_model, _, _ = create_model_and_optimizer(
                "persontoken", config, device, 0.0, model_type="act_model"
            ) # lr设为0表示不训练
            
            #加载模型
            load_act_model_path = config["load_act_model_path"]
            if os.path.exists(load_act_model_path):
                print(f"Loading act_model weights from: {load_act_model_path}...")
                act_checkpoint = torch.load(load_act_model_path, map_location=device)
                load_model(act_model, act_checkpoint)
                print("Act_model weights loaded successfully.")
            else:
                raise FileNotFoundError(f"Error: Pre-trained act_model not found at {load_act_model_path}.")
            # 生成raw
            generate_attnmap(act_model, train_loader_use, device, config["save_train_raw_path"])
            generate_attnmap(act_model, test_loader_use, device, config["save_test_raw_path"])
            
        elif config["method"] in ["1phase", "1phaseplus"]:

            obs_model, _, _ = create_model_and_optimizer(
                config["method"], config, device, 0.0, model_type="obs_model"
            ) # lr设为0表示不训练
            
            #加载模型
            load_obs_model_path = config["load_obs_model_path"]
            if os.path.exists(load_obs_model_path):
                print(f"Loading obs_model weights from: {load_obs_model_path}...")
                obs_checkpoint = torch.load(load_obs_model_path, map_location=device)
                load_model(obs_model, obs_checkpoint)
                print("Obs_model weights loaded successfully.")
            else:
                raise FileNotFoundError(f"Error: Pre-trained obs_model not found at {load_obs_model_path}.")
            # 生成raw
            generate_1phase_winners(config["method"], obs_model, train_loader_use, device, config["save_train_raw_path"])
            generate_1phase_winners(config["method"], obs_model, test_loader_use, device, config["save_test_raw_path"])
        
        elif config["method"] in ["gaze", "gazeplus"]:

            obs_model, _, _ = create_model_and_optimizer(
                config["method"], config, device, 0.0, model_type="obs_model"
            ) # lr设为0表示不训练
            
            #加载模型
            load_obs_model_path = config["load_obs_model_path"]
            if os.path.exists(load_obs_model_path):
                print(f"Loading obs_model weights from: {load_obs_model_path}...")
                obs_checkpoint = torch.load(load_obs_model_path, map_location=device)
                load_model(obs_model, obs_checkpoint)
                print("Obs_model weights loaded successfully.")
            else:
                raise FileNotFoundError(f"Error: Pre-trained obs_model not found at {load_obs_model_path}.")
            # 生成raw
            generate_gaze(config["method"], obs_model, train_loader_use, device, config["save_train_raw_path"])
            generate_gaze(config["method"], obs_model, test_loader_use, device, config["save_test_raw_path"])

    # else train models
    else:
        # Initialize models
        if config["method"] in ["gaze", "gazeplus", "1phase", "1phaseplus"]:
            obs_model, optimizer, scheduler = create_model_and_optimizer(config["method"], config, device, float(config["obs_lr"]), model_type="obs_model")
            act_model = None  # Only obs_model is primary
            
            # 加载预训练模型用于评估
            if not config["iftrain"]:
                model_path = f"/home/yzc/CRUEL/data_splits/weights/best_{config['method']}.pt"
                if os.path.exists(model_path):
                    print(f"Loading pretrained obs_model from: {model_path}")
                    checkpoint = torch.load(model_path, map_location=device)
                    load_model(obs_model, checkpoint)
                    print("Pretrained obs_model loaded successfully.")
                else:
                    raise FileNotFoundError(f"Pretrained model not found at {model_path}")
            
            total_params = count_parameters(obs_model)
            print(f"Obs model total params: {total_params} ({total_params/1e6:.2f}M)")
            if config.get("use_wandb", False):
                wandb.log({"obs_total_params": total_params}, commit=False)

        else:
            act_model, optimizer, scheduler = create_model_and_optimizer(config["method"], config, device, float(config["act_lr"]), model_type="act_model")
            obs_model = None  # Only act_model is primary
            
            # 加载预训练模型用于评估
            if not config["iftrain"]:
                model_path = f"/home/yzc/CRUEL/data_splits/weights/best_{config['method']}.pt"
                if os.path.exists(model_path):
                    print(f"Loading pretrained act_model from: {model_path}")
                    checkpoint = torch.load(model_path, map_location=device)
                    load_model(act_model, checkpoint)
                    print("Pretrained act_model loaded successfully.")
                else:
                    raise FileNotFoundError(f"Pretrained model not found at {model_path}")
                
            total_params = count_parameters(act_model)
            print(f"Act model total params: {total_params} ({total_params/1e6:.2f}M)")
            if config.get("use_wandb", False):
                wandb.log({"act_total_params": total_params}, commit=False)

        train_eval_loop(
            train_method=config["method"],
            training_config=config,
            train_model=config["iftrain"],
            obs_model=obs_model if config["method"] in ["1phase", "1phaseplus", "gaze", "gazeplus"] else None,
            act_model=act_model if config["method"] not in ["1phase", "1phaseplus", "gaze", "gazeplus"] else None,
            optimizer=optimizer, # Use the single optimizer
            scheduler=scheduler, # Use the single scheduler
            train_loader=train_loader_use,
            test_loader=test_loader_use,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
        print(f"--- {config['method']} Finished ---")

        wandb.finish()


if __name__ == "__main__":
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["run_folder"] = os.path.join(
        "my_logs", config["project_name"], config["run_name"]
    )
    os.makedirs(config["run_folder"])

    if config["use_wandb"]:
        wandb.login()
        if "load_run" in config:
            # 取 load_run 路径的最后一部分作为 run id，避免斜杠
            run_id = config["run_id"]
            wandb.init(
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="polluxiaga-nanjing-university",
                resume="must",
                id=run_id,
            )
            wandb.config.update(config, allow_val_change=True)
        else:
            wandb.init(
                mode="offline",
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="polluxiaga-nanjing-university",
            )
            wandb.run.name = config["run_name"]
            wandb.config.update(config)

    main(config)