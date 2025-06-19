import os
from typing import Optional, Dict, List
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
from my_model.backbone import base_model, channel_model, catoken_model
from my_model.selectors import WinnerSelector, WinnerSelectorPlus
from my_training.my_train_utils import act_person_collate_fn, obs_person_collate_fn, act_base_collate_fn
from my_training.my_train_eval_loop import train_eval_loop, load_model


def generate_attnmap(
    model: nn.Module,
    dataloader: DataLoader, # This DataLoader MUST provide (obs_img, attention, ...) compatible with catoken_model
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Generate the RGB attention maps from the act_model (catoken_model) for supervision.
    The attention maps will be returned as a list of tensors, where each tensor corresponds
    to an unpadded sample from the original dataset and has shape (N * spatial_flatten_len).
    This represents the softmax-normalized attention distribution over only the RGB spatial tokens.

    Returns:
        A list of torch.Tensor, where each tensor is the (N * spatial_flatten_len)
        attention map for a single sample, ordered by its original dataset index.
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
        tqdm_iter = tqdm.tqdm(dataloader, desc="Generating act_model Attention Maps", dynamic_ncols=True)
        for batch_data in tqdm_iter:
            # Unpack the batched data, matching persontoken_train's data unpacking
            (
                obs_image,
                _, # Unused from dataloader (e.g., gaze_map)
                person_masks,
                select_labels,
                _, # Unused (e.g., action_label)
                invalid,
                original_indices_batch
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
    
    return all_unpadded_rgb_attention_maps


def generate_selector_predictions(
    model: nn.Module,
    dataloader: DataLoader, # This DataLoader MUST use obs_person_collate_fn
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Generate selector predictions for a single dataset split.
    The predictions will be binary masks similar to winner_labels,
    returned as a list of boolean tensors (unpadded, as per __getitem__ output).
    """
    model.eval() # Set model to evaluation mode for prediction
    
    total_samples = len(dataloader.dataset)
    all_unpadded_predictions = [None] * total_samples # Initialize list to store results in original dataset order
    
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(dataloader, desc="Generating Selector Predictions", dynamic_ncols=True)
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
                
    print(f"Generated {len(all_unpadded_predictions)} selector prediction samples in memory.")
    if len(all_unpadded_predictions) > 0:
        print(f"Example cached prediction shape (first sample): {all_unpadded_predictions[0].shape}")
    
    return all_unpadded_predictions


def create_model_and_optimizer(method_type, config, device, lr, model_type="act_model", model_instance=None):
    """Helper function to create a model and its optimizer/scheduler."""
    model = model_instance
    if model is None:
        if model_type == "obs_model":
            if method_type == "dumobs":
                model = WinnerSelector(
                    context_size=config["context_size"],
                ).to(device)
            elif method_type == "obs":
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
            elif method_type in ["base", "cnnaux", "gazeaux", "personaux"]:
                 model = base_model(
                    method=method_type,
                    context_size=config["context_size"],
                    len_traj_pred=config["len_traj_pred"],
                    encoder=config["obs_encoder"],
                    encoding_size=config["encoding_size"],
                    mha_num_attention_heads=config["mha_num_attention_heads"],
                    mha_num_attention_layers=config["mha_num_attention_layers"],
                    mha_ff_dim_factor=config["mha_ff_dim_factor"],
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

    # Initialize models for multi-stage training or single stage
    obs_model = None
    act_model = None

    # DataLoader preparation - datasets and collate_fns
    # For "phase" training, we need two sets of datasets:
    # 1. For Stage 1 (obs_model training): use_generated_labels=False
    # 2. For Stage 2/3 (act_model training): use_generated_labels=True, with generated predictions
    
    # Create datasets for obs_model training (Stage 1 of "phase", or if method is "dumobs" or "obs")
    train_dataset_obs = ConcatDataset([
        ObsDataset(
            data_folder=config["datasets"]["data"]["data_folder"],
            data_split_folder=config["datasets"]["data"]["train"],
            dataset_name="data",
            image_size=config["image_size"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            obs_type=config["obs_type"],
            use_generated_attnmaps=False, # Selector training uses GT only
            generated_attnmaps_path=None,
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
            use_generated_attnmaps=False, # Selector training uses GT only
            generated_attnmaps_path=None,
        )
    ])

    train_loader_obs = DataLoader(
        train_dataset_obs,
        batch_size=config.get("batch_size_obs_model", config["batch_size"]), # Use specific, fall back to general
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=obs_person_collate_fn # WinnerSelector needs obs_person_collate_fn
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
        collate_fn=obs_person_collate_fn # WinnerSelector needs obs_person_collate_fn
    )


    # For other methods, or Stage 2/3 of "phase", we might need different collate_fns
    train_dataset_act = ConcatDataset([
        ActDataset(
            data_folder=config["datasets"]["data"]["data_folder"],
            data_split_folder=config["datasets"]["data"]["train"],
            dataset_name="data",
            image_size=config["image_size"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            obs_type=config["obs_type"],
            use_generated_labels=False, # Default, will be overridden for phase Stage 2/3
            generated_labels_path=None, # Default, will be overridden for phase Stage 2/3
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
            use_generated_labels=False, # Default, will be overridden for phase Stage 2/3
            generated_labels_path=None, # Default, will be overridden for phase Stage 2/3
        )
    ])

    # Determine the collate_fn for the action model based on its method type
    if config["method"] in ["personaux", "personchannel", "persontoken", "phase"]: # These use person_collate_fn
        act_collate_fn = act_person_collate_fn
    else: # Default for base, cnnaux, gazeaux, gazechannel, gazetoken
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
    

    # Initialize models
    if config["method"] == "phase":
        act_model, act_optimizer, act_scheduler = create_model_and_optimizer("persontoken", config, device, float(config["lr"]), model_type="act_model") # Phase Stage 2/3 uses persontoken
        obs_model = None
    elif config["method"] == "dumobs" or config["method"] == "obs":
        obs_model, optimizer, scheduler = create_model_and_optimizer(config["method"], config, device, float(config["lr"]), model_type="obs_model")
        act_model = None # Only obs_model is primary
    else: # All other single-stage methods train an action model
        act_model, optimizer, scheduler = create_model_and_optimizer(config["method"], config, device, float(config["lr"]), model_type="act_model")
        obs_model = None # Only act_model is primary


    current_epoch = 0
    # Load checkpoint logic adapted for multi-model setup
    if "load_run" in config:
        load_run_folder = os.path.join("my_logs", config["load_run"])
        print("Loading model from ", load_run_folder)

        # Try to load obs_model if it exists and is needed
        if obs_model is not None:
            obs_best_path = os.path.join(load_run_folder, "best_obs_model.pth")
            if os.path.exists(obs_best_path):
                print(f"Loading obs_model from {obs_best_path}")
                obs_checkpoint = torch.load(obs_best_path, map_location=device)
                load_model(obs_model, obs_checkpoint)
                if obs_optimizer and "optimizer_state_dict" in obs_checkpoint:
                    obs_optimizer.load_state_dict(obs_checkpoint["optimizer_state_dict"])
                if obs_scheduler and "scheduler_state_dict" in obs_checkpoint:
                    obs_scheduler.load_state_dict(obs_checkpoint["scheduler_state_dict"])
                # Note: For multi-stage, current_epoch might be better managed by the stage logic itself
                # For simplicity here, we'll let the loop in train_eval_loop manage its own current_epoch
                # current_epoch = max(current_epoch, obs_checkpoint.get("epoch", 0) + 1) # Keep track of max epoch to resume from

        # Try to load act_model if it exists and is needed
        if act_model is not None:
            act_best_path = os.path.join(load_run_folder, "best_act.pth")
            if os.path.exists(act_best_path):
                print(f"Loading act_model from {act_best_path}")
                act_checkpoint = torch.load(act_best_path, map_location=device)
                load_model(act_model, act_checkpoint)
                if act_optimizer and "optimizer_state_dict" in act_checkpoint:
                    act_optimizer.load_state_dict(act_checkpoint["optimizer_state_dict"])
                if act_scheduler and "scheduler_state_dict" in act_checkpoint:
                    act_scheduler.load_state_dict(act_checkpoint["scheduler_state_dict"])
                # current_epoch = max(current_epoch, act_checkpoint.get("epoch", 0) + 1) # Keep track of max epoch to resume from


    # Stage training logic
    if config["method"] == "phase":
        stage0_epochs = config["stage0_epochs"]
        stage1_epochs = config["stage0_epochs"] + config["stage1_epochs"] 
        stage2_epochs = config["stage0_epochs"] + config["stage1_epochs"] + config["stage2_epochs"]
        # Stage 0: Train act_model Using GT winners
        print("\n--- Starting Stage 0: Training Action Model (act_model) with GT winners")
        stage0_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 100),
            "min_delta": config.get("min_delta", 1e-4)
        }
        train_eval_loop(
           train_method="persontoken", # Explicitly train persontoken model
            training_config=stage0_training_config,
            train_model=config["train"],
            obs_model=None, # obs_model is not active as primary here, but its predictions are used via DataLoader
            act_model=act_model, # Pass act_model as the primary target
            optimizer=act_optimizer,
            scheduler=act_scheduler,
            train_loader=train_loader_act, # Use action model loaders, now updated with generated labels
            test_loader=test_loader_act,
            transform=transform,
            epochs=stage0_epochs,
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=0,  # Stage 0 从0开始
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
        print("--- Stage 0 Finished ---") 

        # After Stage 0, generate attention_maps with the trained act_model
        print("\n--- Generating Attention Maps for Stage 1 ---")
        train_attention_maps = generate_attnmap(act_model, train_loader_act, device)
        test_attention_maps = generate_attnmap(act_model, test_loader_act, device)

        for dataset in train_dataset_obs.datasets: # Iterate through individual datasets if ConcatDataset
            dataset.use_generated_attnmaps = True
            dataset.set_generated_attention_maps(train_attention_maps)
        for dataset in test_dataset_obs.datasets:
            dataset.use_generated_attnmaps = True
            dataset.set_generated_attention_maps(test_attention_maps)
        
        print("Generated attention maps have been loaded into dataset for Stage 1.")

        # Stage 1: Train WinnerSelector (obs_model)
        obs_model, obs_optimizer, obs_scheduler = create_model_and_optimizer("obs", config, device, float(config["lr"]), model_type="obs_model")
        print("\n--- Starting Stage 1: Training WinnerSelector (obs_model) ---")
        stage1_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 20),
            "min_delta": config.get("min_delta", 1e-4)
        }
        train_eval_loop(
            train_method="obs", # Explicitly train obs model
            training_config=stage1_training_config,
            train_model=config["train"],
            obs_model=obs_model, # Pass obs_model as the primary target
            act_model=None, # act_model is not active in this stage
            optimizer=obs_optimizer,
            scheduler=obs_scheduler,
            train_loader=train_loader_obs,
            test_loader=test_loader_obs,
            transform=transform,
            epochs=stage1_epochs, # Use stage-specific epochs
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=stage0_epochs,  # 从 stage0 结束的轮次开始
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
        print("--- Stage 1 Finished ---")

        # After Stage 1, generate predictions with the trained obs_model
        print("\n--- Generating Selector Predictions for Stage 2 ---")
        train_predicted_labels = generate_selector_predictions(obs_model, train_loader_obs, device)
        test_predicted_labels = generate_selector_predictions(obs_model, test_loader_obs, device)
        
        # Assign generated labels to the act_model's datasets
        # This assumes gaze_dataset has a method to set generated labels
        # This will modify the underlying datasets used by train_loader_act and test_loader_act
        for dataset in train_dataset_act.datasets: # Iterate through individual datasets if ConcatDataset
            dataset.use_generated_labels = True
            dataset.set_generated_labels(train_predicted_labels)
        for dataset in test_dataset_act.datasets:
            dataset.use_generated_labels = True
            dataset.set_generated_labels(test_predicted_labels)

        print("Generated selector predictions have been loaded into datasets for Stage 2.")

        # Re-initialize optimizer and scheduler for the action model (optional, but good practice for distinct stages)
        # Assuming we want a fresh start for act_model training
        _ , act_optimizer, act_scheduler = create_model_and_optimizer(
            "persontoken", config, device, float(config["lr"]), 
            model_type="act_model", 
            model_instance=act_model
        )
        print("\n--- Starting Stage 2: Training Action Model (act_model) with Selector Predictions ---")
        stage2_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 100),
            "min_delta": config.get("min_delta", 1e-4)
        }
        stage1_epochs = config.get("stage1_epochs", 50)
        train_eval_loop(
            train_method="persontoken", # Explicitly train persontoken model
            training_config=stage2_training_config,
            train_model=config["train"],
            obs_model=None, # obs_model is not active as primary here, but its predictions are used via DataLoader
            act_model=act_model, # Pass act_model as the primary target
            optimizer=act_optimizer,
            scheduler=act_scheduler,
            train_loader=train_loader_act, # Use action model loaders, now updated with generated labels
            test_loader=test_loader_act,
            transform=transform,
            epochs=stage2_epochs,
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=stage1_epochs,  # 从 stage0 + stage1 结束的轮次开始
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
        print("--- Stage 2 Finished ---")
        
        # If there's a Stage 3, you would repeat a similar pattern here:
        # Re-initialize optimizer/scheduler for stage 3 if needed
        # Call train_eval_loop with appropriate parameters for Stage 3

    else: # For all non-"phase" methods, single-stage training
        print(f"\n--- Starting Single Stage Training for method: {config['method']} ---")
        single_stage_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 200),
            "min_delta": config.get("min_delta", 1e-4)
        }
        
        # Determine which model is the primary for this single-stage run
        primary_train_method = config["method"]
        train_loader_use = train_loader_obs if config["method"] == "dumobs" or config["method"] == "obs" else train_loader_act
        test_loader_use = test_loader_obs if config["method"] == "dumobs" or config["method"] == "obs" else test_loader_act
        
        train_eval_loop(
            train_method=primary_train_method,
            training_config=single_stage_training_config,
            train_model=config["train"],
            obs_model=obs_model if config["method"] == "dumobs" or config ["method"] == "obs" else None, # Pass actual obs_model if needed
            act_model=act_model if config["method"] != "dumobs" or config ["method"] == "obs" else None, # Pass actual act_model if needed
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
        print(f"--- Single Stage Training for {config['method']} Finished ---")


    print("FINISHED TRAINING")
    wandb.finish()


if __name__ == "__main__":

    # 如果没有指定 load_run，则更新 run_name 和 run_folder，否则使用旧的
    if "load_run" not in config:
        config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
        config["run_folder"] = os.path.join(
            "my_logs", config["project_name"], config["run_name"]
        )
        os.makedirs(config["run_folder"])  # 新建目录
    else:
        # 使用 load_run 对应的文件夹
        # 假设 config["load_run"] 的格式为 "my_vint/vint_2025_02_21_00_19_42"
        config["run_folder"] = os.path.join("my_logs", config["load_run"])
        print("Continuing from:", config["run_folder"])
        # 此时不要创建新文件夹

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
                #mode="offline",
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="polluxiaga-nanjing-university",
            )
            wandb.run.name = config["run_name"]
            wandb.config.update(config)

    print(config)
    main(config)