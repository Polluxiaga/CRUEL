import os
from typing import Optional, Dict, List
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

from my_data.my_dataset import gaze_dataset
from my_model.backbone import base_model, channel_model, catoken_model
from my_model.selectors import WinnerSelector
from my_training.my_train_utils import person_collate_fn, base_collate_fn
from my_training.my_train_eval_loop import train_eval_loop, load_model


def generate_selector_predictions(
    model: nn.Module,
    dataloader: DataLoader, # This DataLoader MUST use person_collate_fn
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
            # Unpack the batched data from person_collate_fn
            obs_batch, _, mask_batch, _, _, invalid_flag_batch, original_indices_batch = batch_data
            
            # Move tensors to device
            obs_batch = obs_batch.to(device)
            mask_batch = mask_batch.to(device)
            invalid_flag_batch = invalid_flag_batch.to(device)
            original_indices_list = original_indices_batch.cpu().tolist() # Get original dataset indices

            logits = model(obs_batch, mask_batch, invalid_flag_batch) # (B, P_max)
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


def create_model_and_optimizer(method_type, config, device, lr, model_type="act_model"):
    """Helper function to create a model and its optimizer/scheduler."""
    model = None
    if model_type == "obs_model":
        model = WinnerSelector(
            context_size=config["context_size"],
        ).to(device)
    elif model_type == "act_model":
        if method_type == "gazechannel" or method_type == "personchannel":
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
        elif method_type == "gazetoken" or method_type == "persontoken" or method_type == "phase": # "phase" uses catoken_model in stage 2/3
            model = catoken_model(
                method=method_type, # Pass "persontoken" or "gazetoken" if that's the underlying
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                encoder=config["obs_encoder"],
                encoding_size=config["encoding_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            ).to(device)
        elif method_type == "base" or method_type == "cnnaux" or method_type == "gazeaux" or method_type == "personaux":
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

    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

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
    
    # Create datasets for obs_model training (Stage 1 of "phase", or if method is "obs")
    train_dataset_obs = ConcatDataset([
        gaze_dataset(
            data_folder=config["datasets"]["data"]["data_folder"],
            data_split_folder=config["datasets"]["data"]["train"],
            dataset_name="data",
            image_size=config["image_size"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            obs_type=config["obs_type"],
            use_generated_labels=False, # Selector training uses GT only
            generated_labels_path=None,
        )
    ])
    test_dataset_obs = ConcatDataset([
        gaze_dataset(
            data_folder=config["datasets"]["data"]["data_folder"],
            data_split_folder=config["datasets"]["data"]["test"],
            dataset_name="data",
            image_size=config["image_size"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            obs_type=config["obs_type"],
            use_generated_labels=False, # Selector training uses GT only
            generated_labels_path=None,
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
        collate_fn=person_collate_fn # WinnerSelector needs person_collate_fn
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
        collate_fn=person_collate_fn # WinnerSelector needs person_collate_fn
    )

    # For other methods, or Stage 2/3 of "phase", we might need different collate_fns
    train_dataset_act = ConcatDataset([
        gaze_dataset(
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
        gaze_dataset(
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
        act_collate_fn = person_collate_fn
    else: # Default for base, cnnaux, gazeaux, gazechannel, gazetoken
        act_collate_fn = base_collate_fn

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
        obs_model, obs_optimizer, obs_scheduler = create_model_and_optimizer("obs", config, device, float(config["lr"]), model_type="obs_model")
        act_model, act_optimizer, act_scheduler = create_model_and_optimizer("persontoken", config, device, float(config["lr"]), model_type="act_model") # Phase Stage 2/3 uses persontoken
    elif config["method"] == "obs":
        obs_model, optimizer, scheduler = create_model_and_optimizer("obs", config, device, float(config["lr"]), model_type="obs_model")
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
        # Stage 1: Train WinnerSelector (obs_model)
        print("\n--- Starting Stage 1: Training WinnerSelector (obs_model) ---")
        stage1_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 10),
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
            epochs=config.get("stage1_epochs", 50), # Use stage-specific epochs
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch, # Reset current_epoch for each stage if desired, or let train_eval_loop manage
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
        act_model, act_optimizer, act_scheduler = create_model_and_optimizer("persontoken", config, device, float(config["lr"]), model_type="act_model")
        print("\n--- Starting Stage 2: Training Action Model (act_model) with Selector Predictions ---")
        stage2_training_config = {
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 200),
            "min_delta": config.get("min_delta", 1e-4)
        }
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
            epochs=config.get("stage2_epochs", 30) + config.get("stage1_epochs", 50), # Total epochs for consistency, or just stage2_epochs
            device=device,
            run_folder=config["run_folder"],
            wandb_log_freq=config["wandb_log_freq"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=config.get("stage1_epochs", 50), # Start epoch count from end of stage 1
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
        model_to_train = obs_model if config["method"] == "obs" else act_model
        primary_train_method = config["method"]
        train_loader_use = train_loader_obs if config["method"] == "obs" else train_loader_act
        test_loader_use = test_loader_obs if config["method"] == "obs" else test_loader_act
        
        train_eval_loop(
            train_method=primary_train_method,
            training_config=single_stage_training_config,
            train_model=config["train"],
            obs_model=obs_model if config["method"] == "obs" else None, # Pass actual obs_model if needed
            act_model=act_model if config["method"] != "obs" else None, # Pass actual act_model if needed
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
                mode="offline",
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="polluxiaga-nanjing-university",
            )
            wandb.run.name = config["run_name"]
            wandb.config.update(config)

    print(config)
    main(config)