import os
import argparse
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
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from my_data.my_dataset import gaze_dataset
from my_model.base import base_model
from my_training.my_train_utils import person_collate_fn, base_collate_fn
from my_training.my_train_eval_loop import train_eval_loop, load_model


def main(config):

    torch.set_num_threads(8)
    
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


    # Load the data
    train_dataset = []
    test_dataset = []
    method = config["method"]
    data_config = config["datasets"]["data"]
    for data_split_type in ["train", "test"]:
        dataset = gaze_dataset(
            data_folder=data_config["data_folder"],
            data_split_folder=data_config[data_split_type],
            dataset_name="data",
            image_size=config["image_size"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            obs_type=config["obs_type"],
        )
        if data_split_type == "train":
            train_dataset.append(dataset)
        else:
            test_dataset.append(dataset)


    train_dataset = ConcatDataset(train_dataset)
    if method == "sel" or method == "personaux":
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=person_collate_fn  # Use person_collate_fn for sel model
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=base_collate_fn,
        )

    test_dataset = ConcatDataset(test_dataset)
    if method == "sel" or method == "personaux":
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=person_collate_fn  # Use person_collate_fn for sel model
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=base_collate_fn,
        )


    # Create the model
    model = base_model(
        method=method,
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        encoder=config["obs_encoder"],
        encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    

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


    lr = float(config["lr"])
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
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")
        
        if config["warmup"] and config["warmup_epochs"] > 0:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )


    current_epoch = 0
    if "load_run" in config:
        load_run_folder = os.path.join("my_logs", config["load_run"])
        print("Loading model from ", load_run_folder)
        latest_path = os.path.join(load_run_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1


    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())


    # Add stage training config if method is "sel"
    if config["method"] == "sel":
        training_config = {
            "enable_stage_training": True,
            "current_stage": 1,
            "stage1_epochs": config.get("stage1_epochs", 50),
            "stage2_epochs": config.get("stage2_epochs", 30),
            "stage3_epochs": config.get("stage3_epochs", 50),
            "stage1_loss_threshold": config.get("stage1_loss_threshold", 0.1),
            "stage2_loss_threshold": config.get("stage2_loss_threshold", 0.1),
            "stage3_loss_threshold": config.get("stage3_loss_threshold", 0.1),
            # 早停相关参数
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 10),
            "min_delta": config.get("min_delta", 1e-4)
        }
    else:
        training_config = {
            "enable_stage_training": False,
            "early_stopping": config.get("early_stopping", True),
            "patience": config.get("patience", 10),
            "min_delta": config.get("min_delta", 1e-4)
        }

    # Single train_eval_loop call for all methods
    train_eval_loop(
        train_method=config["method"],
        training_config=training_config,
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
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
                project=config["project_name"],
                settings=wandb.Settings(start_method="fork"),
                entity="polluxiaga-nanjing-university",
            )
            wandb.run.name = config["run_name"]
            wandb.config.update(config)

    print(config)
    main(config)