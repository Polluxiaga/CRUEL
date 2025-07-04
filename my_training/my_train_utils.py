import wandb
import numpy as np
import os
import tqdm
import itertools
from sklearn.metrics import precision_recall_curve, auc
from typing import List, Dict, Optional, TextIO
from scipy.spatial.distance import directed_hausdorff
from frechetdist import frdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

from my_data.my_data_utils import ts2np
from my_training.my_visualize_utils import action_visualize, obsp_visualize, gaze_visualize


def obs_base_collate_fn(batch):
    obs_images, fixations, mask_list, attnmap_list, _, original_idx_list = zip(*batch)

    obs_batch = torch.stack(obs_images, dim=0)
    fixations_batch = torch.stack(fixations, dim=0)
    attnmap_batch = torch.stack(attnmap_list, dim=0) # [B, N*spatial_flatten_len]
    original_idx_batch = torch.stack(original_idx_list, dim=0)

    P_list = [m.shape[0] for m in mask_list]
    P_max = max(P_list)

    mask_padded = []
    for masks in mask_list:
        P = masks.shape[0]
        pad = P_max - P

        pad_masks = torch.zeros(pad, *masks.shape[1:], dtype=masks.dtype)
        mask_padded.append(torch.cat([masks, pad_masks], dim=0))

    mask_batch = torch.stack(mask_padded, dim=0)

    return obs_batch, fixations_batch, mask_batch, attnmap_batch, original_idx_batch
    

def act_base_collate_fn(batch):
    """
    Basic collate function that returns only necessary elements for base model:
    - obs_images: observation images
    - gaze_maps: gaze attention maps
    - action_list: action labels
    """
    obs_images, gaze_maps, _, _, action_list, original_idx_list = zip(*batch)
    
    obs_batch = torch.stack(obs_images, dim=0)
    gazemap_batch = torch.stack(gaze_maps, dim=0)
    action_batch = torch.stack(action_list, dim=0)
    original_idx_batch = torch.stack(original_idx_list, dim=0)
    
    return obs_batch, gazemap_batch, action_batch, original_idx_batch


def obs_person_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
      obs_image:  Tensor [3*N, H, W]
      mask_imgs:  Tensor [P_i, N, H, W]
      select_labels: Tensor [P_i]
      action_labels: Tensor [3, 2]
    """

    # 1) 解包
    obs_images, _, mask_list, attnmap_list, select_list, original_idx_list = zip(*batch)
    B = len(batch)

    # 2) Stack obs_images and attnmap
    obs_batch = torch.stack(obs_images, dim=0)  # [B, 3*(C+1), H, W]
    attnmap_batch = torch.stack(attnmap_list, dim=0) # [B, N*spatial_flatten_len]
    original_idx_batch = torch.stack(original_idx_list, dim=0)

    # 3) 找到 P_max
    P_list = [m.shape[0] for m in mask_list]
    P_max = max(P_list)

    # 4) pad mask 和 select_label，并生成 invalid
    mask_padded, select_padded, invalid = [], [], []
    for masks, labels in zip(mask_list, select_list):
        P = masks.shape[0]
        pad = P_max - P

        # masks: (P, N, H, W) + (pad, N, H, W)
        pad_masks = torch.zeros(pad, *masks.shape[1:], dtype=masks.dtype)
        mask_padded.append(torch.cat([masks, pad_masks], dim=0))

        # labels: (P,) + (pad,)
        pad_labels = torch.zeros(pad, dtype=labels.dtype)
        select_padded.append(torch.cat([labels, pad_labels], dim=0))

        # invalid flag: False for real, True for pad
        invalid.append(torch.tensor([False]*P + [True]*pad, dtype=torch.bool))

    mask_batch = torch.stack(mask_padded, dim=0)      # [B, P_max, N, H, W]
    select_batch = torch.stack(select_padded, dim=0)  # [B, P_max]
    invalid_flag = torch.stack(invalid, dim=0)     # [B, P_max]

    return obs_batch, mask_batch, attnmap_batch, select_batch, invalid_flag, original_idx_batch


def act_person_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
      obs_image:  Tensor [3*N, H, W]
      mask_imgs:  Tensor [P_i, N, H, W]
      select_labels: Tensor [P_i]
      action_labels: Tensor [3, 2]
    """

    # 1) 解包
    obs_images, _, mask_list, select_list, action_list, original_idx_list = zip(*batch)
    B = len(batch)

    # 2) Stack obs_images and action_labels
    obs_batch = torch.stack(obs_images, dim=0)  # [B, 3*(C+1), H, W]
    action_batch = torch.stack(action_list, dim=0)    # [B, 3, 2]
    original_idx_batch = torch.stack(original_idx_list, dim=0)

    # 3) 找到 P_max
    P_list = [m.shape[0] for m in mask_list]
    P_max = max(P_list)

    # 4) pad mask 和 select_label，并生成 invalid
    mask_padded, select_padded, invalid = [], [], []
    for masks, labels in zip(mask_list, select_list):
        P = masks.shape[0]
        pad = P_max - P

        # masks: (P, N, H, W) + (pad, N, H, W)
        pad_masks = torch.zeros(pad, *masks.shape[1:], dtype=masks.dtype)
        mask_padded.append(torch.cat([masks, pad_masks], dim=0))

        # labels: (P,) + (pad,)
        pad_labels = torch.zeros(pad, dtype=labels.dtype)
        select_padded.append(torch.cat([labels, pad_labels], dim=0))

        # invalid flag: False for real, True for pad
        invalid.append(torch.tensor([False]*P + [True]*pad, dtype=torch.bool))

    mask_batch = torch.stack(mask_padded, dim=0)      # [B, P_max, N, H, W]
    select_batch = torch.stack(select_padded, dim=0)  # [B, P_max]
    invalid_flag = torch.stack(invalid, dim=0)     # [B, P_max]

    return obs_batch, mask_batch, select_batch, action_batch, invalid_flag, original_idx_batch


class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        latest = round(self.latest(), self.rounding)
        average = round(self.average(), self.rounding)
        moving_average = round(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output

    def log_data(self, data: float):
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        return f"{self.name} ({self.dataset})"

    def latest(self) -> float:
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()


def gaze_log(
    i,
    epoch,
    num_batches,
    run_folder,
    num_images_log,
    loggers,
    obs_images,
    pred_fixations,
    attention_scores,
    gt_fixations,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        gaze_visualize(
            batch_obs_images=ts2np(obs_images),
            batch_pred_fixation=ts2np(pred_fixations),
            batch_label_fixations=ts2np(gt_fixations),
            attention_scores=ts2np(attention_scores),
            mode=mode,
            save_folder=run_folder,
            epoch=epoch,
            num_images_log=num_images_log,
            use_wandb=use_wandb,
        )


def obsp_log(
    i,
    epoch,
    num_batches,
    run_folder,
    num_images_log,
    loggers,
    obs_images,
    winner_masks,
    obs_pred,
    obs_label,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        obsp_visualize(
            batch_obs_images=ts2np(obs_images),
            batch_winner_masks=ts2np(winner_masks),
            batch_pred_select=ts2np(obs_pred),
            batch_label_select=ts2np(obs_label),
            mode=mode,
            save_folder=run_folder,
            epoch=epoch,
            num_images_log=num_images_log,
            use_wandb=use_wandb,
        )


def action_log(
    i,
    epoch,
    num_batches,
    run_folder,
    num_images_log,
    loggers,
    obs_images,
    action_pred,
    attention_scores,
    action_label,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        action_visualize(
            batch_obs_images=ts2np(obs_images),
            batch_pred_waypoints=ts2np(action_pred),
            batch_label_waypoints=ts2np(action_label),
            attention_scores=ts2np(attention_scores),
            mode=mode,
            save_folder=run_folder,
            epoch=epoch,
            num_images_log=num_images_log,
            use_wandb=use_wandb,
        )


def gnm_log(
    i,
    epoch,
    num_batches,
    loggers,
    use_latest,
    print_log_freq=1,
    log_file: Optional[TextIO]=None,
):
    """
    Log data to wandb and print to console.
    """
    if not use_latest:
        for key, logger in loggers.items():
            # <--- 明确调用 logger.average() 获取平均值，并手动构建简洁的总结消息 ---
            logged_average = round(logger.average(), logger.rounding) # 获取平均值并四舍五入
            log_message = f"(epoch {epoch} summary) {logger.full_name()} average: {logged_average:.4f}"
            print(log_message)
            if log_file:
                log_file.write(log_message + "\n")
                log_file.flush()
        return # 打印完所有平均值后，直接退出

    # 批次最新值日志模式 (use_latest=True)：按频率打印详细信息
    if i % print_log_freq != 0 or print_log_freq == 0:
        return # 不满足频率条件，跳过此批次的日志

    for key, logger in loggers.items():
        # <--- 调用 logger.display() 获取详细的最新值、移动平均、总平均信息 ---
        log_message = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
        print(log_message)
        if log_file:
            log_file.write(log_message + "\n")
            log_file.flush()


def compute_baseloss(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
):
    """
    Compute losses for action prediction.

    """
    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return (unreduced_loss).mean()

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    
    # MSE Loss
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # cosine similarity
    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

    # Final Displacement Error (FDE)
    fde = F.pairwise_distance(
        action_pred[:, -1], 
        action_label[:, -1], 
        p=2
    ).mean()

    # Fréchet Distance & Hausdorff Distance
    frechet_dists = []
    hausdorff_dists = []

    for i in range(action_pred.shape[0]):
        pred_np = action_pred[i].detach().cpu().numpy()
        label_np = action_label[i].detach().cpu().numpy()

        # 真实 Fréchet 距离（考虑点顺序）
        frechet = frdist(pred_np, label_np)
        frechet_dists.append(frechet)

        # 对称 Hausdorff 距离
        h1 = directed_hausdorff(pred_np, label_np)[0]
        h2 = directed_hausdorff(label_np, pred_np)[0]
        hausdorff = max(h1, h2)
        hausdorff_dists.append(hausdorff)

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similarity,
        "fde": torch.tensor(fde.item(), device=action_label.device),
        "frechet": torch.tensor(np.mean(frechet_dists), device=action_label.device),
        "hausdorff": torch.tensor(np.mean(hausdorff_dists), device=action_label.device),
    }

    return results


def gnm_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            _, # [batch_size, context_size+1, H, W] 
            action_label,
            _
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        action_label = action_label.to(device)

        optimizer.zero_grad()
    
        with autocast():
            action_pred = model(obs_image)
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

    gnm_log(
        i=0,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=1,
        log_file=None, # 传递文件句柄
    )
        

def gnm_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    log_file_path = os.path.join(run_folder, "evaluation_log.log") # 评估日志文件名
    os.makedirs(run_folder, exist_ok=True) # 确保 run_folder 存在

    with open(log_file_path, 'a', encoding='utf-8') as f_log:

        with torch.no_grad():
            for i, data in enumerate(tqdm_iter):
                obs_image, _, action_label, _ = data

                obs_images = torch.split(obs_image, 3, dim=1)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                action_label = action_label.to(device)

                # 前向推理
                action_pred = model(obs_image)

                # 计算损失并记录（注意：此处直接用 .item() 记录数值）
                losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())
        
        summary_message_start = f"\n--- Epoch {epoch} ---\n"
        print(summary_message_start)
        f_log.write(summary_message_start)
        f_log.flush()

        gnm_log(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=10, # 在评估时，可以设置一个合理的打印频率
        log_file=f_log, # 传递文件句柄
    )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def vint_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            _, # [batch_size, context_size+1, H, W] 
            action_label,
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3]) # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        action_label = action_label.to(device)

        optimizer.zero_grad()
      
        with autocast():
            action_pred, attention_scores = model(obs_image)
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def vint_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, _, action_label, _ = data

            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def compute_cnnaux_loss(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    gaze_map: torch.Tensor,
    gaze_use_map: torch.Tensor,
):
    """
    Compute KL loss for cnn feature maps and gaze map.
    """
    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return (unreduced_loss).mean()

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"

    # MSE Loss
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # cosine similarity
    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

    # Final Displacement Error (FDE)
    fde = F.pairwise_distance(
        action_pred[:, -1], 
        action_label[:, -1], 
        p=2
    ).mean()

    # Fréchet Distance & Hausdorff Distance
    frechet_dists = []
    hausdorff_dists = []

    for i in range(action_pred.shape[0]):
        pred_np = action_pred[i].detach().cpu().numpy()
        label_np = action_label[i].detach().cpu().numpy()

        # 真实 Fréchet 距离（考虑点顺序）
        frechet = frdist(pred_np, label_np)
        frechet_dists.append(frechet)

        # 对称 Hausdorff 距离
        h1 = directed_hausdorff(pred_np, label_np)[0]
        h2 = directed_hausdorff(label_np, pred_np)[0]
        hausdorff = max(h1, h2)
        hausdorff_dists.append(hausdorff)

    assert gaze_use_map.shape == gaze_map.shape, \
        f"gaze_use_map shape {gaze_use_map.shape} != gaze_map shape {gaze_map.shape}"

    # Gaze attention auxiliary loss (KL divergence)
    auxiliary_loss = F.kl_div(
        F.log_softmax(gaze_use_map, dim=1),
        F.softmax(gaze_map, dim=1),
        reduction='batchmean',
        log_target=False,
    )

    # Combine losses with weight
    alpha = 0.5
    total_loss = (1 - alpha) * action_loss + alpha * auxiliary_loss

    results = {
        "action_loss": action_loss,
        "auxiliary_loss": auxiliary_loss,
        "total_loss": total_loss,
        "action_waypts_cos_sim": action_waypts_cos_similarity,
        "fde": torch.tensor(fde.item(), device=action_label.device),
        "frechet": torch.tensor(np.mean(frechet_dists), device=action_label.device),
        "hausdorff": torch.tensor(np.mean(hausdorff_dists), device=action_label.device),
    }
    return results


def gnmgazeaux_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    auxiliary_loss_logger = Logger("auxiliary_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            gaze_attention, # [batch_size, (context_size+1) * H/32 * W/32] 
            action_label,
            _
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device)
        gaze_attention = gaze_attention.squeeze(2)
        output_h = gaze_attention.shape[2] // 32
        output_w = gaze_attention.shape[3] // 32
        gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
        gaze_attention_flattened = gaze_attention_pooled.view(gaze_attention_pooled.shape[0], -1)

        action_label = action_label.to(device)

        optimizer.zero_grad()
        
        with autocast():
            action_pred, gaze_use_map = model(obs_image)
            losses = compute_cnnaux_loss(
                action_label=action_label, 
                action_pred=action_pred, 
                gaze_map=gaze_attention_flattened, 
                gaze_use_map=gaze_use_map
                )
            loss = losses["total_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

    gnm_log(
        i=0,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=1,
        log_file=None, # 传递文件句柄
    )


def gnmgazeaux_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    log_file_path = os.path.join(run_folder, "evaluation_log.log") # 评估日志文件名
    os.makedirs(run_folder, exist_ok=True) # 确保 run_folder 存在

    with open(log_file_path, 'a', encoding='utf-8') as f_log:

        with torch.no_grad():
            for i, data in enumerate(tqdm_iter):
                obs_image, gaze_attention, action_label, _ = data

                obs_images = torch.split(obs_image, 3, dim=1)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                gaze_attention = gaze_attention.to(device)
                gaze_attention = gaze_attention.squeeze(2)
                output_h = gaze_attention.shape[2] // 32
                output_w = gaze_attention.shape[3] // 32
                gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
                gaze_attention_flattened = gaze_attention_pooled.view(gaze_attention_pooled.shape[0], -1)

                action_label = action_label.to(device)

                # 前向推理
                action_pred, gaze_use_map = model(obs_image)

                # 计算损失并记录（注意：此处直接用 .item() 记录数值）
                losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention_flattened, gaze_use_map=gaze_use_map)
                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

        summary_message_start = f"\n--- Epoch {epoch} ---\n"
        print(summary_message_start)
        f_log.write(summary_message_start)
        f_log.flush()

        gnm_log(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=10, # 在评估时，可以设置一个合理的打印频率
        log_file=f_log, # 传递文件句柄
    )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def gnmpersonaux_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    auxiliary_loss_logger = Logger("auxiliary_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
            _
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        # Convert person_masks to person_attention by taking union along num_persons dimension
        person_masks = person_masks.to(device)
        select_labels = select_labels.to(device)
        invalid = invalid.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~invalid  # [B, P]
        select_mask = (select_labels == 1) & valid_masks  # [B, P]
        select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

        person_masks = person_masks * select_mask.float()  # Zero out invalid masks
        person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]
        
        # Pool person_attention to match output size
        output_h = person_attention.shape[2] // 32
        output_w = person_attention.shape[3] // 32
        person_attention_pooled = F.adaptive_avg_pool2d(person_attention, (output_h, output_w))
        
        person_attention_flattened = person_attention_pooled.contiguous().view(person_attention_pooled.shape[0], -1)
        max_val = person_attention_flattened.max()
        if max_val > 0:
            gaze_map_normalized = person_attention_flattened / max_val
        else:
            gaze_map_normalized = person_attention_flattened # If all zeros, keep it as is

        action_label = action_label.to(device)

        optimizer.zero_grad()
        
        with autocast():
            action_pred, gaze_use_map = model(obs_image)
            losses = compute_cnnaux_loss(
                action_label=action_label, 
                action_pred=action_pred, 
                gaze_map=gaze_map_normalized, 
                gaze_use_map=gaze_use_map
                )
            loss = losses["total_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

    gnm_log(
        i=0,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=1,
        log_file=None, # 传递文件句柄
    )


def gnmpersonaux_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    log_file_path = os.path.join(run_folder, "evaluation_log.log") # 评估日志文件名
    os.makedirs(run_folder, exist_ok=True) # 确保 run_folder 存在

    with open(log_file_path, 'a', encoding='utf-8') as f_log:

        with torch.no_grad():
            for i, data in enumerate(tqdm_iter):
                obs_image, person_masks, select_labels, action_label, invalid, _ = data

                obs_images = torch.split(obs_image, 3, dim=1)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                # Convert person_masks to person_attention by taking union along num_persons dimension
                person_masks = person_masks.to(device)
                select_labels = select_labels.to(device)
                invalid = invalid.to(device)
                # Create selection mask using select_labels and invalid flag
                valid_masks = ~invalid  # [B, P]
                select_mask = (select_labels == 1) & valid_masks  # [B, P]
                select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

                person_masks = person_masks * select_mask.float()  # Zero out invalid masks
                person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]
            
                # Pool person_attention to match output size
                output_h = person_attention.shape[2] // 32
                output_w = person_attention.shape[3] // 32
                person_attention_pooled = F.adaptive_avg_pool2d(person_attention, (output_h, output_w))
            
                person_attention_flattened = person_attention_pooled.contiguous().view(person_attention_pooled.shape[0], -1)
                max_val = person_attention_flattened.max()
                if max_val > 0:
                    gaze_map_normalized = person_attention_flattened / max_val
                else:
                    gaze_map_normalized = person_attention_flattened # If all zeros, keep it as is

                action_label = action_label.to(device)

                # 前向推理
                action_pred, gaze_use_map = model(obs_image)

                # 计算损失并记录（注意：此处直接用 .item() 记录数值）
                losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_map_normalized, gaze_use_map=gaze_use_map)
                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

        summary_message_start = f"\n--- Epoch {epoch} ---\n"
        print(summary_message_start)
        f_log.write(summary_message_start)
        f_log.flush()

        gnm_log(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=10, # 在评估时，可以设置一个合理的打印频率
        log_file=f_log, # 传递文件句柄
    )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def compute_aux_loss(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    gaze_map: torch.Tensor,
    attention_scores: torch.Tensor,
):
    """
    Compute loss for token attention and gaze map.
    Args:
        action_label: Ground truth actions [batch_size, action_dim]
        action_pred: Predicted actions [batch_size, action_dim]
        gaze_map: Ground truth gaze maps [batch_size, H*W]
        attention_scores: Attention scores [batch_size, seq_len, seq_len]
    """
    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return (unreduced_loss).mean()

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"

    # Action prediction loss
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # Cosine similarity
    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

    # Final Displacement Error (FDE)
    fde = F.pairwise_distance(
        action_pred[:, -1], 
        action_label[:, -1], 
        p=2
    ).mean()

    # Fréchet Distance & Hausdorff Distance
    frechet_dists = []
    hausdorff_dists = []

    for i in range(action_pred.shape[0]):
        pred_np = action_pred[i].detach().cpu().numpy()
        label_np = action_label[i].detach().cpu().numpy()

        # 真实 Fréchet 距离（考虑点顺序）
        frechet = frdist(pred_np, label_np)
        frechet_dists.append(frechet)

        # 对称 Hausdorff 距离
        h1 = directed_hausdorff(pred_np, label_np)[0]
        h2 = directed_hausdorff(label_np, pred_np)[0]
        hausdorff = max(h1, h2)
        hausdorff_dists.append(hausdorff)

    # Process attention scores to get token importance vector
    # Sum over queries to get importance of each key
    gaze_use_vector = attention_scores.sum(dim=1)  # [batch_size, seq_len]
    
    # Ensure both distributions sum to 1
    gaze_use_vector = F.softmax(gaze_use_vector, dim=1)
    gaze_map_normalized = F.softmax(gaze_map.view(gaze_map.size(0), -1), dim=1)

    assert gaze_use_vector.shape == gaze_map_normalized.shape, \
        f"Shape mismatch for KLDivLoss: gaze_use_vector {gaze_use_vector.shape} vs gaze_map_normalized {gaze_map_normalized.shape}"

    # Token attention auxiliary loss (KL divergence)
    token_aux_loss = F.kl_div(
        gaze_use_vector.log(),  # [batch_size, seq_len]
        gaze_map_normalized,    # [batch_size, H*W]
        reduction='batchmean',
        log_target=False,
    )

    # Combine losses with weight
    alpha = 0.5
    total_loss = (1 - alpha) * action_loss + alpha * token_aux_loss

    results = {
        "action_loss": action_loss,
        "auxiliary_loss": token_aux_loss,
        "total_loss": total_loss,
        "action_waypts_cos_sim": action_waypts_cos_similarity,
        "fde": torch.tensor(fde.item(), device=action_label.device),
        "frechet": torch.tensor(np.mean(frechet_dists), device=action_label.device),
        "hausdorff": torch.tensor(np.mean(hausdorff_dists), device=action_label.device),
    }
    return results


def gazeaux_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    auxiliary_loss_logger = Logger("auxiliary_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            gaze_attention, # [batch_size, context_size+1, H, W]
            action_label,
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device)
        gaze_attention = gaze_attention.squeeze(2)
        output_h = gaze_attention.shape[2] // 32
        output_w = gaze_attention.shape[3] // 32
        gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
        gaze_attention_flattened = gaze_attention_pooled.contiguous().view(gaze_attention_pooled.shape[0], -1)

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image)
            losses = compute_aux_loss(
            action_label=action_label,
            action_pred=action_pred,
            gaze_map=gaze_attention_flattened,
            attention_scores=attention_scores
            )
            loss = losses["total_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def gazeaux_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, gaze_attention, action_label, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            gaze_attention = gaze_attention.to(device)
            gaze_attention = gaze_attention.squeeze(2)
            output_h = gaze_attention.shape[2] // 32
            output_w = gaze_attention.shape[3] // 32
            gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
            gaze_attention_flattened = gaze_attention_pooled.contiguous().view(gaze_attention_pooled.shape[0], -1)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_aux_loss(
                action_label=action_label,
                action_pred=action_pred,
                gaze_map=gaze_attention_flattened,
                attention_scores=attention_scores
            )
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def personaux_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    auxiliary_loss_logger = Logger("auxiliary_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        # Convert person_masks to person_attention by taking union along num_persons dimension
        person_masks = person_masks.to(device)
        select_labels = select_labels.to(device)
        invalid = invalid.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~invalid  # [B, P]
        select_mask = (select_labels == 1) & valid_masks  # [B, P]
        select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

        person_masks = person_masks * select_mask.float()  # Zero out invalid masks
        person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]
        
        # Pool person_attention to match output size
        output_h = person_attention.shape[2] // 32
        output_w = person_attention.shape[3] // 32
        person_attention_pooled = F.adaptive_avg_pool2d(person_attention, (output_h, output_w))
        
        person_attention_flattened = person_attention_pooled.contiguous().view(person_attention_pooled.shape[0], -1)
        max_val = person_attention_flattened.max()
        if max_val > 0:
            gaze_map_normalized = person_attention_flattened / max_val
        else:
            gaze_map_normalized = person_attention_flattened # If all zeros, keep it as is

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image)
            losses = compute_aux_loss(
            action_label=action_label,
            action_pred=action_pred,
            gaze_map=gaze_map_normalized,
            attention_scores=attention_scores
            )
            loss = losses["total_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def personaux_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, person_masks, select_labels, action_label, invalid, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            person_masks = person_masks.to(device)
            select_labels = select_labels.to(device)
            invalid = invalid.to(device)
            # Create selection mask using select_labels and invalid flag
            valid_masks = ~invalid  # [B, P]
            select_mask = (select_labels == 1) & valid_masks  # [B, P]
            select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

            person_masks = person_masks * select_mask.float()  # Zero out invalid masks
            person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]
        
            # Pool person_attention to match output size
            output_h = person_attention.shape[2] // 32
            output_w = person_attention.shape[3] // 32
            person_attention_pooled = F.adaptive_avg_pool2d(person_attention, (output_h, output_w))
        
            person_attention_flattened = person_attention_pooled.contiguous().view(person_attention_pooled.shape[0], -1)
            max_val = person_attention_flattened.max()
            if max_val > 0:
                gaze_map_normalized = person_attention_flattened / max_val
            else:
                gaze_map_normalized = person_attention_flattened # If all zeros, keep it as is

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_aux_loss(
                action_label=action_label,
                action_pred=action_pred,
                gaze_map=gaze_map_normalized,
                attention_scores=attention_scores
            )
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def gnmgazechannel_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            gaze_maps, # [batch_size, context_size+1, H, W] 
            action_label,
            _
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_maps = gaze_maps.to(device)

        action_label = action_label.to(device)

        optimizer.zero_grad()
    
        with autocast():
            action_pred = model(obs_image, gaze_maps)
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        ) """

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())
    
    # epoch 结束后的总结日志（同时打印并写入文件）
    summary_message_start = f"\n--- Epoch {epoch} ---\n"
    print(summary_message_start)

    gnm_log(
        i=0,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=1,
        log_file=None, # 传递文件句柄
    )
        
    
def gnmgazechannel_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    log_file_path = os.path.join(run_folder, "evaluation_log.log") # 评估日志文件名
    os.makedirs(run_folder, exist_ok=True) # 确保 run_folder 存在

    with open(log_file_path, 'a', encoding='utf-8') as f_log:
        # 写入评估开始的提示信息（同时打印）
        header_message = f"\n--- 开始评估 Epoch {epoch} ---\n"
        print(header_message)
        f_log.write(header_message)
        f_log.flush()

        with torch.no_grad():
            for i, data in enumerate(tqdm_iter):
                obs_image, gaze_maps, action_label, _ = data

                obs_images = torch.split(obs_image, 3, dim=1)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                gaze_maps = gaze_maps.to(device)

                action_label = action_label.to(device)

                # 前向推理
                action_pred = model(obs_image, gaze_maps)

                # 计算损失并记录（注意：此处直接用 .item() 记录数值）
                losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())
        
        summary_message_start = f"\n--- Epoch {epoch} 评估总结 ---\n"
        print(summary_message_start)
        f_log.write(summary_message_start)
        f_log.flush()

        gnm_log(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=10, # 在评估时，可以设置一个合理的打印频率
        log_file=f_log, # 传递文件句柄
    )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def gazechannel_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            gaze_maps, # [batch_size, context_size+1, H, W]
            action_label,
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        gaze_maps = gaze_maps.to(device)

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image, gaze_maps)
            losses = compute_baseloss(
            action_label=action_label, action_pred=action_pred,)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def gazechannel_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, gaze_maps, action_label, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            gaze_maps = gaze_maps.to(device)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image, gaze_maps)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred,)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def gnmpersonchannel_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
            _
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        person_masks = person_masks.to(device)
        select_labels = select_labels.to(device)
        invalid = invalid.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~invalid  # [B, P]
        select_mask = (select_labels == 1) & valid_masks  # [B, P]
        select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

        person_masks = person_masks * select_mask.float()  # Zero out invalid masks
        person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred = model(obs_image, person_attention)
            losses = compute_baseloss(
            action_label=action_label, action_pred=action_pred,)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

    # epoch 结束后的总结日志（同时打印并写入文件）
    summary_message_start = f"\n--- Epoch {epoch} ---\n"
    print(summary_message_start)

    gnm_log(
        i=0,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=1,
        log_file=None, # 传递文件句柄
    )


def gnmpersonchannel_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    log_file_path = os.path.join(run_folder, "evaluation_log.log") # 评估日志文件名
    os.makedirs(run_folder, exist_ok=True) # 确保 run_folder 存在

    with open(log_file_path, 'a', encoding='utf-8') as f_log:
        # 写入评估开始的提示信息（同时打印）
        header_message = f"\n--- 开始评估 Epoch {epoch} ---\n"
        print(header_message)
        f_log.write(header_message)
        f_log.flush()

        with torch.no_grad():
            for i, data in enumerate(tqdm_iter):
                obs_image, person_masks, select_labels, action_label, invalid, _ = data
        
                viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

                obs_images = torch.split(obs_image, 3, dim=1)
                obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
                obs_image = torch.cat(obs_images, dim=1)

                # Convert person_masks to person_attention by taking union along num_persons dimension
                person_masks = person_masks.to(device)
                select_labels = select_labels.to(device)
                invalid = invalid.to(device)
                # Create selection mask using select_labels and invalid flag
                valid_masks = ~invalid  # [B, P]
                select_mask = (select_labels == 1) & valid_masks  # [B, P]
                select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

                person_masks = person_masks * select_mask.float()  # Zero out invalid masks
                person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

                action_label = action_label.to(device)

                # 前向推理
                action_pred = model(obs_image, person_attention)

                # 计算损失并记录（注意：此处直接用 .item() 记录数值）
                losses = compute_baseloss(action_label=action_label, action_pred=action_pred,)
                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

        summary_message_start = f"\n--- Epoch {epoch} 评估总结 ---\n"
        print(summary_message_start)
        f_log.write(summary_message_start)
        f_log.flush()

        gnm_log(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        loggers=loggers,
        use_latest=False,
        print_log_freq=10, # 在评估时，可以设置一个合理的打印频率
        log_file=f_log, # 传递文件句柄
    )
    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def personchannel_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        person_masks = person_masks.to(device)
        select_labels = select_labels.to(device)
        invalid = invalid.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~invalid  # [B, P]
        select_mask = (select_labels == 1) & valid_masks  # [B, P]
        select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

        person_masks = person_masks * select_mask.float()  # Zero out invalid masks
        person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image, person_attention)
            losses = compute_baseloss(
            action_label=action_label, action_pred=action_pred,)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def personchannel_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, person_masks, select_labels, action_label, invalid, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            person_masks = person_masks.to(device)
            select_labels = select_labels.to(device)
            invalid = invalid.to(device)
            # Create selection mask using select_labels and invalid flag
            valid_masks = ~invalid  # [B, P]
            select_mask = (select_labels == 1) & valid_masks  # [B, P]
            select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

            person_masks = person_masks * select_mask.float()  # Zero out invalid masks
            person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image, person_attention)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred,)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def gazetoken_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            gaze_maps, # [batch_size, context_size+1, H, W]
            action_label,
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        gaze_maps = gaze_maps.to(device)

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image, gaze_maps)
            losses = compute_baseloss(
            action_label=action_label, action_pred=action_pred,)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def gazetoken_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, gaze_maps, action_label, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            gaze_maps = gaze_maps.to(device)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image, gaze_maps)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred,)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def persontoken_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    fde_logger = Logger("fde", "train", window_size=print_log_freq)
    frechet_logger = Logger("frechet", "train", window_size=print_log_freq)
    hausdorff_logger = Logger("hausdorff", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "fde": fde_logger,
        "frechet": frechet_logger,
        "hausdorff": hausdorff_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        person_masks = person_masks.to(device)
        select_labels = select_labels.to(device)
        invalid = invalid.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~invalid  # [B, P]
        select_mask = (select_labels == 1) & valid_masks  # [B, P]
        select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

        person_masks = person_masks * select_mask.float()  # Zero out invalid masks, [B, P, C, H, W]
        person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image, person_attention)
            losses = compute_baseloss(
            action_label=action_label, action_pred=action_pred,)
            loss = losses["action_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        action_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            action_pred=action_pred,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def persontoken_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
        "fde": Logger("fde", "test"),
        "frechet": Logger("frechet", "test"),
        "hausdorff": Logger("hausdorff", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, person_masks, select_labels, action_label, invalid, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            person_masks = person_masks.to(device)
            select_labels = select_labels.to(device)
            invalid = invalid.to(device)
            # Create selection mask using select_labels and invalid flag
            valid_masks = ~invalid  # [B, P]
            select_mask = (select_labels == 1) & valid_masks  # [B, P]
            select_mask = select_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, P, 1, 1, 1]

            person_masks = person_masks * select_mask.float()  # Zero out invalid masks
            person_attention = (person_masks.sum(dim=1) > 0).float()  # [batch_size, context_size+1, H, W]

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image, person_attention)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_baseloss(action_label=action_label, action_pred=action_pred,)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                action_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    action_pred=action_pred,
                    attention_scores=attention_scores,
                    action_label=action_label,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # 返回主要评估指标
    return loggers["action_loss"].average()

###################################################################################################

def compute_dumobsloss(
    true_winners: torch.Tensor, # 这是一个布尔张量，形状为 [B, P]
    logits: torch.Tensor,     # 模型的输出，形状为 [B, P]，包含 -inf
    pad: torch.Tensor = None, # 从 obs_train 传入的 padding 掩码，形状为 [B, P]，True 表示 padding
    pos_weight: float = 3, # 新增：正样本的权重，float 类型，直接传入 BCEWithLogitsLoss
    prediction_threshold: float = 0.4, # 新增：用于二值分类的预测阈值
):
    """
    Compute losses and metrics for select prediction.
    Handles -inf logits for padded entries by masking the loss and metrics.
    Includes pos_weight for BCEWithLogitsLoss and F1-score calculation.
    """

    assert logits.shape == true_winners.shape, f"Logits shape {logits.shape} != true_winners shape {true_winners.shape}"
    if pad is not None:
        assert logits.shape == pad.shape, f"Logits shape {logits.shape} != pad shape {pad.shape}"

    # Convert boolean labels to float for loss computation
    true_winner_float = true_winners.float()

    # Prepare pos_weight for BCEWithLogitsLoss.
    # If a float is provided, convert it to a tensor on the same device and dtype as true_winner_float.
    # This is for robustness, though BCEWithLogitsLoss can often handle float directly.
    _pos_weight_tensor = None
    if pos_weight is not None:
        _pos_weight_tensor = torch.tensor(pos_weight, device=true_winner_float.device, dtype=true_winner_float.dtype)

    # Initialize loss function with optional pos_weight and reduction='none'
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=_pos_weight_tensor)

    # Calculate per-element loss
    # select_loss will have shape [B, P]
    select_loss = loss_fn(logits, true_winner_float)

    # Convert logits to predicted probabilities using sigmoid
    predicted_probs = torch.sigmoid(logits)
    # Convert probabilities to binary predictions using the configurable threshold
    predicted_winners = (predicted_probs > prediction_threshold) # Boolean tensor of predictions [B, P]


    # Initialize metrics to zero
    total_obs_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    accuracy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    recall = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    precision = torch.tensor(0.0, device=logits.device, dtype=logits.dtype) 
    f1_score = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)   


    # Handle padding:
    if pad is not None:
        valid_masks = ~pad # [B, P]
        
        # Mask out loss for padded positions
        masked_select_loss = select_loss * valid_masks.float()
        
        # Count the number of valid elements for averaging
        num_valid_elements = valid_masks.sum().float()
        
        if num_valid_elements > 0:
            total_obs_loss = masked_select_loss.sum() / num_valid_elements
            
            # Calculate correct predictions only for valid elements
            correct_predictions = ((predicted_winners == true_winners) & valid_masks).sum().float()
            accuracy = correct_predictions / num_valid_elements

            # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), Actual Positives (AP)
            true_positives = ((predicted_winners == True) & (true_winners == True) & valid_masks).sum().float()
            false_positives = ((predicted_winners == True) & (true_winners == False) & valid_masks).sum().float()
            actual_positives = (true_winners & valid_masks).sum().float() # Equivalent to TP + FN

            # Calculate Precision
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            # else: precision remains 0.0 if no positives predicted

            # Calculate Recall
            if actual_positives > 0:
                recall = true_positives / actual_positives
            else:
                recall = torch.tensor(1.0, device=logits.device, dtype=logits.dtype) # Convention for no actual positives (perfect recall if nothing to miss)
                # If true_positives > 0 and actual_positives == 0, this implies a logical error or empty ground truth for positives.
                # In such an extreme case, recall should typically be 0.0, but the if-else structure for actual_positives covers this.
            # Calculate F1-Score
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)

        else:
            # If the entire batch is padded, all metrics are 0.0
            total_obs_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            accuracy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            recall = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            precision = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            f1_score = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    else:
        # If no pad is provided, calculate loss and metrics for all elements
        total_obs_loss = select_loss.mean()
        
        correct_predictions = (predicted_winners == true_winners).sum().float()
        accuracy = correct_predictions / true_winners.numel()

        true_positives = ((predicted_winners == True) & (true_winners == True)).sum().float()
        false_positives = ((predicted_winners == True) & (true_winners == False)).sum().float()
        false_negatives = ((predicted_winners == False) & (true_winners == True)).sum().float()
        actual_positives = true_winners.sum().float()

        # Calculate Precision
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)

        # Calculate Recall
        if actual_positives > 0:
            recall = true_positives / actual_positives
        else:
            recall = torch.tensor(1.0, device=logits.device, dtype=logits.dtype) # Convention for no actual positives
            # Similar logic as above for true_positives > 0 and actual_positives == 0.

        # Calculate F1-Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)


    results = {
        "obs_loss": total_obs_loss,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision, 
        "f1_score": f1_score,   
    }
    return results


def onephase_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    obs_loss_logger = Logger("obs_loss", "train", window_size=print_log_freq)
    accuracy_logger = Logger("accuracy", "train", window_size=print_log_freq)
    recall_logger = Logger("recall", "train", window_size=print_log_freq)
    precision_logger = Logger("precision", "train", window_size=print_log_freq)
    f1_logger = Logger("f1_score", "train", window_size=print_log_freq)
    
    loggers = {
        "obs_loss": obs_loss_logger,
        "accuracy": accuracy_logger,
        "recall": recall_logger,
        "precision": precision_logger,
        "f1_score":f1_logger,
    }

    # Lists to accumulate true labels and predicted probabilities for AUC-PR
    all_true_winners_epoch = []
    all_predicted_probs_epoch = []

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            candidates_masks, # [batch_size, num_persons, context_size+1, H, W]
            _,
            chosen, # [batch_size, num_persons]
            pad, # [batch_size, num_persons]
            _
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        candidates_masks = candidates_masks.to(device)
        chosen = chosen.to(device)
        pad = pad.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~pad  # [B, P]
        true_winners = (chosen == 1) & valid_masks  # [B, P]

        optimizer.zero_grad()

        with autocast():
            logits = model(obs_image, candidates_masks, pad)  # [B, P]
            
            losses = compute_dumobsloss(
            true_winners=true_winners, logits=logits, pad=pad)
            loss = losses["obs_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        # Accumulate data for AUC-PR
        current_predicted_probs = torch.sigmoid(logits).detach().cpu().numpy()
        current_true_winners = true_winners.detach().cpu().numpy()
        current_valid_masks = valid_masks.detach().cpu().numpy()

        all_predicted_probs_epoch.extend(current_predicted_probs[current_valid_masks])
        all_true_winners_epoch.extend(current_true_winners[current_valid_masks])

        probabilities = torch.sigmoid(logits)  # [B, P]

        viz_winner_masks = candidates_masks.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, context_size+1, num_persons, H, W]

        obsp_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            winner_masks=viz_winner_masks,
            obs_pred=probabilities,
            obs_label=chosen,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )

    # --- Calculate and Log AUC-PR at the end of the epoch ---
    if len(all_true_winners_epoch) > 0 and (np.sum(all_true_winners_epoch) > 0 and np.sum(1 - np.array(all_true_winners_epoch)) > 0):
        # Only calculate if there are both positive and negative samples
        precisions, recalls, _ = precision_recall_curve(all_true_winners_epoch, all_predicted_probs_epoch)
        auc_pr_epoch = auc(recalls, precisions)
        
        print(f"Epoch {epoch} Train AUC-PR: {auc_pr_epoch:.4f}")
        if use_wandb:
            wandb.log({"train_auc_pr": auc_pr_epoch})
    elif len(all_true_winners_epoch) > 0:
        # Handle cases where only one class is present after filtering by valid_masks
        if np.sum(all_true_winners_epoch) == 0:
            print(f"Epoch {epoch} Train AUC-PR: N/A (No positive samples in valid data)")
            if use_wandb:
                wandb.log({"train_auc_pr": 0.0}) # Conventionally 0 if no positives
        else: # Only positives
            print(f"Epoch {epoch} Train AUC-PR: N/A (No negative samples in valid data)")
            if use_wandb:
                wandb.log({"train_auc_pr": 1.0}) # Conventionally 1 if no negatives
    else:
        print(f"Epoch {epoch} Train AUC-PR: N/A (No valid samples for PR curve)")
        if use_wandb:
            wandb.log({"train_auc_pr": 0.0})


def onephase_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "obs_loss": Logger("obs_loss", "test"),
        "accuracy": Logger("accuracy", "test"),
        "recall": Logger("recall", "test"),
        "precision": Logger("precision", "test"),
        "f1_score": Logger("f1_score", "test")
    }

    # Lists to accumulate true labels and predicted probabilities for AUC-PR
    all_true_winners_eval = []
    all_predicted_probs_eval = []

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, candidates_masks, _, chosen, pad, _= data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            candidates_masks = candidates_masks.to(device)
            chosen = chosen.to(device)
            pad = pad.to(device)
            # Create selection mask using select_labels and invalid flag
            valid_masks = ~pad  # [B, P]
            true_winners = (chosen ==1) & valid_masks  #[B. P]

            # 前向推理
            logits = model(obs_image, candidates_masks, pad)  # [B, P]

            losses = compute_dumobsloss(true_winners=true_winners, logits=logits, pad=pad)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # Accumulate data for AUC-PR
            current_predicted_probs = torch.sigmoid(logits).cpu().numpy()
            current_true_winners = true_winners.cpu().numpy()
            current_valid_masks = valid_masks.cpu().numpy()

            all_predicted_probs_eval.extend(current_predicted_probs[current_valid_masks])
            all_true_winners_eval.extend(current_true_winners[current_valid_masks])

            probabilities = torch.sigmoid(logits)

            viz_winner_masks = candidates_masks.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, context_size+1, num_persons, H, W]

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                obsp_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    winner_masks=viz_winner_masks,
                    obs_pred=probabilities,
                    obs_label=chosen,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # --- Calculate and Log AUC-PR at the end of evaluation ---
    if len(all_true_winners_eval) > 0 and (np.sum(all_true_winners_eval) > 0 and np.sum(1 - np.array(all_true_winners_eval)) > 0):
        precisions, recalls, _ = precision_recall_curve(all_true_winners_eval, all_predicted_probs_eval)
        auc_pr_eval = auc(recalls, precisions)
        
        print(f"Epoch {epoch} Test AUC-PR: {auc_pr_eval:.4f}")
        if use_wandb:
            wandb.log({"test_auc_pr": auc_pr_eval})
        return auc_pr_eval # Return AUC-PR as a primary metric
    elif len(all_true_winners_eval) > 0:
        if np.sum(all_true_winners_eval) == 0:
            print(f"Epoch {epoch} Test AUC-PR: N/A (No positive samples in valid data)")
            if use_wandb:
                wandb.log({"test_auc_pr": 0.0})
            return 0.0
        else: # Only positives
            print(f"Epoch {epoch} Test AUC-PR: N/A (No negative samples in valid data)")
            if use_wandb:
                wandb.log({"test_auc_pr": 1.0})
            return 1.0
    else:
        print(f"Epoch {epoch} Test AUC-PR: N/A (No valid samples for PR curve)")
        if use_wandb:
            wandb.log({"test_auc_pr": 0.0})
        return 0.0

###################################################################################################

def compute_obsloss(
    true_winners: torch.Tensor, # 这是一个布尔张量，形状为 [B, P]
    logits: torch.Tensor,     # 模型的输出，形状为 [B, P]，包含 -inf
    model_attn_map: torch.Tensor, # WinnerSelectorPlus 模型输出的注意力图，形状为 [B, N * spatial_flatten_len]，已 softmax
    gt_attn_map: torch.Tensor,    # 从 act_model 生成的地面真实注意力图，形状为 [B, N * spatial_flatten_len]，已 softmax
    pad: torch.Tensor = None, # 从 obs_train 传入的 padding 掩码，形状为 [B, P]，True 表示 padding
    pos_weight: float = 3, # 正样本的权重，float 类型，直接传入 BCEWithLogitsLoss
    prediction_threshold: float = 0.4, # 新增：用于二值分类的预测阈值
):
    """
    Compute losses and metrics for select prediction.
    Handles -inf logits for padded entries by masking the loss and metrics.
    Includes pos_weight for BCEWithLogitsLoss, F1-score calculation,
    and an auxiliary KLDivLoss for attention map supervision.
    """
    # --- 1. Winner Selection Loss (BCEWithLogitsLoss) ---
    assert logits.shape == true_winners.shape, f"Logits shape {logits.shape} != true_winners shape {true_winners.shape}"
    if pad is not None:
        assert logits.shape == pad.shape, f"Logits shape {logits.shape} != pad shape {pad.shape}"

    # Convert boolean labels to float for loss computation
    true_winner_float = true_winners.float()

    # Prepare pos_weight for BCEWithLogitsLoss.
    # If a float is provided, convert it to a tensor on the same device and dtype as true_winner_float.
    # This is for robustness, though BCEWithLogitsLoss can often handle float directly.
    _pos_weight_tensor = None
    if pos_weight is not None:
        _pos_weight_tensor = torch.tensor(pos_weight, device=true_winner_float.device, dtype=true_winner_float.dtype)

    # Initialize loss function with optional pos_weight and reduction='none'
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=_pos_weight_tensor)

    # Calculate per-element loss
    # select_loss will have shape [B, P]
    select_loss = loss_fn(logits, true_winner_float)

    # --- 2. Auxiliary Attention Map Loss (KLDivLoss) ---
    assert model_attn_map.shape == gt_attn_map.shape, \
        f"Model attention map shape {model_attn_map.shape} != GT attention map shape {gt_attn_map.shape}"
    
    auxiliary_loss = F.kl_div(
        torch.log(model_attn_map), # Input: log-probabilities
        gt_attn_map,                         # Target: probabilities
        reduction='batchmean'                # Averages KLDivLoss over the batch
    )

    # --- Aggregated Total Loss & Metrics Calculation (with padding handling) ---
    
    # Convert logits to predicted probabilities using sigmoid
    predicted_probs = torch.sigmoid(logits)
    # Convert probabilities to binary predictions using the configurable threshold
    predicted_winners = (predicted_probs > prediction_threshold) # Boolean tensor of predictions [B, P]


    # Initialize metrics to zero
    total_obs_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    accuracy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    recall = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    precision = torch.tensor(0.0, device=logits.device, dtype=logits.dtype) 
    f1_score = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)   


    # Handle padding:
    if pad is not None:
        valid_masks = ~pad # [B, P]
        
        # Mask out loss for padded positions
        masked_select_loss = select_loss * valid_masks.float()
        
        # Count the number of valid elements for averaging
        num_valid_elements = valid_masks.sum().float()
        
        if num_valid_elements > 0:
            # Average the primary selection loss over valid elements
            primary_loss_mean = masked_select_loss.sum() / num_valid_elements
            
            # Combine primary loss with auxiliary loss
            alpha = 0.5
            total_obs_loss = (1-alpha) * primary_loss_mean + alpha * auxiliary_loss
            
            # Calculate correct predictions only for valid elements
            correct_predictions = ((predicted_winners == true_winners) & valid_masks).sum().float()
            accuracy = correct_predictions / num_valid_elements

            # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), Actual Positives (AP)
            true_positives = ((predicted_winners == True) & (true_winners == True) & valid_masks).sum().float()
            false_positives = ((predicted_winners == True) & (true_winners == False) & valid_masks).sum().float()
            actual_positives = (true_winners & valid_masks).sum().float() # Equivalent to TP + FN

            # Calculate Precision
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            # else: precision remains 0.0 if no positives predicted

            # Calculate Recall
            if actual_positives > 0:
                recall = true_positives / actual_positives
            else:
                recall = torch.tensor(1.0, device=logits.device, dtype=logits.dtype) # Convention for no actual positives (perfect recall if nothing to miss)
                # If true_positives > 0 and actual_positives == 0, this implies a logical error or empty ground truth for positives.
                # In such an extreme case, recall should typically be 0.0, but the if-else structure for actual_positives covers this.
            # Calculate F1-Score
            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)

        else:
            alpha = 0.5
            total_obs_loss = alpha * auxiliary_loss
            # If the entire batch is padded, all metrics are 0.0
            primary_loss_mean = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            accuracy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            recall = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            precision = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            f1_score = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    else:
        # If no pad is provided, calculate loss and metrics for all elements
        primary_loss_mean = select_loss_per_element.mean()
        total_obs_loss = primary_loss_mean + kl_weight * auxiliary_loss
        
        correct_predictions = (predicted_winners == true_winners).sum().float()
        accuracy = correct_predictions / true_winners.numel()

        true_positives = ((predicted_winners == True) & (true_winners == True)).sum().float()
        false_positives = ((predicted_winners == True) & (true_winners == False)).sum().float()
        false_negatives = ((predicted_winners == False) & (true_winners == True)).sum().float()
        actual_positives = true_winners.sum().float()

        # Calculate Precision
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)

        # Calculate Recall
        if actual_positives > 0:
            recall = true_positives / actual_positives
        else:
            recall = torch.tensor(1.0, device=logits.device, dtype=logits.dtype) # Convention for no actual positives
            # Similar logic as above for true_positives > 0 and actual_positives == 0.

        # Calculate F1-Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)


    results = {
        "obs_loss": total_obs_loss,
        "select_loss_primary": primary_loss_mean if 'primary_loss_mean' in locals() else torch.tensor(0.0, device=logits.device, dtype=logits.dtype), # Add primary loss for separate logging
        "auxiliary_loss_attn": auxiliary_loss, # Separate logging for auxiliary loss
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision, 
        "f1_score": f1_score,   
    }
    return results


def onephaseplus_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    obs_loss_logger = Logger("obs_loss", "train", window_size=print_log_freq)
    select_loss_primary_logger = Logger("select_loss_primary", "train", window_size=print_log_freq)
    auxiliary_loss_attn_logger = Logger("auxiliary_loss_attn", "train", window_size=print_log_freq)
    accuracy_logger = Logger("accuracy", "train", window_size=print_log_freq)
    recall_logger = Logger("recall", "train", window_size=print_log_freq)
    precision_logger = Logger("precision", "train", window_size=print_log_freq)
    f1_logger = Logger("f1_score", "train", window_size=print_log_freq)
    
    loggers = {
        "obs_loss": obs_loss_logger,
        "select_loss_primary": select_loss_primary_logger,
        "auxiliary_loss_attn": auxiliary_loss_attn_logger,
        "accuracy": accuracy_logger,
        "recall": recall_logger,
        "precision": precision_logger,
        "f1_score": f1_logger,
    }

    # Lists to accumulate true labels and predicted probabilities for AUC-PR
    all_true_winners_epoch = []
    all_predicted_probs_epoch = []

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            candidates_masks, # [batch_size, num_persons, context_size+1, H, W]
            act_attnmaps, # [batch_size, N*spatial_flatten_len]
            chosen, # [batch_size, num_persons]
            pad, # [batch_size, num_persons]
            _ # Batched original indices
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        candidates_masks = candidates_masks.to(device)
        act_attnmaps = act_attnmaps.to(device)
        chosen = chosen.to(device)
        pad = pad.to(device)
        # Create selection mask using select_labels and invalid flag
        valid_masks = ~pad  # [B, P]
        true_winners = (chosen == 1) & valid_masks  # [B, P]

        optimizer.zero_grad()

        with autocast():
            logits, obs_attnmaps = model(obs_image, candidates_masks, pad)  # [B, P]
            
            losses = compute_obsloss(
                true_winners=chosen,
                logits=logits,
                model_attn_map=obs_attnmaps,       # Model's attention map
                gt_attn_map=act_attnmaps,      # Ground truth attention map (already flattened)
                pad=pad,                    # Padding mask
                )
            loss = losses["obs_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        # Accumulate data for AUC-PR
        current_predicted_probs = torch.sigmoid(logits).detach().cpu().numpy()
        current_true_winners = true_winners.detach().cpu().numpy()
        current_valid_masks = valid_masks.detach().cpu().numpy()

        all_predicted_probs_epoch.extend(current_predicted_probs[current_valid_masks])
        all_true_winners_epoch.extend(current_true_winners[current_valid_masks])

        probabilities = torch.sigmoid(logits)  # [B, P]

        viz_winner_masks = candidates_masks.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, context_size+1, num_persons, H, W]

        obsp_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            winner_masks=viz_winner_masks,
            obs_pred=probabilities,
            obs_label=chosen,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )

    # --- Calculate and Log AUC-PR at the end of the epoch ---
    if len(all_true_winners_epoch) > 0 and (np.sum(all_true_winners_epoch) > 0 and np.sum(1 - np.array(all_true_winners_epoch)) > 0):
        # Only calculate if there are both positive and negative samples
        precisions, recalls, _ = precision_recall_curve(all_true_winners_epoch, all_predicted_probs_epoch)
        auc_pr_epoch = auc(recalls, precisions)
        
        print(f"Epoch {epoch} Train AUC-PR: {auc_pr_epoch:.4f}")
        if use_wandb:
            wandb.log({"train_auc_pr": auc_pr_epoch})
    elif len(all_true_winners_epoch) > 0:
        # Handle cases where only one class is present after filtering by valid_masks
        if np.sum(all_true_winners_epoch) == 0:
            print(f"Epoch {epoch} Train AUC-PR: N/A (No positive samples in valid data)")
            if use_wandb:
                wandb.log({"train_auc_pr": 0.0}) # Conventionally 0 if no positives
        else: # Only positives
            print(f"Epoch {epoch} Train AUC-PR: N/A (No negative samples in valid data)")
            if use_wandb:
                wandb.log({"train_auc_pr": 1.0}) # Conventionally 1 if no negatives
    else:
        print(f"Epoch {epoch} Train AUC-PR: N/A (No valid samples for PR curve)")
        if use_wandb:
            wandb.log({"train_auc_pr": 0.0})


def onephaseplus_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "obs_loss": Logger("obs_loss", "test"),
        "select_loss_primary": Logger("select_loss_primary", "test"),
        "auxiliary_loss_attn": Logger("auxiliary_loss_attn", "test"),
        "accuracy": Logger("accuracy", "test"),
        "recall": Logger("recall", "test"),
        "precision": Logger("precision", "test"),
        "f1_score": Logger("f1_score", "test")
    }

    # Lists to accumulate true labels and predicted probabilities for AUC-PR
    all_true_winners_eval = []
    all_predicted_probs_eval = []

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, candidates_masks, act_attnmaps, chosen, pad, _= data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            candidates_masks = candidates_masks.to(device)
            act_attnmaps = act_attnmaps.to(device)
            chosen = chosen.to(device)
            pad = pad.to(device)
            # Create selection mask using select_labels and invalid flag
            valid_masks = ~pad  # [B, P]
            true_winners = (chosen ==1) & valid_masks  #[B. P]

            # 前向推理
            logits, obs_attnmaps = model(obs_image, candidates_masks, pad)  # [B, P]

            losses = compute_obsloss(true_winners=true_winners, logits=logits, model_attn_map=obs_attnmaps, gt_attn_map=act_attnmaps,pad=pad)

            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # Accumulate data for AUC-PR
            current_predicted_probs = torch.sigmoid(logits).cpu().numpy()
            current_true_winners = true_winners.cpu().numpy()
            current_valid_masks = valid_masks.cpu().numpy()

            all_predicted_probs_eval.extend(current_predicted_probs[current_valid_masks])
            all_true_winners_eval.extend(current_true_winners[current_valid_masks])

            probabilities = torch.sigmoid(logits)

            viz_winner_masks = candidates_masks.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, context_size+1, num_persons, H, W]

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                obsp_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    winner_masks=viz_winner_masks,
                    obs_pred=probabilities,
                    obs_label=chosen,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    # --- Calculate and Log AUC-PR at the end of evaluation ---
    if len(all_true_winners_eval) > 0 and (np.sum(all_true_winners_eval) > 0 and np.sum(1 - np.array(all_true_winners_eval)) > 0):
        precisions, recalls, _ = precision_recall_curve(all_true_winners_eval, all_predicted_probs_eval)
        auc_pr_eval = auc(recalls, precisions)
        
        print(f"Epoch {epoch} Test AUC-PR: {auc_pr_eval:.4f}")
        if use_wandb:
            wandb.log({"test_auc_pr": auc_pr_eval})
        return auc_pr_eval # Return AUC-PR as a primary metric
    elif len(all_true_winners_eval) > 0:
        if np.sum(all_true_winners_eval) == 0:
            print(f"Epoch {epoch} Test AUC-PR: N/A (No positive samples in valid data)")
            if use_wandb:
                wandb.log({"test_auc_pr": 0.0})
            return 0.0
        else: # Only positives
            print(f"Epoch {epoch} Test AUC-PR: N/A (No negative samples in valid data)")
            if use_wandb:
                wandb.log({"test_auc_pr": 1.0})
            return 1.0
    else:
        print(f"Epoch {epoch} Test AUC-PR: N/A (No valid samples for PR curve)")
        if use_wandb:
            wandb.log({"test_auc_pr": 0.0})
        return 0.0

##################################################################################################

def compute_dumgazeloss(
    predicted_fixations: torch.Tensor, # 模型的输出，形状为 [B, 2]
    gt_fixations: torch.Tensor,       # 地面真实注视点坐标，形状为 [B, 2]
) -> Dict[str, torch.Tensor]:
    """
    Compute losses and metrics for gaze fixation prediction for single points.
    Main loss is MSE between predicted and ground truth fixation points.

    Args:
        predicted_fixations (torch.Tensor): Model's output fixation points, shape [B, 2] (x, y coordinates).
        gt_fixations (torch.Tensor): Ground truth fixation points, shape [B, 2] (x, y coordinates).

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing various loss components and metrics.
    """
    # --- 1. Main Fixation Loss (MSELoss) ---
    assert predicted_fixations.shape == gt_fixations.shape, \
        f"Predicted fixations shape {predicted_fixations.shape} != GT fixations shape {gt_fixations.shape}"
    assert predicted_fixations.ndim == 2 and predicted_fixations.shape[1] == 2, \
        f"Input fixations must be of shape [B, 2], but got {predicted_fixations.shape}"


    # Calculate Mean Squared Error (MSE) per batch element
    # F.mse_loss with reduction='mean' will correctly average over all elements in the tensor.
    main_fixation_loss = F.mse_loss(predicted_fixations, gt_fixations, reduction='mean') # Scalar loss

    # --- Metrics for Regression ---
    # RMSE: Root Mean Squared Error (spatial distance)
    # MAE: Mean Absolute Error (spatial distance)

    # Calculate squared differences across coordinates (x and y)
    # This will be [B, 2] where each column is (pred_x - gt_x)^2 and (pred_y - gt_y)^2
    squared_diff = (predicted_fixations - gt_fixations)**2 
    
    # Sum squared differences across x and y coordinates (dim=1 for [B, 2] shape),
    # then take sqrt for Euclidean distance for each sample
    # Resulting shape: [B] (each element is the Euclidean distance for one sample)
    per_sample_euclidean_distance = torch.sqrt(squared_diff.sum(dim=1)) 

    # Calculate RMSE (Root Mean Squared Error) - The average Euclidean distance across the batch
    rmse = per_sample_euclidean_distance.mean() # Scalar

    # Calculate MAE (Mean Absolute Error)
    # F.l1_loss with reduction='mean' will correctly average over all elements.
    mae = F.l1_loss(predicted_fixations, gt_fixations, reduction='mean') # Scalar

    results = {
        "fixation_loss_mse": main_fixation_loss,
        "rmse": rmse, # Reporting the average Euclidean distance
        "mae": mae,
    }
    return results


def render_fixations_to_gaze_maps(
    fixations_batch: torch.Tensor, # 形状: [B, C, 2]，其中 C 是帧数 (例如 context_size)
    H: int, # Gaze Map 的高度
    W: int, # Gaze Map 的宽度
    sigma: float, # 高斯核的标准差
    device: torch.device # 运行设备
) -> torch.Tensor: # 返回: [B, C, H, W]
    
    B, C, _ = fixations_batch.shape
    gaze_maps_batch = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)

    # 预生成网格坐标，避免在循环中重复创建
    x_coords = torch.arange(0, W, device=device)
    y_coords = torch.arange(0, H, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    for b in range(B): # 遍历 Batch 中的每个样本
        for c in range(C): # 遍历每个样本中的每一帧
            fx, fy = fixations_batch[b, c, 0], fixations_batch[b, c, 1]

            # 将浮点坐标四舍五入到最近的像素整数坐标
            fx_int = round(float(fx))
            fy_int = round(float(fy))

            # 仅在坐标有效（在图像边界内）时进行渲染
            if 0 <= fx_int < W and 0 <= fy_int < H:
                # 生成 2D 高斯核，中心在 (fx_int, fy_int)
                gaussian = torch.exp(-((x_grid - fx_int)**2 + (y_grid - fy_int)**2) / (2 * sigma**2))
                
                # 归一化高斯核，使其最大值为 1
                if gaussian.max() > 0:
                    gaussian = gaussian / gaussian.max() 
                
                # 将生成的 Gaze Map 赋值到对应的位置
                gaze_maps_batch[b, c, :, :] = gaussian
            # 如果坐标无效，对应的 gaze_map_batch[b, c, :, :] 保持为零（由初始化保证）
            
    return gaze_maps_batch


def gaze_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    fixation_loss_mse_logger = Logger("fixation_loss_mse", "train", window_size=print_log_freq)
    rmse_logger = Logger("rmse", "train", window_size=print_log_freq)
    mae_logger = Logger("mae", "train", window_size=print_log_freq)
    
    loggers = {
        "fixation_loss_mse": fixation_loss_mse_logger,
        "rmse": rmse_logger,
        "mae": mae_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            fixations, # [batch_size, context_size+1, 2]
            _, # [batch_size, num_persons, context_size+1, H, W]
            _,
            _,
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        fixations = fixations.to(device)
        prev_fixation = fixations[:, :-1, :]  # [B, C, 2] - previous fixations

        prev_gaze_maps_for_model = render_fixations_to_gaze_maps(
            fixations_batch=prev_fixation,
            H=128, # 使用传入的图像高度
            W=160,  # 使用传入的图像宽度
            sigma=10.0, # 使用传入的高斯核标准差
            device=device
        )
        
        optimizer.zero_grad()

        with autocast():
            predicted_fixation, obs_attnmaps = model(obs_image, prev_gaze_maps_for_model)  # [B, P]
            
            losses = compute_dumgazeloss(
                predicted_fixations= predicted_fixation, gt_fixations=fixations[:, -1, :])
            loss = losses["fixation_loss_mse"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        gaze_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            pred_fixations=predicted_fixation,
            attention_scores=obs_attnmaps,
            gt_fixations=fixations,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def gaze_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "fixation_loss_mse": Logger("fixation_loss_mse", "test"),
        "rmse": Logger("rmse", "test"),
        "mae": Logger("mae", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, fixations, _, _, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            fixations = fixations.to(device)
            prev_fixation = fixations[:, :-1, :]  # [B, C, 2] - previous fixations

            prev_gaze_maps_for_model = render_fixations_to_gaze_maps(
                fixations_batch=prev_fixation,
                H=128, # 使用传入的图像高度
                W=160,  # 使用传入的图像宽度
                sigma=10.0, # 使用传入的高斯核标准差
                device=device
            )
            
            # 前向推理
            predicted_fixation, obs_attnmaps = model(obs_image, prev_gaze_maps_for_model)  # [B, P]

            losses = compute_dumgazeloss(predicted_fixations=predicted_fixation, gt_fixations=fixations[:, -1, :])
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                gaze_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    pred_fixations=predicted_fixation,
                    attention_scores=obs_attnmaps,
                    gt_fixations=fixations,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    return loggers["fixation_loss_mse"].average()

###################################################################################################

def compute_gazeloss(
    predicted_fixations: torch.Tensor, # 模型的输出，形状为 [B, N, 2]
    gt_fixations: torch.Tensor,       # 地面真实注视点坐标，形状为 [B, N, 2]
    model_attn_map: torch.Tensor,     # GazePredictor 模型输出的注意力图，形状为 [B, N * spatial_flatten_len]，已 softmax
    gt_attn_map: torch.Tensor,        # 从 act_model 生成的地面真实注意力图，形状为 [B, N * spatial_flatten_len]，已 softmax
    aux_loss_weight: float = 0.5,
):
    """
    Compute losses and metrics for gaze fixation prediction.
    Main loss is MSE between predicted and ground truth fixation points across all frames in context.
    Includes an auxiliary KLDivLoss for attention map supervision.

    Args:
        predicted_fixations (torch.Tensor): Model's predicted fixation points, shape [B, N, 2].
        gt_fixations (torch.Tensor): Ground truth fixation points, shape [B, N, 2].
        model_attn_map (torch.Tensor): Model's output attention map, shape [B, N * spatial_flatten_len], already softmaxed.
        gt_attn_map (torch.Tensor): Ground truth attention map, shape [B, N * spatial_flatten_len], already softmaxed.
        aux_loss_weight (float): Weight for the auxiliary attention map loss. Should be between 0 and 1.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing various loss components and metrics.
    """
    # --- 1. Main Fixation Loss (MSELoss) ---
    assert predicted_fixations.shape == gt_fixations.shape, \
        f"Predicted fixations shape {predicted_fixations.shape} != GT fixations shape {gt_fixations.shape}"
    assert predicted_fixations.ndim == 2 and predicted_fixations.shape[1] == 2, \
        f"Input fixations must be of shape [B, 2], but got {predicted_fixations.shape}"
    
    # Calculate Mean Squared Error (MSE) per batch element
    main_fixation_loss = F.mse_loss(predicted_fixations, gt_fixations, reduction='mean') # Scalar loss

    # --- 2. Auxiliary Attention Map Loss (KLDivLoss) ---
    assert model_attn_map.shape == gt_attn_map.shape, \
        f"Model attention map shape {model_attn_map.shape} != GT attention map shape {gt_attn_map.shape}"
    
    # KLDivLoss requires log-probabilities for the input and probabilities for the target.
    # model_attn_map is already softmaxed (probabilities), so we need to apply torch.log.
    # gt_attn_map is also probabilities.
    auxiliary_attn_loss = F.kl_div(
        torch.log(model_attn_map + 1e-9), # Add a small epsilon for numerical stability with log(0)
        gt_attn_map,                      
        reduction='batchmean'             # Averages KLDivLoss over the batch
    )

    # --- Aggregated Total Loss & Metrics Calculation ---
    # Combine main fixation loss with auxiliary attention loss
    total_obs_loss = (1 - aux_loss_weight) * main_fixation_loss + aux_loss_weight * auxiliary_attn_loss

    # --- Metrics for Regression ---
    # RMSE: Root Mean Squared Error (spatial distance)
    # MAE: Mean Absolute Error (spatial distance)
    
    # Calculate squared differences (already done by MSE internally, but we need it per element for RMSE)
    squared_diff = (predicted_fixations - gt_fixations)**2 # [B, 2]
    # Sum squared differences across x and y coordinates, then take sqrt for Euclidean distance
    per_sample_euclidean_distance = torch.sqrt(squared_diff.sum(dim=1)) # [B]

    # Calculate RMSE (Root Mean Squared Error) - The average Euclidean distance per sample
    rmse = per_sample_euclidean_distance.mean() # This is the rmse_alt from previous version

    # Calculate MAE (Mean Absolute Error)
    mae = F.l1_loss(predicted_fixations, gt_fixations, reduction='mean') # Scalar, mean of (abs_diff_x + abs_diff_y) / 2

    results = {
        "obs_loss": total_obs_loss,
        "fixation_loss_mse": main_fixation_loss,
        "auxiliary_loss_attn": auxiliary_attn_loss,
        "rmse": rmse, # Reporting the average Euclidean distance
        "mae": mae,
    }
    return results


def gazeplus_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    print_log_freq: int = 10,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        run_folder: folder to save images to
        epoch: current epoch
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    obs_loss_logger = Logger("obs_loss", "train", window_size=print_log_freq)
    fixation_loss_mse_logger = Logger("fixation_loss_mse", "train", window_size=print_log_freq)
    auxiliary_loss_attn_logger = Logger("auxiliary_loss_attn", "train", window_size=print_log_freq)
    rmse_logger = Logger("rmse", "train", window_size=print_log_freq)
    mae_logger = Logger("mae", "train", window_size=print_log_freq)
    
    loggers = {
        "obs_loss": obs_loss_logger,
        "fixation_loss_mse": fixation_loss_mse_logger,
        "auxiliary_loss_attn": auxiliary_loss_attn_logger,
        "rmse": rmse_logger,
        "mae": mae_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, # [batch_size, 3 * (context_size+1), H, W]
            fixations, # [batch_size, 2]
            _, # [batch_size, num_persons, context_size+1, H, W]
            act_attnmaps, # [batch_size, N*spatial_flatten_len]
            _,
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])  # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)  # [batch_size, 3, H, W] * (context_size+1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # [batch_size, 3 * (context_size+1), H, W]

        # Convert person_masks to person_attention by taking union along num_persons dimension
        act_attnmaps = act_attnmaps.to(device)
        fixations = fixations.to(device)
        prev_fixation = fixations[:, :-1, :]  # [B, C, 2] - previous fixations

        prev_gaze_maps_for_model = render_fixations_to_gaze_maps(
                fixations_batch=prev_fixation,
                H=128, # 使用传入的图像高度
                W=160,  # 使用传入的图像宽度
                sigma=10.0, # 使用传入的高斯核标准差
                device=device
            )
        
        optimizer.zero_grad()

        with autocast():
            predicted_fixation, obs_attnmaps = model(obs_image, prev_gaze_maps_for_model)  # [B, P]
            
            losses = compute_gazeloss(
                predicted_fixations= predicted_fixation,
                gt_fixations=fixations[:, -1, :],
                model_attn_map=obs_attnmaps,       # Model's attention map
                gt_attn_map=act_attnmaps,      # Ground truth attention map (already flattened)
                )
            loss = losses["obs_loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        gaze_log(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=int(num_images_log/3),
            loggers=loggers,
            obs_images=viz_obs_images,
            pred_fixations=predicted_fixation,
            attention_scores=obs_attnmaps,
            gt_fixations=fixations,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def gazeplus_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int = 0,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """

    # 设置模型为评估模式
    model = model.to(device)
    model.eval()

    # 初始化日志器
    loggers = {
        "obs_loss": Logger("obs_loss", "test"),
        "fixation_loss_mse": Logger("fixation_loss_mse", "test"),
        "auxiliary_loss_attn": Logger("auxiliary_loss_attn", "test"),
        "rmse": Logger("rmse", "test"),
        "mae": Logger("mae", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )
    
    with torch.no_grad():
        for i, data in enumerate(tqdm_iter):
            obs_image, fixations, _, act_attnmaps, _ = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Convert person_masks to person_attention by taking union along num_persons dimension
            act_attnmaps = act_attnmaps.to(device)
            fixations = fixations.to(device)
            prev_fixation = fixations[:, :-1, :]  # [B, C, 2] - previous fixations

            prev_gaze_maps_for_model = render_fixations_to_gaze_maps(
                fixations_batch=prev_fixation,
                H=128, # 使用传入的图像高度
                W=160,  # 使用传入的图像宽度
                sigma=10.0, # 使用传入的高斯核标准差
                device=device
            )
            # 前向推理
            predicted_fixation, obs_attnmaps = model(obs_image, prev_gaze_maps_for_model)  # [B, P]

            losses = compute_gazeloss(predicted_fixations=predicted_fixation, gt_fixations=fixations[:, -1, :], model_attn_map=obs_attnmaps, gt_attn_map=act_attnmaps)

            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                gaze_log(
                    i=0,
                    epoch=epoch,
                    num_batches=num_batches,
                    run_folder=run_folder,
                    num_images_log=num_images_log,
                    loggers=loggers,
                    obs_images=viz_obs_images,
                    pred_fixations=predicted_fixation,
                    attention_scores=obs_attnmaps,
                    gt_fixations=fixations,
                    use_wandb=use_wandb,
                    mode="test",
                    use_latest=False,
                    wandb_log_freq=1,
                    print_log_freq=1,
                    image_log_freq=1,
                    wandb_increment_step=False,
                )

    return loggers["obs_loss"].average()