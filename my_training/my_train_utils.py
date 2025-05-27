import wandb
import numpy as np
import tqdm
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

from my_data.my_data_utils import ts2np
from my_training.my_visualize_utils import bc_visualize


def base_collate_fn(batch):
    """
    Basic collate function that returns only necessary elements for base model:
    - obs_images: observation images
    - gaze_maps: gaze attention maps
    - action_list: action labels
    """
    obs_images, gaze_maps, _, _, action_list = zip(*batch)
    
    obs_batch = torch.stack(obs_images, dim=0)
    gaze_batch = torch.stack(gaze_maps, dim=0)
    action_batch = torch.stack(action_list, dim=0)
    
    # Return dummy tensors for mask and select to maintain compatibility
    batch_size = obs_batch.shape[0]
    dummy_mask = torch.zeros((batch_size, 1, 1, 1, 1), dtype=torch.bool)
    dummy_select = torch.zeros((batch_size, 1), dtype=torch.bool)
    
    return obs_batch, gaze_batch, dummy_mask, dummy_select, action_batch


def person_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
      obs_image:  Tensor [3*N, H, W]
      mask_imgs:  Tensor [P_i, N, H, W]
      select_labels: Tensor [P_i]
      action_labels: Tensor [3, 2]
    """

    # 1) 解包
    obs_images, _, mask_list, select_list, action_list = zip(*batch)
    B = len(batch)

    # 2) Stack obs_images and action_labels
    obs_batch = torch.stack(obs_images, dim=0)  # [B, 3*(C+1), H, W]
    action_batch = torch.stack(action_list, dim=0)    # [B, 3, 2]

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

    return obs_batch, None, mask_batch, select_batch, action_batch, invalid_flag


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


def log_data(
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
        bc_visualize(
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
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similarity,
    }
    return results


def base_train(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            _,
            _,
            action_label,
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

        log_data(
            i=i,
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
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def base_evaluate(
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
            obs_image, _, _, _, action_label = data

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
                log_data(
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

    # Action prediction loss
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # Cosine similarity metric
    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

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
    }
    return results


def cnnaux_train(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            _,
            _,
            action_label,
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3]) # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device)
        output_h = gaze_attention.shape[2] // 32
        output_w = gaze_attention.shape[3] // 32
        gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
        gaze_attention_flattened = gaze_attention_pooled.view(gaze_attention_pooled.shape[0], -1)

        action_label = action_label.to(device)

        optimizer.zero_grad()
        
        with autocast():
            action_pred, attention_scores, gaze_use_map = model(obs_image)
            losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention_flattened, gaze_use_map=gaze_use_map)
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

        log_data(
            i=i,
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
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def cnnaux_evaluate(
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
            obs_image, gaze_attention, _, _, action_label = data

            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            gaze_attention = gaze_attention.to(device)
            output_h = gaze_attention.shape[2] // 32
            output_w = gaze_attention.shape[3] // 32
            gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
            gaze_attention_flattened = gaze_attention_pooled.view(gaze_attention_pooled.shape[0], -1)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores, gaze_use_map = model(obs_image)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention_flattened, gaze_use_map=gaze_use_map)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())
    
            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                log_data(
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

def compute_tokenaux_loss(
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

    # Action prediction loss
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # Cosine similarity metric
    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))

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
    }
    return results


def tokenaux_train(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            _,
            _,
            action_label,
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device)
        output_h = gaze_attention.shape[2] // 32
        output_w = gaze_attention.shape[3] // 32
        gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
        gaze_attention_flattened = gaze_attention_pooled.contiguous().view(gaze_attention_pooled.shape[0], -1)

        action_label = action_label.to(device)

        optimizer.zero_grad()

        with autocast():
            action_pred, attention_scores = model(obs_image)
            losses = compute_tokenaux_loss(
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

        log_data(
            i=i,
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
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def tokenaux_evaluate(
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
            obs_image, gaze_attention, _, _, action_label = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            gaze_attention = gaze_attention.to(device)
            output_h = gaze_attention.shape[2] // 32
            output_w = gaze_attention.shape[3] // 32
            gaze_attention_pooled = F.adaptive_avg_pool2d(gaze_attention, (output_h, output_w))
            gaze_attention_flattened = gaze_attention_pooled.contiguous().view(gaze_attention_pooled.shape[0], -1)

            action_label = action_label.to(device)

            # 前向推理
            action_pred, attention_scores = model(obs_image)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_tokenaux_loss(
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
                log_data(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "auxiliary_loss": auxiliary_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
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
            losses = compute_tokenaux_loss(
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

        log_data(
            i=i,
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
            obs_image, _, person_masks, select_labels, action_label, invalid = data
    
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
            losses = compute_tokenaux_loss(
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
                log_data(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            person_masks, # [batch_size, num_persons, context_size+1, H, W]
            select_labels, # [batch_size, num_persons]
            action_label,
            invalid, # [batch_size, num_persons]
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

        log_data(
            i=i,
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
            obs_image, _, person_masks, select_labels, action_label, invalid = data
    
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
                log_data(
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
    
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            _, # [batch_size, num_persons, context_size+1, H, W]
            _, # [batch_size, num_persons]
            action_label,
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

        log_data(
            i=i,
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
            obs_image, gaze_maps, _, _, action_label= data
    
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
                log_data(
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

def compute_selloss(
    select_label: torch.Tensor,
    action_label: torch.Tensor,
    select_pred: torch.Tensor,
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
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        action_pred, action_label, dim=-1
    ))
    
    # Selection loss
    log_probs = F.log_softmax(select_pred, dim=1)  # shape: (B, P_max)
    select_loss = F.nll_loss(log_probs, select_label, reduction="mean")  # select_label: (B,)

    # Combine losses with weight
    alpha = 0.5
    total_loss = (1 - alpha) * action_loss + alpha * select_loss

    results = {
        "action_loss": action_loss,
        "select_loss": select_loss,
        "total_loss": total_loss,
        "action_waypts_cos_sim": action_waypts_cos_similarity,
    }
    return results


def sel_train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    run_folder: str,
    epoch: int,
    training_stage: dict = None,  # 新增参数
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
        training_stage: Dict containing stage configuration:
            {
                "stage": 1/2/3,  # Current training stage
                "use_gt_masks": bool,  # Whether to use ground truth masks
                "freeze_selector": bool,  # Whether to freeze selector
                "freeze_transformer": bool  # Whether to freeze transformer
            }
    """
    model = model.to(device)
    model.train()
    
    # Handle training stages if provided
    if training_stage is not None:
        # Stage 1: Train transformer with GT masks
        if training_stage["freeze_selector"]:
            for param in model.selector.parameters():
                param.requires_grad = False
        # Stage 2: Train selector
        if training_stage["freeze_transformer"]:
            for param in model.transformer.parameters():
                param.requires_grad = False
    
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    select_loss_logger = Logger("select_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    
    loggers = {
        "action_loss": action_loss_logger,
        "select_loss": select_loss_logger,
        "total_loss": total_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
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
            _, # [batch_size, (context_size+1) * H/32 * W/32] 
            person_mask, # [batch_size, num_persons, context_size+1, H, W]
            select_label, # [batch_size, num_persons]
            action_label, # [batch_size, len_traj_pred, 2]
            invalid, # [batch_size, num_persons]
        ) = data

        viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3]) # [batch_size, context_size+1, 3, H, W]

        obs_images = torch.split(obs_image, 3, dim=1)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        person_masks = person_mask.to(device).bool()

        select_label = select_label.to(device)
        action_label = action_label.to(device)
        invalid = invalid.to(device)

        optimizer.zero_grad()
      
        if training_stage is not None and training_stage["use_gt_masks"]:
            select_pred, action_pred, attention_scores = model(
                obs_image, person_masks, invalid)
        else:
            # Get masks from selector if not using GT masks
            with torch.no_grad():
                select_logits = model.selector(obs_image)
                pred_masks = model.get_masks_from_logits(select_logits)
            select_pred, action_pred, attention_scores = model(
                obs_image, pred_masks, invalid)

        losses = compute_selloss(select_label=select_label, action_label=action_label, select_pred=select_pred, action_pred=action_pred)

        losses["total_loss"].backward()

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

        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        log_data(
            i=i,
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
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )


def sel_evaluate(
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
        "select_loss": Logger("select_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
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
            (
                obs_image, # [batch_size, 3 * (context_size+1), H, W]
                _, # [batch_size, (context_size+1) * H/32 * W/32] 
                person_mask, # [batch_size, num_persons, context_size+1, H, W]
                select_label, # [batch_size, num_persons]
                action_label, # [batch_size, len_traj_pred, 2]
                invalid, # [batch_size, num_persons]
            ) = data
    
            viz_obs_images = obs_image.view(obs_image.shape[0], -1, 3, obs_image.shape[2], obs_image.shape[3])

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            person_masks = person_mask.to(device).bool()

            select_label = select_label.to(device)
            action_label = action_label.to(device)
            invalid = invalid.to(device)

            # 前向推理
            select_pred, action_pred, attention_scores = model(obs_image,person_masks, invalid)

            # 计算损失并记录（注意：此处直接用 .item() 记录数值）
            losses = compute_selloss(select_label=select_label, action_label=action_label,select_pred=select_pred, action_pred=action_pred)
            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

            # 只对最后一个batch进行可视化
            if i == num_batches - 1: 
                log_data(
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