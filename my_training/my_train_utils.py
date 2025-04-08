import wandb
import numpy as np
import tqdm
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from my_data.my_data_utils import ts2np
from my_training.my_visualize_utils import bc_visualize


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


def bc_log_data(
    i,
    epoch,
    num_batches,
    run_folder,
    num_images_log,
    loggers,
    obs_image,
    action_pred,
    obs_features,
    obs_features_grad,
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
            batch_obs_images=ts2np(obs_image),
            batch_pred_waypoints=ts2np(action_pred),
            batch_label_waypoints=ts2np(action_label),
            obs_features=ts2np(obs_features),
            obs_features_grads=ts2np(obs_features_grad),
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
    model.train()
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
            _, # [batch_size, (context_size+1) * H/32 * W/32] 
            action_label,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        action_label = action_label.to(device)

        optimizer.zero_grad()
      
        action_pred, obs_features, attention_scores = model(obs_image)

        losses = compute_baseloss(action_label=action_label, action_pred=action_pred)

        losses["action_loss"].backward()

        # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        )
        # 选取最后一帧图像的特征图和梯度图
        viz_obs_feature = obs_features[-1]
        viz_obs_feature_grad = obs_features_grad[-1]

        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        bc_log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            action_pred=action_pred,
            obs_features=viz_obs_feature,
            obs_features_grad=viz_obs_feature_grad,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")


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
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    # 第一部分：无梯度模式下，正常评估并记录损失
    last_data = None

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        obs_image, _, action_label = data

        # 图像分割通道并处理
        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
        obs_image = torch.cat(obs_images, dim=1)
        # 确保输入开启梯度追踪
        obs_image.requires_grad_(True)

        action_label = action_label.to(device)

        # 前向推理
        action_pred, _, _ = model(obs_image)

        # 计算损失并记录（注意：此处直接用 .item() 记录数值）
        losses = compute_baseloss(action_label=action_label, action_pred=action_pred)
        for key, value in losses.items():
            if key in loggers:
                loggers[key].log_data(value.item())

        # 保存最后一个batch的数据，用于后续Grad-CAM可视化
        last_data = (obs_image, action_label, viz_obs_image, action_pred)
        
    # 第二部分：对最后一个batch生成Grad-CAM可视化（需要开启梯度）
    if last_data is not None:
        # 确保 obs_image 具有梯度追踪
        obs_image, action_label, viz_obs_image, _ = last_data
        obs_image = obs_image.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            # 重新前向推理，并获取特征图和梯度
            action_pred, obs_features, attention_scores= model(obs_image)

            # 计算损失并回传梯度
            loss = compute_baseloss(action_label=action_label, action_pred=action_pred)
            loss["action_loss"].backward()
            
            # 取回obs_features的梯度
            obs_features_grad = model._grad_obs_features

            if obs_features_grad is not None:
                obs_features_grad = obs_features_grad.contiguous().view(
                model.context_size + 1,
                -1,
                obs_features.shape[2],
                obs_features.shape[3],
                obs_features.shape[4]
            )

            # 选取最后一帧图像的特征图和梯度图
            viz_obs_feature = obs_features[-1]
            viz_obs_feature_grad = obs_features_grad[-1]
            
            # 记录可视化和其他日志信息
            bc_log_data(
                i=0,
                epoch=epoch,
                num_batches=num_batches,
                run_folder=run_folder,
                num_images_log=num_images_log,
                loggers=loggers,
                obs_image=viz_obs_image,
                action_pred=action_pred,
                obs_features=viz_obs_feature,
                obs_features_grad=viz_obs_feature_grad,
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

    # Gaze attention auxiliary loss (KL divergence)
    auxiliary_loss = F.kl_div(
        F.log_softmax(gaze_use_map, dim=1),
        F.softmax(gaze_map, dim=1),
        reduction='batchmean',
        log_target=False,
    )

    # Combine losses with weight
    alpha = 0.2
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
    model.train()
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
            action_label,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device).squeeze(-1)
        action_label = action_label.to(device)

        optimizer.zero_grad()
      
        action_pred, obs_features, attention_scores, gaze_use_map = model(obs_image)
        assert gaze_attention.shape == gaze_use_map.shape, f"Shape mismatch: gaze_attention {gaze_attention.shape} != gaze_use_map {gaze_use_map.shape}"

        losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention, gaze_use_map=gaze_use_map)

        losses["total_loss"].backward()

        # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        )
        # 选取最后一帧图像的特征图和梯度图
        viz_obs_feature = obs_features[-1]
        viz_obs_feature_grad = obs_features_grad[-1]

        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        bc_log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            action_pred=action_pred,
            obs_features=viz_obs_feature,
            obs_features_grad=viz_obs_feature_grad,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")


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
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    # 第一部分：无梯度模式下，正常评估并记录损失
    last_data = None

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        obs_image, gaze_attention, action_label = data

        # 图像分割通道并处理
        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
        obs_image = torch.cat(obs_images, dim=1)
        # 确保输入开启梯度追踪
        obs_image.requires_grad_(True)

        gaze_attention = gaze_attention.to(device).squeeze(-1)
        action_label = action_label.to(device)

        # 前向推理
        action_pred, _, _, gaze_use_map = model(obs_image)

        # 计算损失并记录（注意：此处直接用 .item() 记录数值）
        losses = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention, gaze_use_map=gaze_use_map)
        for key, value in losses.items():
            if key in loggers:
                loggers[key].log_data(value.item())

        # 保存最后一个batch的数据，用于后续Grad-CAM可视化
        last_data = (obs_image, action_label, viz_obs_image, action_pred)
        
    # 第二部分：对最后一个batch生成Grad-CAM可视化（需要开启梯度）
    if last_data is not None:
        # 确保 obs_image 具有梯度追踪
        obs_image, action_label, viz_obs_image, _ = last_data
        obs_image = obs_image.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            # 重新前向推理，并获取特征图和梯度
            action_pred, obs_features, attention_scores, gaze_use_map= model(obs_image)

            # 计算损失并回传梯度
            loss = compute_cnnaux_loss(action_label=action_label, action_pred=action_pred, gaze_map=gaze_attention, gaze_use_map=gaze_use_map)
            loss["total_loss"].backward()
            
            # 取回obs_features的梯度
            obs_features_grad = model._grad_obs_features

            if obs_features_grad is not None:
                obs_features_grad = obs_features_grad.contiguous().view(
                model.context_size + 1,
                -1,
                obs_features.shape[2],
                obs_features.shape[3],
                obs_features.shape[4]
            )

            # 选取最后一帧图像的特征图和梯度图
            viz_obs_feature = obs_features[-1]
            viz_obs_feature_grad = obs_features_grad[-1]
            
            # 记录可视化和其他日志信息
            bc_log_data(
                i=0,
                epoch=epoch,
                num_batches=num_batches,
                run_folder=run_folder,
                num_images_log=num_images_log,
                loggers=loggers,
                obs_image=viz_obs_image,
                action_pred=action_pred,
                obs_features=viz_obs_feature,
                obs_features_grad=viz_obs_feature_grad,
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

    # Token attention auxiliary loss (KL divergence)
    token_aux_loss = F.kl_div(
        gaze_use_vector.log(),  # [batch_size, seq_len]
        gaze_map_normalized,    # [batch_size, H*W]
        reduction='batchmean',
        log_target=False,
    )

    # Combine losses with weight
    alpha = 0.2
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
    model.train()
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
            action_label,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        gaze_attention = gaze_attention.to(device).squeeze(-1)
        action_label = action_label.to(device)

        optimizer.zero_grad()
      
        action_pred, obs_features, attention_scores = model(obs_image)

        losses = compute_tokenaux_loss(
            action_label=action_label,
            action_pred=action_pred,
            gaze_map=gaze_attention,
            attention_scores=attention_scores
        )

        losses["total_loss"].backward()

        # 取回obs_features的梯度
        obs_features_grad = model._grad_obs_features

        if obs_features_grad is not None:
            obs_features_grad = obs_features_grad.contiguous().view(
            model.context_size + 1,
            -1,
            obs_features.shape[2],
            obs_features.shape[3],
            obs_features.shape[4]
        )
        # 选取最后一帧图像的特征图和梯度图
        viz_obs_feature = obs_features[-1]
        viz_obs_feature_grad = obs_features_grad[-1]

        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        bc_log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            run_folder=run_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            action_pred=action_pred,
            obs_features=viz_obs_feature,
            obs_features_grad=viz_obs_feature_grad,
            attention_scores=attention_scores,
            action_label=action_label,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
        )
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")


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
    model.eval()

    # 初始化日志器
    loggers = {
        "action_loss": Logger("action_loss", "test"),
        "auxiliary_loss": Logger("auxiliary_loss", "test"),
        "total_loss": Logger("total_loss", "test"),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test"),
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    # 第一部分：无梯度模式下，正常评估并记录损失
    last_data = None

    tqdm_iter = tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Evaluating for epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        obs_image, gaze_attention, action_label = data

        # 图像分割通道并处理
        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = obs_images[-1]
        obs_images = [transform(obs_img).to(device) for obs_img in obs_images]
        obs_image = torch.cat(obs_images, dim=1)
        # 确保输入开启梯度追踪
        obs_image.requires_grad_(True)

        gaze_attention = gaze_attention.to(device).squeeze(-1)
        action_label = action_label.to(device)

        # 前向推理
        action_pred, _, attention_scores = model(obs_image)

        # 计算损失并记录（注意：此处直接用 .item() 记录数值）
        losses = compute_tokenaux_loss(
            action_label=action_label,
            action_pred=action_pred,
            gaze_map=gaze_attention,
            attention_scores=attention_scores
        )
        for key, value in losses.items():
            if key in loggers:
                loggers[key].log_data(value.item())

        # 保存最后一个batch的数据，用于后续Grad-CAM可视化
        last_data = (obs_image, action_label, viz_obs_image, action_pred)
        
    # 第二部分：对最后一个batch生成Grad-CAM可视化（需要开启梯度）
    if last_data is not None:
        # 确保 obs_image 具有梯度追踪
        obs_image, action_label, viz_obs_image, _ = last_data
        obs_image = obs_image.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            # 重新前向推理，并获取特征图和梯度
            action_pred, obs_features, attention_scores = model(obs_image)

            # 计算损失并回传梯度
            loss = compute_tokenaux_loss(
                action_label=action_label,
                action_pred=action_pred,
                gaze_map=gaze_attention,
                attention_scores=attention_scores
            )
            loss["total_loss"].backward()
            
            # 取回obs_features的梯度
            obs_features_grad = model._grad_obs_features

            if obs_features_grad is not None:
                obs_features_grad = obs_features_grad.contiguous().view(
                model.context_size + 1,
                -1,
                obs_features.shape[2],
                obs_features.shape[3],
                obs_features.shape[4]
            )

            # 选取最后一帧图像的特征图和梯度图
            viz_obs_feature = obs_features[-1]
            viz_obs_feature_grad = obs_features_grad[-1]
            
            # 记录可视化和其他日志信息
            bc_log_data(
                i=0,
                epoch=epoch,
                num_batches=num_batches,
                run_folder=run_folder,
                num_images_log=num_images_log,
                loggers=loggers,
                obs_image=viz_obs_image,
                action_pred=action_pred,
                obs_features=viz_obs_feature,
                obs_features_grad=viz_obs_feature_grad,
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