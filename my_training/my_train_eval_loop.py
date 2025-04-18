import wandb
import os
from typing import Optional
from prettytable import PrettyTable

from my_training.my_train_utils import base_train, base_evaluate, cnnaux_train, cnnaux_evaluate, tokenaux_train, tokenaux_evaluate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


def train_eval_loop(
    train_method: str,
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_loader: DataLoader,
    test_loader: DataLoader,
    transform: transforms,
    epochs: int,
    device: torch.device,
    run_folder: str,
    wandb_log_freq: int = 1,
    print_log_freq: int = 1,
    image_log_freq: int = 1,
    num_images_log: int = 8,
    current_epoch: int = 0,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        run_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    latest_path = os.path.join(run_folder, f"latest.pth")

    for epoch in range(current_epoch, epochs):
        if train_model:
            print(
                f"Start ViNT Training Epoch {epoch}/{epochs - 1}"
            )
            if train_method == "base":
                base_train(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    transform=transform,
                    device=device,
                    run_folder=run_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    wandb_log_freq=wandb_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    use_wandb=use_wandb,
                )
            elif train_method == "cnnaux":
                cnnaux_train(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    transform=transform,
                    device=device,
                    run_folder=run_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    wandb_log_freq=wandb_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    use_wandb=use_wandb,
                )
            elif train_method == "tokenaux":
                tokenaux_train(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    transform=transform,
                    device=device,
                    run_folder=run_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    wandb_log_freq=wandb_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    use_wandb=use_wandb,
                )

        
        print(
            f"Start ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
        )
        if train_method == "base":
            action_test_loss = base_evaluate(
                model=model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "cnnaux":
            action_test_loss = cnnaux_evaluate(
                model=model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "tokenaux":
            action_test_loss = tokenaux_evaluate(
                model=model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "action_test_loss": action_test_loss,
            "scheduler": scheduler
        }
        # log average eval loss
        wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(action_test_loss)
            else:
                scheduler.step()
        wandb.log({
            "action_test_loss": action_test_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        numbered_path = os.path.join(run_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

        # 每个epoch结束后清空GPU缓存
        torch.cuda.empty_cache()

    # Flush the last set of eval logs
    wandb.log({})
    print()


def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=False)
    except AttributeError as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params