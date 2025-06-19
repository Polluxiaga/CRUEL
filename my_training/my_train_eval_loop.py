import wandb
import os
from typing import Optional, Dict 
from prettytable import PrettyTable

from my_training.my_train_utils import (
    base_train, base_evaluate,
    cnnaux_train, cnnaux_evaluate,
    gazeaux_train, gazeaux_evaluate,
    personaux_train, personaux_evaluate,
    gazechannel_train, gazechannel_evaluate,
    personchannel_train, personchannel_evaluate,
    gazetoken_train, gazetoken_evaluate,
    persontoken_train, persontoken_evaluate,
    dumobs_train, dumobs_evaluate,
    obs_train, obs_evaluate
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


def load_model(model: nn.Module, checkpoint: Dict) -> None:
    """
    Load model state_dict from checkpoint.
    Assumes checkpoint is a dict with "model_state_dict" key.
    """
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint dictionary must contain 'model_state_dict' key.")
    
    # Load the state_dict, allowing for partial matches (strict=False)
    # This is helpful if model architecture changes slightly or if loading from DataParallel
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Model state dictionary loaded successfully.")


def train_eval_loop(
    train_method: str,
    training_config: dict,
    train_model: bool,
    # obs_model and act_model are passed, but only one will be 'active' for this loop based on train_method
    obs_model: Optional[nn.Module], # Optional, as it might not be relevant for all methods
    act_model: Optional[nn.Module], # Optional, as it might not be relevant for all methods
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
    Train and evaluate the model for several epochs with early stopping support.

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

    # Determine which model is the primary target for *this specific* train_eval_loop call
    if train_method == "obs" or train_method == "dumobs":
        primary_model = obs_model
        # Use a specific path for obs_model best checkpoint
        best_model_path = os.path.join(run_folder, "best_obs_model.pth")
        latest_path = os.path.join(run_folder, "latest_obs.pth")
    else: # All other methods train the act_model
        primary_model = act_model
        # Use a specific path for act_model best checkpoint
        best_model_path = os.path.join(run_folder, "best_act_model.pth")
        latest_path = os.path.join(run_folder, "latest_act.pth")
    
    if primary_model is None:
        raise ValueError(f"Primary model is None for train_method: {train_method}")
    
    primary_model = primary_model.to(device)

    best_test_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0


    for epoch in range(current_epoch, epochs):
        if train_model:
            if train_method == "base":
                base_train(
                    model=act_model,
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
                    model=act_model,
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
            elif train_method == "gazeaux":
                gazeaux_train(
                    model=act_model,
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
            elif train_method == "personaux":
                personaux_train(
                    model=act_model,
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
            elif train_method == "gazechannel":
                gazechannel_train(
                    model=act_model,
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
            elif train_method == "personchannel":
                personchannel_train(
                    model=act_model,
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
            elif train_method == "gazetoken":
                gazetoken_train(
                    model=act_model,
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
            elif train_method == "persontoken":
                persontoken_train(
                    model=act_model,
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
            elif train_method == "dumobs":
                dumobs_train(
                    model=obs_model,
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
            elif train_method == "obs":
                obs_train(
                    model=obs_model,
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

        # Evaluation
        test_loss = None
        if train_method == "base":
            test_loss = base_evaluate(
                model=act_model,
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
            test_loss = cnnaux_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "gazeaux":
            test_loss = gazeaux_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "personaux":
            test_loss = personaux_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "gazechannel":
            test_loss = gazechannel_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "personchannel":
            test_loss = personchannel_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "gazetoken":
            test_loss = gazetoken_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "persontoken":
            test_loss = persontoken_evaluate(
                model=act_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "dumobs":
            test_loss = dumobs_evaluate(
                model=obs_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )
        elif train_method == "obs":
            test_loss = obs_evaluate(
                model=obs_model,
                dataloader=test_loader,
                transform=transform,
                device=device,
                run_folder=run_folder,
                epoch=epoch,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

        # Early stopping check
        if training_config.get("early_stopping", False) and test_loss is not None:
            if test_loss < best_test_loss - training_config.get("min_delta", 1e-4):
                # 有显著改善
                best_test_loss = test_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": primary_model.state_dict(), # Correctly save state_dict
                    "optimizer_state_dict": optimizer.state_dict(), # Correctly save state_dict
                    "test_loss": test_loss,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None, # Correctly save state_dict
                    "training_config": training_config # Save the early stopping config
                }
                torch.save(checkpoint, best_model_path)
                print(f"Saved best model ({os.path.basename(best_model_path)}) with test_loss {test_loss:.4f} at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            # Check if we should stop
            patience = training_config.get("patience", 10)
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                print(f"Best performance was {best_test_loss:.4f} at epoch {best_epoch}")

                # Load best model
                if os.path.exists(best_model_path):
                    best_checkpoint = torch.load(best_model_path, map_location=device)
                    load_model(primary_model, best_checkpoint) # Use the updated load_model
                    print(f"Loaded best model from {os.path.basename(best_model_path)} for final state.")
                break

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        wandb.log({
            "test_loss": test_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        # Save latest checkpoint for this specific training run
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": primary_model.state_dict(), # Correctly save state_dict
            "optimizer_state_dict": optimizer.state_dict(), # Correctly save state_dict
            "test_loss": test_loss,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None, # Correctly save state_dict
            "training_config": training_config # Save the early stopping config
        }
        torch.save(checkpoint, latest_path)
        # Save epoch-numbered checkpoint (optional, but good for tracking)
        torch.save(checkpoint, os.path.join(run_folder, f"{train_method}_epoch_{epoch}.pth")) 

        # Clear GPU cache after each epoch
        torch.cuda.empty_cache()

    # Final log commit (if any pending) - although previous logs were committed per epoch
    wandb.log({}, commit=True) # Ensure final commit if something was batched or missed
    print("Training loop finished for method:", train_method)
    print()


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