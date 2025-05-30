import wandb
import os
from typing import Optional
from prettytable import PrettyTable

from my_training.my_train_utils import base_train, base_evaluate, cnnaux_train, cnnaux_evaluate, gazeaux_train, gazeaux_evaluate, personaux_train, personaux_evaluate, gazechannel_train, gazechannel_evaluate, personchannel_train, personchannel_evaluate, gazetoken_train, gazetoken_evaluate, persontoken_train, persontoken_evaluate, sel_train, sel_evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


def train_eval_loop(
    train_method: str,
    training_config: dict,
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
        training_config: Optional dict containing stage training configuration:
            {
                "enable_stage_training": bool,
                "current_stage": 1/2/3,
                "stage1_epochs": int,
                "stage2_epochs": int,
                "stage3_epochs": int,
                "stage1_loss_threshold": float,
                "stage2_loss_threshold": float,
                "stage3_loss_threshold": float,
                "early_stopping": bool,
                "patience": int,
                "min_delta": float
            }
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    latest_path = os.path.join(run_folder, f"latest.pth")
    best_test_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = os.path.join(run_folder, "best.pth")

    for epoch in range(current_epoch, epochs):
        if train_model:
            training_stage = None
            if training_config and training_config["enable_stage_training"]:
                stage = training_config["current_stage"]
                training_stage = {
                    "stage": stage,
                    "use_gt_masks": (stage == 1),
                    "freeze_selector": (stage == 1),
                    "freeze_transformer": (stage == 2)
                }
            if train_method == "sel":
                sel_train(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    transform=transform,
                    device=device,
                    run_folder=run_folder,
                    training_stage=training_stage,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    wandb_log_freq=wandb_log_freq,
                    image_log_freq=image_log_freq,
                    num_images_log=num_images_log,
                    use_wandb=use_wandb,
                )
            elif train_method == "base":
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
            elif train_method == "gazeaux":
                gazeaux_train(
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
            elif train_method == "personaux":
                personaux_train(
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
            elif train_method == "gazechannel":
                gazechannel_train(
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
            elif train_method == "personchannel":
                personchannel_train(
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
            elif train_method == "gazetoken":
                gazetoken_train(
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
            elif train_method == "persontoken":
                persontoken_train(
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

        # Evaluation
        test_loss = None
        if train_method == "sel":
            test_loss = sel_evaluate(
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
        elif train_method == "base":
            test_loss = base_evaluate(
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
            test_loss = cnnaux_evaluate(
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
        elif train_method == "gazeaux":
            test_loss = gazeaux_evaluate(
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
        elif train_method == "personaux":
            test_loss = personaux_evaluate(
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
        elif train_method == "gazechannel":
            test_loss = gazechannel_evaluate(
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
        elif train_method == "personchannel":
            test_loss = personchannel_evaluate(
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
        elif train_method == "gazetoken":
            test_loss = gazetoken_evaluate(
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
        elif train_method == "persontoken":
            test_loss = persontoken_evaluate(
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

        # Early stopping check
        if training_config and training_config.get("early_stopping", False):
            if test_loss < best_test_loss - training_config.get("min_delta", 1e-4):
                # 有显著改善
                best_test_loss = test_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                checkpoint = {
                    "epoch": epoch,
                    "model": model,
                    "optimizer": optimizer,
                    "test_loss": test_loss,
                    "scheduler": scheduler,
                    "training_config": training_config
                }
                torch.save(checkpoint, best_model_path)
                print(f"Saved best model with test_loss {test_loss:.4f} at epoch {epoch}")
            else:
                epochs_without_improvement += 1

            # Check if we should stop
            patience = training_config.get("patience", 10)
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                print(f"Best performance was {best_test_loss:.4f} at epoch {best_epoch}")

                # Load best model
                best_checkpoint = torch.load(best_model_path)
                load_model(model, best_checkpoint)
                break

        # Handle stage transitions
        if training_config and training_config["enable_stage_training"]:
            stage = training_config["current_stage"]
            if stage == 1 and (
                epoch >= training_config["stage1_epochs"] or
                test_loss < training_config["stage1_loss_threshold"]
            ):
                print(f"Moving to stage 2 at epoch {epoch}")
                training_config["current_stage"] = 2
                optimizer.param_groups[0]['lr'] = optimizer.defaults['lr']
            
            elif stage == 2 and (
                epoch >= training_config["stage2_epochs"] + training_config["stage1_epochs"] or
                test_loss < training_config["stage2_loss_threshold"]
            ):
                print(f"Moving to stage 3 at epoch {epoch}")
                training_config["current_stage"] = 3
                optimizer.param_groups[0]['lr'] = optimizer.defaults['lr']

        # Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "test_loss": test_loss,
            "scheduler": scheduler,
            "training_config": training_config
        }
        # log average eval loss
        wandb.log({}, commit=False)

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