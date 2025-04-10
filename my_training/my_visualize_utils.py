import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import wandb
import seaborn as sns

VIZ_IMAGE_SIZE = (700, 400)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def np2img(arr: np.ndarray) -> Image:
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,  # [pred_waypoints, label_waypoints]
    traj_colors: list = [CYAN, MAGENTA],
    traj_labels: list = ["prediction", "ground truth"],
):
    """
    Plot trajectories and points.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        traj_labels: list of labels for trajectories
    """

    # Define the start position (0, 0)
    start_pos = (0, 0)

    for i, traj in enumerate(list_trajs):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=traj_colors[i],
            label=traj_labels[i],
            alpha=1.0,
            marker="o",
        )

        ax.plot(
            [start_pos[0], traj[0, 0]],  # X coordinates of the line: (0, 1st point of traj)
            [start_pos[1], traj[0, 1]],  # Y coordinates of the line: (0, 1st point of traj)
            color=traj_colors[i],  # Use the same color as the trajectory
            alpha=1.0,
        )
    
    ax.set_aspect("equal", "box")
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)


def compute_gradcam_heatmap(features: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """
    根据特征图和梯度计算Grad-CAM热力图。
    Args:
        features: 形状为 (C, H, W) 的特征图
        grads: 与features形状相同的梯度
    Returns:
        热力图，范围 [0, 1]
    """
    # 对每个通道计算梯度均值作为权重
    weights = np.mean(grads, axis=(1, 2), keepdims=True)  # shape: (C, 1, 1)
    cam = np.sum(weights * features, axis=0)  # 聚合特征图，shape: (H, W)
    # 仅保留正值，并归一化
    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    return cam       


def bc_visualize(
    batch_obs_images: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    obs_features: np.ndarray,
    obs_features_grads: np.ndarray,
    attention_scores: np.ndarray,
    mode: str,
    save_folder: str,
    epoch: int,
    num_images_log: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    Compare predicted path with the groundtruth path of waypoints using egocentric visualization. This visualization is for the last batch in the dataset.

    Args:
        batch_obs_images (np.ndarray): batch of observation images [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): batch of goal images [batch_size, height, width, channels]
        dataset_names: indices corresponding to the dataset name
        batch_pred_waypoints (np.ndarray): batch of predicted waypoints [batch_size, learn_traj_pred, 2]
        batch_label_waypoints (np.ndarray): batch of label waypoints [batch_size, learn_traj_pred, 2]
        obs_features (np.ndarray): batch of observation features [batch_size*(context_size+1), num_features, height, width]
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
    """

    assert (
        len(batch_obs_images)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
    )

    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", mode, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    batch_size = batch_obs_images.shape[0]
    wandb_list = []

    for i in range(min(batch_size, num_images_log)):
        obs_img = np2img(batch_obs_images[i])
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]
        obs_feature = obs_features[i]
        obs_feature_grad = obs_features_grads[i]
        attention_score = attention_scores[i]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        bc_draw(
            obs_img,
            pred_waypoints,
            label_waypoints,
            obs_feature,
            obs_feature_grad,
            attention_score,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{mode}_action_prediction": wandb_list}, commit=False)


def bc_draw(
    obs_img,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    obs_feature: np.ndarray,
    obs_feature_grad: Optional[np.ndarray],
    attention_scores: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    使用Grad-CAM加权特征生成热力图，并与观测图叠加，同时绘制预测轨迹和标注轨迹.

    Args:
        obs_img: PIL Image格式的观测图
        dataset_name: 数据集名称
        pred_waypoints: 预测轨迹，形状 [N,2]
        label_waypoints: 标注轨迹，形状 [N,2]
        obs_feature: 观测图对应的特征图 (C, H, W)
        obs_feature_grad: 特征图的梯度信息，形状与obs_feature相同
        save_path: 保存图像的路径
        display: 是否显示图像
    """
    features = (
        obs_feature
        if isinstance(obs_feature, np.ndarray)
        else obs_feature.detach().cpu().numpy()
    )
    
    grads = (
        obs_feature_grad
        if isinstance(obs_feature_grad, np.ndarray)
        else obs_feature_grad.detach().cpu().numpy()
    )

    # 计算Grad-CAM热力图 (H, W)
    cam = compute_gradcam_heatmap(features, grads)
    heatmap = cv2.resize(cam, (obs_img.size[0], obs_img.size[1]))
    heatmap = np.uint8(255 * (1 - heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加热力图到观测图
    obs_img_np = np.array(obs_img)
    heatmap_img = cv2.addWeighted(obs_img_np, 0.6, heatmap, 0.4, 0)

    # 获取原图宽高比
    img_height, img_width = obs_img_np.shape[:2]
    aspect_ratio = img_width / img_height

    # 绘制预测轨迹和标注轨迹
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 17.5 / aspect_ratio))

    plot_trajs_and_points(
        ax[0],
        [pred_waypoints, label_waypoints],
        traj_colors=[CYAN, MAGENTA],
    )
    ax[0].set_title("Action Prediction")

    ax[1].imshow(heatmap_img)
    ax[1].set_title("Current with Grad-CAM")

    sns.heatmap(attention_scores, cmap="viridis", annot=False, fmt=".2f", ax=ax[2], square=True)
    ax[2].set_title("Attention Map")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if not display:
        plt.close(fig)