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
    """Convert numpy array to PIL Image.
    
    Args:
        arr: numpy array of shape (C, H, W) or (H, W, C)
        
    Returns:
        PIL Image of size VIZ_IMAGE_SIZE
    """
    # Check array shape
    if len(arr.shape) != 3:
        raise ValueError(f"Expected 3D array (C,H,W) or (H,W,C), got shape {arr.shape}")
    
    # If array is in (C,H,W) format, transpose to (H,W,C)
    if arr.shape[0] == 3:  # Channels-first format
        arr = np.transpose(arr, (1, 2, 0))
    
    # Convert to uint8 and create PIL Image
    img = Image.fromarray(np.uint8(255 * arr))
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


def bc_draw(
    obs_imgs: list,  # List of 6 PIL Images
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    obs_features: np.ndarray,  # [T, C, H, W]
    obs_features_grad: Optional[np.ndarray],  # [T, C, H, W]
    attention_scores: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    创建3x3的可视化布局:
    - 第一行：轨迹预测(1列) + 注意力图(2-3列)
    - 第二行和第三行：6张观测图的Grad-CAM热力图
    """
    # 创建3x3布局
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    
    # 第一行第一列：轨迹预测
    plot_trajs_and_points(
        axes[0, 0],
        [pred_waypoints, label_waypoints],
        traj_colors=[CYAN, MAGENTA],
    )
    axes[0, 0].set_title("Action Prediction")

    # 第一行第二、三列：注意力图
    attention_plot = sns.heatmap(
        attention_scores, 
        cmap="viridis", 
        annot=False, 
        fmt=".2f", 
        ax=axes[0, 1:].ravel()[0],  # 跨两列显示
        square=True
    )
    attention_plot.set_title("Attention Map")
    axes[0, 2].remove()  # 移除多余的轴

    # 第二行和第三行：6张Grad-CAM热力图
    for idx, (img, feat, grad) in enumerate(zip(obs_imgs, obs_features, obs_features_grad)):
        row = 1 + idx // 3  # 第二行开始
        col = idx % 3       # 从左到右排列
        
        # 计算当前图像的Grad-CAM
        cam = compute_gradcam_heatmap(feat, grad)
        
        # Convert PIL Image to numpy array
        obs_img_np = np.array(img)
        
        # Get image dimensions
        h, w = obs_img_np.shape[:2]
        
        # Resize CAM to match image dimensions
        heatmap = cv2.resize(cam, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加热力图到观测图
        heatmap_img = cv2.addWeighted(obs_img_np, 0.6, heatmap, 0.4, 0)
        
        # 显示叠加后的图像
        axes[row, col].imshow(heatmap_img)
        axes[row, col].set_title(f"Frame {idx+1} with Grad-CAM")
        axes[row, col].axis('off')

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if not display:
        plt.close(fig)


def bc_visualize(
    batch_obs_images: np.ndarray,  # [B, T, 3, H, W]
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    obs_features: np.ndarray,  # [B, T, C, H, W]
    obs_features_grads: np.ndarray,  # [B, T, C, H, W]
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
        obs_imgs = []
        for j in range(batch_obs_images.shape[1]):
            obs_imgs.append(np2img(batch_obs_images[i][j]))  # T * [C, H, W]

        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]
        features = obs_features[i]  # [T, C, H, W]
        features_grad = obs_features_grads[i]  # [T, C, H, W]
        attention_score = attention_scores[i]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        bc_draw(
            obs_imgs,
            pred_waypoints,
            label_waypoints,
            features,
            features_grad,
            attention_score,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{mode}_action_prediction": wandb_list}, commit=False)