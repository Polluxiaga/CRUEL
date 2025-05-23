import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import wandb
import seaborn as sns

VIZ_IMAGE_SIZE = (500, 400)
FEATURE_SIZE = (4, 5)
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


def compute_attention_map(attention_scores: np.ndarray, time_idx: int) -> np.ndarray:
    """
    根据attention scores计算特定时间步的注意力热力图。
    Args:
        attention_scores: 形状为 [seq_len, seq_len] 的注意力分数
        time_idx: 时间步索引
    Returns:
        热力图，范围 [0, 1]
    """
    # 先在行方向上平均，得到每个token被关注的平均程度 [1, seq_len]
    mean_attention = attention_scores.mean(axis=0)
    
    # 对整个序列的attention scores进行归一化
    mean_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
    
    # 获取对应时间步的attention scores
    start_idx = time_idx * (FEATURE_SIZE[0] * FEATURE_SIZE[1])
    end_idx = (time_idx + 1) * (FEATURE_SIZE[0] * FEATURE_SIZE[1])
    frame_attention = mean_attention[start_idx:end_idx]  # [feature_len]
    
    # 将一维attention weights重塑为二维特征图
    H, W = FEATURE_SIZE
    attention_map = frame_attention.reshape(H, W)
    
    return attention_map     


def bc_draw(
    obs_imgs: list,  # List of 6 PIL Images
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    attention_scores: np.ndarray,  # [seq_len, seq_len]
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    创建3x3的可视化布局:
    - 第一行：轨迹预测(1列) + 注意力图(2-3列)
    - 第二行和第三行：6张观测图的attention map叠加图
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
        ax=axes[0, 1:].ravel()[0],
        square=True
    )
    attention_plot.set_title("Attention Map")
    axes[0, 2].remove()
    
    # 第二行和第三行：6张attention map叠加图
    for idx, img in enumerate(obs_imgs):
        row = 1 + idx // 3
        col = idx % 3
        
        # 计算当前时间步的attention map
        attention_map = compute_attention_map(attention_scores, idx)
        
        # Convert PIL Image to numpy array
        obs_img_np = np.array(img)
        
        # Get image dimensions
        h, w = obs_img_np.shape[:2]
        
        # Resize attention map to match image dimensions
        heatmap = cv2.resize(attention_map, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加热力图到观测图
        heatmap_img = cv2.addWeighted(obs_img_np, 0.6, heatmap, 0.4, 0)
        
        # 显示叠加后的图像
        axes[row, col].imshow(heatmap_img)
        axes[row, col].set_title(f"Frame {idx+1} with Attention Map")
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
        attention_score = attention_scores[i]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        bc_draw(
            obs_imgs,
            pred_waypoints,
            label_waypoints,
            attention_score,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{mode}_action_prediction": wandb_list}, commit=False)