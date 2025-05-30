import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import wandb
import seaborn as sns
import matplotlib.gridspec as gridspec

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


def draw(
    obs_imgs: list,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    attention_scores: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Creates a 3x3 visualization layout:
    - Row 1, Column 1: Trajectory prediction
    - Row 1, Column 2: Full attention map
    - Row 1, Column 3: Bar chart of attention received by each token
    - Rows 2 & 3: 6 observation image attention map overlays
    """
    fig = plt.figure(figsize=(24, 24)) # Revert to 3x3 figsize
    gs = gridspec.GridSpec(3, 3, figure=fig) # Revert to 3x3 GridSpec

    # Row 1, Column 1: Trajectory Prediction
    ax_traj_pred = fig.add_subplot(gs[0, 0])
    plot_trajs_and_points(
        ax_traj_pred,
        [pred_waypoints, label_waypoints],
        traj_colors=[CYAN, MAGENTA],
    )
    ax_traj_pred.set_title("Action Prediction", fontsize=18)

    # Row 1, Column 2: Full Attention Map
    ax_full_attn_map = fig.add_subplot(gs[0, 1]) # Now occupies a single column
    sns.heatmap(
        attention_scores,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        ax=ax_full_attn_map,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Attention Score'}
    )
    ax_full_attn_map.set_title("Overall Attention Map (All Tokens)", fontsize=18)
    ax_full_attn_map.set_xlabel("Key Tokens", fontsize=14)
    ax_full_attn_map.set_ylabel("Query Tokens", fontsize=14)
    ax_full_attn_map.tick_params(labelsize=10)

    # Row 1, Column 3: Bar chart of attention received by each token
    ax_received_attn = fig.add_subplot(gs[0, 2])

    # Calculate attention received by each token (sum of each column)
    # The sum along axis=0 gives the total attention received by each key token from all queries.
    attention_received_by_token = attention_scores.sum(axis=0) # Shape: [seq_len]

    # Create the bar plot
    bars = ax_received_attn.bar(
        np.arange(attention_received_by_token.shape[0]),
        attention_received_by_token,
        color='skyblue'
    )
    ax_received_attn.set_title("Attention Received by Each Token", fontsize=18)
    ax_received_attn.set_xlabel("Token Index", fontsize=14)
    ax_received_attn.set_ylabel("Total Attention Received", fontsize=14)
    ax_received_attn.tick_params(labelsize=12)
    ax_received_attn.grid(axis='y', linestyle='--', alpha=0.7)

    # Add numerical labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax_received_attn.annotate(f'{height:.2f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=10)
    
    # Rows 2 & 3: 6 Observation Image Attention Map Overlays
    num_frames = len(obs_imgs)
    for idx, img in enumerate(obs_imgs):
        if idx >= num_frames:
            break
            
        row = 1 + idx // 3 # Now starts from row 2 (index 1)
        col = idx % 3
        
        ax_img_overlay = fig.add_subplot(gs[row, col])
        
        attention_map = compute_attention_map(attention_scores, idx)
        
        obs_img_np = np.array(img)
        h, w = obs_img_np.shape[:2]
        
        heatmap = cv2.resize(attention_map, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap_img = cv2.addWeighted(obs_img_np, 0.6, heatmap, 0.4, 0)
        
        ax_img_overlay.imshow(heatmap_img)
        ax_img_overlay.set_title(f"Frame {idx+1} with Attention Map", fontsize=16)
        ax_img_overlay.axis('off')

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if not display:
        plt.close(fig)


def visualize(
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

        draw(
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