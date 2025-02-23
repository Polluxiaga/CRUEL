import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 # type: ignore
from typing import Optional
import wandb
import yaml

# load data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../my_data/my_data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

VIZ_IMAGE_SIZE = (640, 480)
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


def visualize_traj_pred_BC(
    batch_obs_images: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
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
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        normalized (bool): whether the waypoints are normalized
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
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)


    dataset_name = "scand"
    batch_size = batch_obs_images.shape[0]
    wandb_list = []

    for i in range(min(batch_size, num_images_preds)):
        obs_img = np2img(batch_obs_images[i])
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]

        if normalized:
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        compare_waypoints_pred_to_label_BC(
            obs_img,
            dataset_name,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)


def visualize_traj_pred_GOAL(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
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
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        normalized (bool): whether the waypoints are normalized
        save_folder (str): folder to save the images. If None, will not save the images
        epoch (int): current epoch number
        num_images_preds (int): number of images to visualize
        use_wandb (bool): whether to use wandb to log the images
        display (bool): whether to display the images
    """

    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
    )

    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)


    dataset_name = "scand"
    batch_size = batch_obs_images.shape[0]
    wandb_list = []

    for i in range(min(batch_size, num_images_preds)):
        obs_img = np2img(batch_obs_images[i])
        goal_img = np2img(batch_goal_images[i])
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]

        if normalized:
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        compare_waypoints_pred_to_label_GOAL(
            obs_img,
            goal_img,
            dataset_name,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)


def compare_waypoints_pred_to_label_BC(
    obs_img,
    dataset_name: str,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """

    fig, ax = plt.subplots(1, 2)
    trajs = [pred_waypoints, label_waypoints]
    plot_trajs_and_points(
        ax[0],
        trajs,
        traj_colors=[CYAN, MAGENTA],
    )
    """
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        traj_colors=[CYAN, MAGENTA],
    )
    """
    ax[1].imshow(obs_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    """ax[1].set_title(f"Observation")"""
    ax[1].set_title(f"Current")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


def compare_waypoints_pred_to_label_GOAL(
    obs_img,
    goal_img,
    dataset_name: str,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    Compare predicted path with the gt path of waypoints using egocentric visualization.

    Args:
        obs_img: image of the observation
        goal_img: image of the goal
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        pred_waypoints: predicted waypoints in the image
        label_waypoints: label waypoints in the image
        save_path: path to save the figure
        display: whether to display the figure
    """

    fig, ax = plt.subplots(1, 2)
    trajs = [pred_waypoints, label_waypoints]
    plot_trajs_and_points(
        ax[0],
        trajs,
        traj_colors=[CYAN, MAGENTA],
    )
    """
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        traj_colors=[CYAN, MAGENTA],
    )
    """
    ax[1].imshow(goal_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    """ax[1].set_title(f"Observation")"""
    ax[1].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


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


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,
    traj_colors: list = [CYAN, MAGENTA],
):
    """
    Plot trajectories and points on an image. If there is no configuration for the camera interinstics of the dataset, the image will be plotted as is.
    Args:
        ax: matplotlib axis
        img: image to plot
        dataset_name: name of the dataset found in data_config.yaml (e.g. "recon")
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2)
        traj_colors: list of colors for trajectories
    """
    assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"

    ax.imshow(img)
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        for i, traj in enumerate(list_trajs):
            traj_pixels = get_pos_pixels(
                traj, camera_height, camera_x_offset, camera_matrix, dist_coeffs
            )
            ax.plot(
                traj_pixels[:250, 0],
                traj_pixels[:250, 1],
                color=traj_colors[i],
                lw=2.5,
                )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


def get_pos_pixels(
    points: np.ndarray,  # traj (waypoints in local coordinates)
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        points: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients

    Returns:
        pixels: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    print(pixels)
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    print(pixels)
    pixels = np.array(
        [
            p
            for p in pixels
            if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
        ]
    )
    return pixels


def project_points(
    xy: np.ndarray,  # traj (waypoints in local coordinates)
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.

    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
        camera_height: height of the camera above the ground (in meters)
        camera_x_offset: offset of the camera from the center of the car (in meters)
        camera_matrix: 3x3 matrix representing the camera's intrinsic parameters
        dist_coeffs: vector of distortion coefficients


    Returns:
        uv: array of shape (batch_size, horizon, 2) representing (u, v) coordinates on the 2D image plane
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv