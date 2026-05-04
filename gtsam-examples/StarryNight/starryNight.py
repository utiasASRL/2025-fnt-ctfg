import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from pathlib import Path


def psitoC(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=float).reshape(3,)
    psinorm = np.linalg.norm(psi)
    if psinorm == 0:
        return np.eye(3)
    u = psi / psinorm
    ux, uy, uz = u
    hat = np.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0],
    ])
    return (
        np.cos(psinorm) * np.eye(3)
        + (1 - np.cos(psinorm)) * np.outer(u, u)
        - np.sin(psinorm) * hat
    )


def load_poses(filepath: str):
    result_poses = pd.read_csv(filepath)
    n = len(result_poses)
    C_array = np.zeros((3, 3, n))
    t_array = np.zeros((3, n))
    for i, row in result_poses.iterrows():
        rot = R.from_euler("ZYX", [row.yaw, row.pitch, row.roll])
        C_array[:, :, i] = rot.as_matrix()
        t_array[:, i] = [row.x, row.y, row.z]
    return C_array, t_array


def load_landmarks(filepath: str):
    result_landmarks = pd.read_csv(filepath)
    n = len(result_landmarks)
    t_array = np.zeros((3, n))
    for i, row in result_landmarks.iterrows():
        t_array[:, i] = [row.x, row.y, row.z]
    return t_array


def load_marginals(filepath: str):
    marg_input = pd.read_csv(filepath, header=None).to_numpy()
    n = marg_input.shape[0]
    sigmas_3 = np.zeros((6, n))
    for i in range(n):
        P = marg_input[i].reshape(6, 6)
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            print("Matrix is not symmetric positive definite")
        sigmas_3[:, i] = 3 * np.sqrt(np.diag(P))
    return sigmas_3


def compute_error_theta(C_est: np.ndarray, C_gt: np.ndarray) -> np.ndarray:
    n = C_gt.shape[2]
    error_theta = np.zeros((3, n))
    for k in range(n):
        dC = np.eye(3) - C_est[:, :, k] @ C_gt[:, :, k].T
        error_theta[:, k] = [-dC[1, 2], dC[0, 2], -dC[0, 1]]
    return error_theta


if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent
    # Get groundtruth from raw data
    data = loadmat(this_dir / "starryNight.mat")
    theta_vk_i = data["theta_vk_i"]
    r_i_vk_i = data["r_i_vk_i"]
    t = data["t"].reshape(-1)

    k1 = 1215
    k2 = 1714
    n = k2 - k1 + 1
    r_gt = r_i_vk_i[:, k1:k2 + 1]
    theta_gt = theta_vk_i[:, k1:k2 + 1]
    C_gt = np.zeros((3, 3, n))
    for i in range(n):
        C_gt[:, :, i] = psitoC(theta_gt[:, i]).T

    # load GTSAM results
    fp = this_dir / "results"
    pose_interval = 5
    C_array_odom, t_array_odom = load_poses(fp / "starryNightOdom_poses.csv")
    C_array_wnoa, t_array_wnoa = load_poses(fp / "starryNightWNOA_poses.csv")
    C_array_interpolated, t_array_interpolated = load_poses(
        fp / "starryNightInterp_poses.csv"
    )
    l_array = load_landmarks(fp / "starryNight_landmarks.csv")

    # get marginals (covariances)
    sigmas_3_all = load_marginals(fp / "starryNightOdom_marginals.csv")
    sigmas_3_wnoa = load_marginals(fp / "starryNightWNOA_marginals.csv")
    sigmas_3_interpolated = load_marginals(fp / "starryNightInterp_marginals.csv")

    # plotting, same format as a3.m
    plots_dir = this_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.0, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        l_array[0, :],
        l_array[1, :],
        l_array[2, :],
        linewidths=1.5,
        edgecolors=(0, 0.3, 0),
        label="Landmarks",
    )
    ax.plot(
        r_gt[0, :],
        r_gt[1, :],
        r_gt[2, :],
        ".-",
        color=(0.2, 0.6, 0.1),
        markersize=10,
        linewidth=0.5,
        label="Ground-Truth Trajectory",
    )
    ax.plot(
        t_array_wnoa[0, :],
        t_array_wnoa[1, :],
        t_array_wnoa[2, :],
        "m.-",
        markersize=5,
        linewidth=0.5,
        label="Optimized Trajectory (WNOA and Measurements, without Interpolation)",
    )
    ax.plot(
        t_array_interpolated[0, :],
        t_array_interpolated[1, :],
        t_array_interpolated[2, :],
        ".-",
        linewidth=0.5,
        markersize=5,
        color="b",
        label="Optimized Trajectory (WNOA and Measurements, with Interpolation)",
    )
    ax.scatter(
        t_array_interpolated[0, ::pose_interval],
        t_array_interpolated[1, ::pose_interval],
        t_array_interpolated[2, ::pose_interval],
        marker="*",
        color='cyan',
        s=35,
        label="Estimated States (in Main Solve)",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=None, azim=None)
    ax.set_proj_type("persp")
    ax.set_xlim3d(auto=True)
    ax.set_ylim3d(auto=True)
    ax.set_zlim3d(auto=True)
    ax.set_position([0.05, 0.05, 0.9, 0.9])
    ax.figure.canvas.draw_idle()
    fig.savefig(plots_dir / "starryNight.png", dpi=300, bbox_inches="tight")

    # errors
    interp_end_idx = 200  # plot just a subsection for clarity in paper

    plot_type_choices = {"wnoa", "interpolated"}
    plot_type = "interpolated"
    assert plot_type in plot_type_choices

    if plot_type == "wnoa":
        error_r = t_array_wnoa[:, :interp_end_idx] - r_gt[:, :interp_end_idx]
        error_theta = compute_error_theta(C_array_wnoa, C_gt[:, :, :interp_end_idx])
        end_idx_diff = t_array_wnoa.shape[1] - interp_end_idx
        errors_to_plot = np.vstack([error_r, error_theta])
        sigmas_to_plot = sigmas_3_wnoa[:, :interp_end_idx]
        marker_color = (0.9, 0.0, 0.9)
        plot_title = "Pose Errors, without Interpolation"
    else:
        error_r = t_array_interpolated[:, :interp_end_idx] - r_gt[:, :interp_end_idx]
        error_theta = compute_error_theta(
            C_array_interpolated, C_gt[:, :, :interp_end_idx]
        )
        end_idx_diff = t_array_interpolated.shape[1] - interp_end_idx
        errors_to_plot = np.vstack([error_r, error_theta])
        sigmas_to_plot = sigmas_3_interpolated[:, :interp_end_idx]
        marker_color = (0.2, 0.7, 0.8)
        plot_title = "Pose Errors, with Interpolation"

    lim_trans = 0.55
    lim_rot = 0.25
    fig = plt.figure(figsize=(7.5, 10.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(6, 1, hspace=0.1)
    y_label_array = [
        "Error in x [m]",
        "Error in y [m]",
        "Error in z [m]",
        "Error in θ_x [rad]",
        "Error in θ_y [rad]",
        "Error in θ_z [rad]",
    ]
    for i in range(6):
        ax = fig.add_subplot(gs[i, 0])
        ax.scatter(
            t[k1 : k2 - end_idx_diff + 1],
            errors_to_plot[i, :],
            s=100,
            marker=".",
            edgecolors=marker_color,
        )
        x2 = np.concatenate([t[k1 : k2 - end_idx_diff + 1], t[k1 : k2 - end_idx_diff + 1][::-1]])
        in_between = np.concatenate([sigmas_to_plot[i, :], -sigmas_to_plot[i, :][::-1]])
        ax.fill(
            x2,
            in_between,
            color="b",
            linestyle="--",
            edgecolor="none",
            alpha=0.2,
        )
        if plot_type == "interpolated":
            ax.scatter(
                t[k1 : k2 - end_idx_diff + 1 : pose_interval],
                errors_to_plot[i, ::pose_interval],
                s=40,
                marker=".",
                edgecolors="blue",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_label_array[i])
        if i < 3:
            ax.set_ylim([-lim_trans, lim_trans])
        else:
            ax.set_ylim([-lim_rot, lim_rot])
        ax.set_xlim([111.5, None])
        ax.grid(True)
    fig.suptitle(plot_title, fontsize=14, fontweight="normal")
    fig.savefig(plots_dir / "starryNightErrPlot.png", dpi=300, bbox_inches="tight")

    # just comparing covariance between interpolation and no interpolation
    interp_end_idx = t_array_interpolated.shape[1]
    end_idx_diff = t_array_interpolated.shape[1] - interp_end_idx
    fig = plt.figure(figsize=(8.0, 10.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(6, 1, hspace=0.1)
    y_label_array = [
        "σ_x",
        "σ_y",
        "σ_z",
        "σ_θ_x",
        "σ_θ_y",
        "σ_θ_z",
    ]
    lim_trans = 3.0
    lim_rot = 1.2
    for i in range(6):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t[k1:k2 + 1], sigmas_3_wnoa[i, :], "-", linewidth=1.5, color="m", label="No Interpolation")
        ax.plot(
            t[k1 : k2 - end_idx_diff + 1],
            sigmas_3_interpolated[i, :],
            ".-",
            linewidth=1.5,
            color="blue",
            label="With Interpolation",
        )
        ax.scatter(
            t[k1:k2 + 1 : pose_interval],
            sigmas_3_interpolated[i, ::pose_interval],
            marker="o",
            linewidths=2,
            edgecolors="blue",
            label="States in main solve",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel(y_label_array[i])
        if i < 3:
            ax.set_ylim([0, lim_trans])
        else:
            ax.set_ylim([0, lim_rot])
    fig.suptitle("Covariance plot for WNOA and measurements, with and without interpolation")
    fig.legend(loc="lower center", ncol=3)
    fig.savefig(plots_dir / "starryNightCovPlot.png", dpi=300, bbox_inches="tight")
