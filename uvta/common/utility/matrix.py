import os
import subprocess

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.spatial.transform import Rotation as R


def relative_transformation(T0, Tt):
    """Compute the relative transformation given the initial transformation T0 and current transformation Tt."""
    T_relative = (
        invert_transformation(T0) @ Tt
    )  # Relative transformation is T0^{-1} * Tt
    return T_relative


# Function to invert a homogeneous transformation matrix
def invert_transformation(T):
    R = T[:3, :3]  # Extract the rotation matrix
    t = T[:3, 3]  # Extract the translation vector

    # Inverse of the rotation matrix is its transpose (since it's orthogonal)
    R_inv = R.T

    # Inverse of the translation is -R_inv * t
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def convert_homogeneous_matrix(R, p):
    """Create a homogeneous transformation matrix from rotation matrix R and translation vector p."""
    T = np.eye(4)  # Initialize a 4x4 identity matrix
    T[:3, :3] = R  # Set the rotation part
    T[:3, 3] = p  # Set the translation part
    return T


def homogeneous_matrix_to_6dof(homogeneous_matrix):
    """Convert a 4x4 homogeneous transformation matrix to a 6-DOF vector.

    The 6-DOF vector consists of translation (x, y, z) and rotation vector representation (rx, ry, rz).

    Args:
        homogeneous_matrix (numpy.ndarray): A 4x4 homogeneous transformation matrix.
            The upper-left 3x3 submatrix represents rotation, and the upper-right 3x1 vector represents translation.

    Returns:
        numpy.ndarray: A 6-element vector containing:
            - First 3 elements: translation components [x, y, z]
            - Last 3 elements: rotation vector components [rx, ry, rz] (axis-angle representation)
    """
    translation = homogeneous_matrix[:3, 3]
    rotation = homogeneous_matrix[:3, :3]
    rotvec = R.from_matrix(rotation).as_rotvec()
    return np.concatenate((translation, rotvec))


def homogeneous_matrix_to_xyzwxzy(homogeneous_matrix):
    """Convert a 4x4 homogeneous transformation matrix to translation and quaternion (w,x,y,z).

    Args:
        homogeneous_matrix (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: A 7-element vector containing:
            - First 3 elements: translation components [x, y, z]
            - Last 4 elements: quaternion components [w, x, y, z]
    """
    translation = homogeneous_matrix[:3, 3]
    rotation = homogeneous_matrix[:3, :3]
    quat = R.from_matrix(rotation).as_quat()  # Returns x,y,z,w
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Reorder to w,x,y,z
    return np.concatenate((translation, quat))


def vec6dof_to_homogeneous_matrix(translation, rotation_vector):
    """
    Construct a 4x4 homogeneous transformation matrix from translation and rotation vector.

    Args:
        translation (numpy.ndarray): A 3-element array representing the translation vector [x, y, z].
        rotation_vector (numpy.ndarray): A 3-element array representing the rotation vector [rx, ry, rz] in axis-angle representation.

    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = R.from_rotvec(rotation_vector).as_matrix()
    return T


def construct_coordinate(origin, x_p, y_p):
    """
    Construct a homogeneous transformation matrix given the origin and points on the x and y axes.

    Parameters:
    origin (np.array): The origin point.
    x_p (np.array): A point on the x-axis.
    y_p (np.array): A point on the y-axis.

    Returns:
    np.array: The homogeneous transformation matrix.
    """
    # x -axis
    x_axis = x_p - origin
    y_axis = y_p - origin

    z_axis = np.cross(x_axis, y_axis)
    # z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)  # xyzxyz

    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # Create homogeneous transformation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))
    T = convert_homogeneous_matrix(R, origin)
    return T


def construct_coordinate_general(origin, point1, point2, axis1="x", axis2="y"):
    """
    Construct a homogeneous transformation matrix given the origin and two points.
    Cross product order follows right-hand rule: xy->z, yz->x, zx->y

    Parameters:
    origin (np.array): The origin point.
    point1 (np.array): A point defining the first axis.
    point2 (np.array): A point defining the second axis.
    axis1 (str): First axis ('x', 'y', 'z' or '-x', '-y', '-z').
    axis2 (str): Second axis ('x', 'y', 'z' or '-x', '-y', '-z').
    """
    # Validate axes
    valid_axes = ["x", "y", "z", "-x", "-y", "-z"]
    if axis1 not in valid_axes or axis2 not in valid_axes:
        raise ValueError("Invalid axes")
    if axis1.replace("-", "") == axis2.replace("-", ""):
        raise ValueError("Axes must be different")

    # Get base axes and signs
    sign1 = -1 if axis1.startswith("-") else 1
    sign2 = -1 if axis2.startswith("-") else 1
    base_axis1 = axis1.replace("-", "")
    base_axis2 = axis2.replace("-", "")

    # Get axes indices (0=x, 1=y, 2=z)
    axes_map = {"x": 0, "y": 1, "z": 2}
    i1 = axes_map[base_axis1]
    i2 = axes_map[base_axis2]
    i3 = 3 - i1 - i2  # remaining axis index

    # Calculate and normalize vectors
    v1 = (point1 - origin) / np.linalg.norm(point1 - origin) * sign1
    v2 = (point2 - origin) / np.linalg.norm(point2 - origin) * sign2

    # Make v2 perpendicular to v1
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    # Right-hand rule: if going backwards in cyclic order (yx,xz,zy), negate cross product
    cyclic_pairs = [(0, 1), (1, 2), (2, 0)]  # xy, yz, zx
    v3 = np.cross(v1, v2)
    if (i1, i2) not in cyclic_pairs:
        v3 = -v3

    # Create rotation matrix
    R = np.zeros((3, 3))
    R[:, i1] = v1
    R[:, i2] = v2
    R[:, i3] = v3

    return convert_homogeneous_matrix(R, origin)


def visualize_frames_and_points(frames, points_dict=None):
    """Visualize coordinate frames and point trajectories in 3D space.

    Args:
        frames (list): List of 4x4 homogeneous transformation matrices representing coordinate frames
        points_dict (dict, optional): Dictionary of point trajectories where each value has shape (T,N,3)
                                    T: number of timesteps, N: number of points, 3: xyz coordinates

    Returns:
        None: Displays the interactive 3D plot
    """
    # Create figure
    fig = go.Figure()

    # Length of coordinate axes
    axis_length = 0.05

    # Colors for x,y,z axes
    colors = {"x": "red", "y": "green", "z": "blue"}

    # Plot coordinate frames for each matrix
    for matrix in frames:
        origin = matrix[:3, 3]

        # Plot axes
        for axis, color in enumerate(colors.values()):
            direction = matrix[:3, axis]
            end_point = origin + direction * axis_length

            # Draw axis line
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], end_point[0]],
                    y=[origin[1], end_point[1]],
                    z=[origin[2], end_point[2]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                )
            )

    # Plot points if provided
    if points_dict is not None:
        # Generate a color palette for different groups
        colors = ["purple", "orange", "cyan", "magenta", "yellow", "brown"]

        for i, (name, points) in enumerate(points_dict.items()):
            color = colors[
                i % len(colors)
            ]  # Cycle through colors if more groups than colors
            # For each point, plot its trajectory over time
            for n in range(points.shape[1]):
                fig.add_trace(
                    go.Scatter3d(
                        x=points[:, n, 0],
                        y=points[:, n, 1],
                        z=points[:, n, 2],
                        mode="lines+markers",
                        marker=dict(size=3, color=color),
                        line=dict(color=color, width=2),
                        name=f"{name}_point{n}",
                    )
                )

    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        width=800,
        height=800,
        showlegend=True,
    )

    fig.show()


def visualize_point_groups(points_dict, show_axes=False, axis_length=0.01):
    """Visualize multiple groups of 3D point trajectories.

    Args:
        points_dict (dict): Dictionary of point trajectories where each value has shape (T,N,3)
                          T: number of timesteps, N: number of points, 3: xyz coordinates
        show_axes (bool): Whether to show coordinate axes at origin
        axis_length (float): Length of coordinate axes if shown

    Returns:
        None: Displays the interactive 3D plot
    """
    # Create figure
    fig = go.Figure()

    # Plot coordinate axes if requested
    if show_axes:
        # Colors for x,y,z axes
        axis_colors = {"x": "red", "y": "green", "z": "blue"}
        origin = np.zeros(3)

        # Plot each axis
        for axis_name, color in axis_colors.items():
            direction = np.zeros(3)
            if axis_name == "x":
                direction[0] = 1
            elif axis_name == "y":
                direction[1] = 1
            else:
                direction[2] = 1

            end_point = origin + direction * axis_length

            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], end_point[0]],
                    y=[origin[1], end_point[1]],
                    z=[origin[2], end_point[2]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"{axis_name}-axis",
                )
            )

    # Generate a color palette for different groups
    colors = ["purple", "orange", "cyan", "magenta", "yellow", "brown"]

    # Plot each group of points
    for i, (name, points) in enumerate(points_dict.items()):
        color = colors[i % len(colors)]
        for n in range(points.shape[1]):
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, n, 0],
                    y=points[:, n, 1],
                    z=points[:, n, 2],
                    mode="lines+markers",
                    marker=dict(size=3, color=color),
                    line=dict(color=color, width=2),
                    name=f"{name}_point{n}",
                )
            )

    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        width=800,
        height=800,
        showlegend=True,
    )

    fig.show()


def get_frame_color(frame_name):
    """Generate a consistent color based on frame name hash"""
    import hashlib

    # Get hash of frame name
    hash_value = int(hashlib.md5(frame_name.encode()).hexdigest(), 16)
    # Generate RGB values between 0.2 and 0.8 to avoid too light/dark colors
    r = 0.2 + (hash_value & 255) / 255 * 0.6
    g = 0.2 + ((hash_value >> 8) & 255) / 255 * 0.6
    b = 0.2 + ((hash_value >> 16) & 255) / 255 * 0.6
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def visualize_multiple_frames_and_points(
    frames_dict,
    points_dict=None,
    show_axes=False,
    show_frames=True,
    axis_length=0.01,
    save_path=None,
):
    """
    Visualize time-varying coordinate frames and point trajectories in 3D space.
    All timesteps of the same frame axis share the same legend entry.

    Args:
        frames_dict: Dictionary of frame matrices over time
        points_dict: Optional dictionary of point trajectories
        show_axes: Whether to show global XYZ axes (default: False)
        show_frames: Whether to show frame axes (default: True)
        axis_length: Length of coordinate axes (default: 0.05)
        save_path: Path to save the image (default: None)
    """
    fig = go.Figure()
    global_axis_length = axis_length * 10  # Make global axes 4x longer for visibility
    base_colors = {"x": "red", "y": "green", "z": "blue"}

    # Add global XYZ axes if enabled
    if show_axes:
        origin = np.array([0, 0, 0])
        for axis, (axis_name, base_color) in enumerate(base_colors.items()):
            direction = np.zeros(3)
            direction[axis] = 1
            end_point = origin + direction * global_axis_length

            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], end_point[0]],
                    y=[origin[1], end_point[1]],
                    z=[origin[2], end_point[2]],
                    mode="lines",
                    line=dict(color=base_color, width=4),
                    opacity=1.0,
                    name=f"global_{axis_name}",
                    showlegend=True,
                    legendgroup=f"global_{axis_name}",
                )
            )

    # Plot coordinate frames for each named matrix sequence
    for frame_name, matrices in frames_dict.items():
        num_timesteps = matrices.shape[0]
        frame_color = get_frame_color(frame_name)

        if show_frames:
            # Create the legend entries first with full opacity
            for axis_name, base_color in base_colors.items():
                # Add a single trace for legend purposes
                fig.add_trace(
                    go.Scatter3d(
                        x=[0],
                        y=[0],
                        z=[0],
                        mode="lines",
                        line=dict(color=base_color, width=3),
                        opacity=1.0,
                        name=f"{frame_name}_{axis_name}",
                        legendgroup=f"{frame_name}_{axis_name}",
                        showlegend=True,
                    )
                )

        # Add start point indicator
        start_point = matrices[0, :3, 3]
        fig.add_trace(
            go.Scatter3d(
                x=[start_point[0]],
                y=[start_point[1]],
                z=[start_point[2]],
                mode="markers",
                marker=dict(color="black", size=8, symbol="circle"),
                name=f"{frame_name}_start",
                showlegend=True,
            )
        )

        # Add trajectory
        fig.add_trace(
            go.Scatter3d(
                x=matrices[:, 0, 3],
                y=matrices[:, 1, 3],
                z=matrices[:, 2, 3],
                mode="lines",
                line=dict(color=frame_color, width=10, dash="dot"),
                opacity=0.5,
                name=f"{frame_name}_trajectory",
                showlegend=True,
                legendgroup=f"{frame_name}_trajectory",
            )
        )

        # Plot the actual frame axes if enabled
        if show_frames:
            for t in range(num_timesteps):
                matrix = matrices[t]
                origin = matrix[:3, 3]
                alpha = 0.01 + 0.99 * (t / max(1, num_timesteps - 1))

                # Plot axes
                for axis, (axis_name, base_color) in enumerate(base_colors.items()):
                    direction = matrix[:3, axis]
                    end_point = origin + direction * axis_length

                    fig.add_trace(
                        go.Scatter3d(
                            x=[origin[0], end_point[0]],
                            y=[origin[1], end_point[1]],
                            z=[origin[2], end_point[2]],
                            mode="lines",
                            line=dict(color=base_color, width=3),
                            opacity=alpha,
                            name=f"{frame_name}_{axis_name}",
                            legendgroup=f"{frame_name}_{axis_name}",
                            showlegend=False,  # Hide from legend
                        )
                    )

    # Plot points if provided
    if points_dict is not None:
        colors = ["purple", "orange", "cyan", "magenta", "yellow", "brown"]

        for i, (name, points) in enumerate(points_dict.items()):
            color = colors[i % len(colors)]
            # Add start point indicator for each point trajectory
            fig.add_trace(
                go.Scatter3d(
                    x=points[0, :, 0],
                    y=points[0, :, 1],
                    z=points[0, :, 2],
                    mode="markers",
                    marker=dict(color="black", size=8, symbol="circle"),
                    name=f"{name}_start",
                    showlegend=True,
                )
            )
            for n in range(points.shape[1]):
                fig.add_trace(
                    go.Scatter3d(
                        x=points[:, n, 0],
                        y=points[:, n, 1],
                        z=points[:, n, 2],
                        mode="lines+markers",
                        marker=dict(size=3, color=color),
                        line=dict(color=color, width=2),
                        name=f"{name}_point{n}",
                        legendgroup=f"{name}_point{n}",
                        showlegend=True,
                    )
                )

    fig.update_layout(
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        width=800,
        height=800,
        showlegend=True,
    )

    if save_path:
        # Save as HTML for interactive viewing
        html_path = save_path.rsplit(".", 1)[0] + ".html"
        print(f"Saving interactive HTML to {html_path}")
        fig.write_html(html_path, auto_open=False)

        # Save static image if requested
        if save_path.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
            print(f"Saving static image to {save_path}")
            fig.write_image(save_path)

    fig.show()
