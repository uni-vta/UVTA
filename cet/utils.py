import os
import re
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from pathlib import Path
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from hdt.constants import RETARGETTING_INDICES, ACTION_STATE_VEC_SIZE, OUTPUT_LEFT_EEF,\
    OUTPUT_RIGHT_EEF, OUTPUT_HEAD_EEF, OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_KEYPOINTS,\
        QPOS_INDICES, ACTION_STATE_VEC_SIZE, H1_QPOS_LEFT_HAND_INDICES, H1_QPOS_RIGHT_HAND_INDICES

import torch
def parse_id(base_dir, prefix):
    """
    Searches for the first subdirectory within the specified base directory that starts with a given prefix.

    Args:
        base_dir (str): The path to the base directory in which to search for subdirectories.
        prefix (str): The prefix to match at the beginning of subdirectory names.

    Returns:
        tuple: A tuple containing:
            - str: The full path of the first matching subdirectory as a string.
            - str: The name of the first matching subdirectory.
            If no matching subdirectory is found, returns (None, None).

    Raises:
        ValueError: If the provided base directory does not exist or is not a directory.
    """
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The provided base directory does not exist or is not a directory: \n{base_path}")

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            return str(subfolder), subfolder.name
    
    # If no matching subfolder is found
    return None, None

#! after process
def rename_processed_files(start_num: int, folder_path: str):
    files = sorted([f for f in os.listdir(folder_path) if f.startswith("processed_episode_") and f.endswith(".hdf5")])
    
    temp_files = []

    for i, filename in enumerate(files):
        temp_filename = f"temp_processed_episode_{i}.hdf5"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_files.append(temp_filename)
        print(f"Temporarily renamed '{filename}' to '{temp_filename}'")

    for i, temp_filename in enumerate(temp_files):
        new_filename = f"processed_episode_{start_num + i}.hdf5"
        temp_path = os.path.join(folder_path, temp_filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"Renamed '{temp_filename}' to '{new_filename}'")

def get_sorted_files(folder_path: str, file_extension: str):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    
    return sorted_files

def rename_files(start_num: int, folder_path: str):
    # Step 1: Rename .hdf5 files
    hdf5_files = get_sorted_files(folder_path, ".hdf5")
    temp_hdf5_files = []
    print("Renaming .hdf5 files:")
    
    for i, filename in enumerate(hdf5_files):
        temp_filename = f"temp_episode_{i}.hdf5"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_hdf5_files.append(temp_filename)
        print(f"Temporarily renamed '{filename}' to '{temp_filename}'")
    
    for i, temp_filename in enumerate(temp_hdf5_files):
        new_filename = f"episode_{start_num + i}.hdf5"
        temp_path = os.path.join(folder_path, temp_filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"Renamed '{temp_filename}' to '{new_filename}'")

    # Step 2: Rename left camera video files
    left_camera_files = get_sorted_files(folder_path, "left_camera.mp4")
    temp_left_camera_files = []
    print("\nRenaming left camera files:")
    
    for i, filename in enumerate(left_camera_files):
        temp_filename = f"temp_episode_{i}_left_camera.mp4"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_left_camera_files.append(temp_filename)
        print(f"Temporarily renamed '{filename}' to '{temp_filename}'")
    
    for i, temp_filename in enumerate(temp_left_camera_files):
        new_filename = f"episode_{start_num + i}_left_camera.mp4"
        temp_path = os.path.join(folder_path, temp_filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"Renamed '{temp_filename}' to '{new_filename}'")

    # Step 3: Rename right camera video files
    right_camera_files = get_sorted_files(folder_path, "right_camera.mp4")
    temp_right_camera_files = []
    print("\nRenaming right camera files:")
    
    for i, filename in enumerate(right_camera_files):
        temp_filename = f"temp_episode_{i}_right_camera.mp4"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_right_camera_files.append(temp_filename)
        print(f"Temporarily renamed '{filename}' to '{temp_filename}'")
    
    for i, temp_filename in enumerate(temp_right_camera_files):
        new_filename = f"episode_{start_num + i}_right_camera.mp4"
        temp_path = os.path.join(folder_path, temp_filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"Renamed '{temp_filename}' to '{new_filename}'")
    
    # Step 4: Rename video_timestamp.npz files
    video_timestamp_files = get_sorted_files(folder_path, "video_timestamp.npz")
    temp_video_timestamp_files = []
    print("\nRenaming video timestamp files:")

    for i, filename in enumerate(video_timestamp_files):
        temp_filename = f"temp_episode_{i}_video_timestamp.npz"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_filename)
        
        os.rename(old_path, temp_path)
        temp_video_timestamp_files.append(temp_filename)
        print(f"Temporarily renamed '{filename}' to '{temp_filename}'")
        
    for i, temp_filename in enumerate(temp_video_timestamp_files):
        new_filename = f"episode_{start_num + i}_video_timestamp.npz"
        temp_path = os.path.join(folder_path, temp_filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(temp_path, new_path)
        print(f"Renamed '{temp_filename}' to '{new_filename}'")

def index_dict(cmd_dict, indices):
    return {k: cmd_dict[k][indices] for k in cmd_dict.keys()}

def cmd_dict2policy(cmd_dict, qpos, src='h1_inspire', match_human=True, match_robot=False, fk_dict=None, still_head=False):
    '''
    cmd_dict: dict of cmds
    qpos: np.array, shape (num_timesteps, 26)
    return:
        policy_action: np.array, shape (num_timesteps, 6 + 9 + 9 + 6*3*2) dim-128
        policy_states: np.array, shape (num_timesteps, 26)
    '''
    #! always 128 dims
    num_timesteps = cmd_dict['head_mat'].shape[0]
    print("num_timesteps", num_timesteps)
    
    left_cmds_matrix = cmd_dict['rel_left_wrist_mat'].reshape((-1, 4, 4))  
    right_cmds_matrix = cmd_dict['rel_right_wrist_mat'].reshape((-1, 4, 4))  
    head_cmds_matrix = cmd_dict['head_mat'].reshape((-1, 4, 4))
    if still_head:
        import hdt.constants
        head_cmds_matrix[:, :3, :3] = hdt.constants.STILL_HEAD_MAT

    left_rot_matrix = torch.tensor(left_cmds_matrix[:, :3, :3], dtype=torch.float32)  
    right_rot_matrix = torch.tensor(right_cmds_matrix[:, :3, :3], dtype=torch.float32) 
    head_rot_matrix = torch.tensor(head_cmds_matrix[:, :3, :3], dtype=torch.float32)

    left_rot_6d = matrix_to_rotation_6d(left_rot_matrix).numpy()
    right_rot_6d = matrix_to_rotation_6d(right_rot_matrix).numpy()
    head_rot_6d = matrix_to_rotation_6d(head_rot_matrix).numpy()

    left_wrist_action = np.concatenate([left_cmds_matrix[:, 0:3, 3], left_rot_6d], axis=1)
    right_wrist_action = np.concatenate([right_cmds_matrix[:, 0:3, 3], right_rot_6d], axis=1)

    left_hand_action = cmd_dict['rel_left_hand_keypoints'].reshape((-1, 25, 3))[:, RETARGETTING_INDICES, :]
    right_hand_action = cmd_dict['rel_right_hand_keypoints'].reshape((-1, 25, 3))[:, RETARGETTING_INDICES, :]

    head_action = np.concatenate([0 * head_cmds_matrix[:, 0:3, 3], head_rot_6d], axis=1)  # mask the translation to 0

    # [0,128)
    policy_action = np.zeros((num_timesteps, ACTION_STATE_VEC_SIZE))
    # [80, 89)
    policy_action[:, OUTPUT_LEFT_EEF] = left_wrist_action
    #! [10, 28) right gripper, arm
    policy_action[:, OUTPUT_LEFT_KEYPOINTS] = left_hand_action.reshape(num_timesteps, -1)

    # [30, 39)
    policy_action[:, OUTPUT_RIGHT_EEF] = right_wrist_action
    #! [40, 58) right gripper, arm
    policy_action[:, OUTPUT_RIGHT_KEYPOINTS] = right_hand_action.reshape(num_timesteps, -1)
    #! right arm
    policy_action[:, OUTPUT_HEAD_EEF] = head_action

    policy_states = np.zeros((num_timesteps, ACTION_STATE_VEC_SIZE))

    if src in ["h1_inspire", "h1_inspire_sim", "gr1_inspire_sim", "h1_2_inspire_cmu", "h1_2_inspire_sim"]:
        policy_states[:, QPOS_INDICES] = qpos[:, :]
    elif "human" in src:
        # Shifts actions by one
        policy_states[1:] = policy_action[:-1]
        policy_states[0] = policy_action[0]
        # Mask out head observations to match robot
    else:
        raise ValueError(f"Invalid source: {src}")

    if match_human and fk_dict:
        policy_states = fk_cmd_dict2policy(fk_dict, num_timesteps, src)
    
    if match_robot:
        policy_states[:, QPOS_INDICES] = qpos[:, :]

    return policy_action, policy_states

def fk_cmd_dict2policy(fk_dict, num_timesteps, src='h1_inspire'):
    # #! always 128 dims
    # num_timesteps = fk_dict['head_mat'].shape[0]
    # print("num_timesteps", num_timesteps)
    
    left_cmds_matrix = fk_dict['rel_left_wrist_mat'].reshape((-1, 4, 4))  
    right_cmds_matrix = fk_dict['rel_right_wrist_mat'].reshape((-1, 4, 4))  
    head_cmds_matrix = fk_dict['head_mat'].reshape((-1, 4, 4))

    left_rot_matrix = torch.tensor(left_cmds_matrix[:, :3, :3], dtype=torch.float32)  
    right_rot_matrix = torch.tensor(right_cmds_matrix[:, :3, :3], dtype=torch.float32) 
    head_rot_matrix = torch.tensor(head_cmds_matrix[:, :3, :3], dtype=torch.float32)

    left_rot_6d = matrix_to_rotation_6d(left_rot_matrix).numpy()
    right_rot_6d = matrix_to_rotation_6d(right_rot_matrix).numpy()
    head_rot_6d = matrix_to_rotation_6d(head_rot_matrix).numpy()

    left_wrist_action = np.concatenate([left_cmds_matrix[:, 0:3, 3], left_rot_6d], axis=1)
    right_wrist_action = np.concatenate([right_cmds_matrix[:, 0:3, 3], right_rot_6d], axis=1)

    left_hand_action = fk_dict['rel_left_hand_keypoints']
    right_hand_action = fk_dict['rel_right_hand_keypoints']

    head_action = np.concatenate([0 * head_cmds_matrix[:, 0:3, 3], head_rot_6d], axis=1)  # mask the translation to 0

    # [0,128)
    policy_state = np.zeros((num_timesteps, ACTION_STATE_VEC_SIZE))
    # [80, 89)
    policy_state[:, OUTPUT_LEFT_EEF] = left_wrist_action
    #! [10, 28) right gripper, arm
    policy_state[:, OUTPUT_LEFT_KEYPOINTS] = left_hand_action.reshape(num_timesteps, -1)

    # [30, 39)
    policy_state[:, OUTPUT_RIGHT_EEF] = right_wrist_action
    #! [40, 58) right gripper, arm
    policy_state[:, OUTPUT_RIGHT_KEYPOINTS] = right_hand_action.reshape(num_timesteps, -1)
    #! right arm
    policy_state[:, OUTPUT_HEAD_EEF] = head_action

    return policy_state

def policy2ctrl_cmd(policy_action):
    '''
    policy_action: np.array, shape (num_timesteps, 128)
    return:
        head_mat: np.array, shape (num_timesteps, 4, 4)
        rel_left_wrist_mat: np.array, shape (num_timesteps, 4, 4)
        rel_right_wrist_mat: np.array, shape (num_timesteps, 4, 4)
        rel_left_hand_keypoints: np.array, shape (num_timesteps, 25, 3)
        rel_right_hand_keypoints: np.array, shape (num_timesteps, 25, 3)
    return can be used for robot controller
    '''
    num_timesteps = policy_action.shape[0]
    left_wrist_action = policy_action[:, OUTPUT_LEFT_EEF]
    left_hand_action = policy_action[:, OUTPUT_LEFT_KEYPOINTS].reshape(num_timesteps, -1, 3)
    right_wrist_action = policy_action[:, OUTPUT_RIGHT_EEF]
    right_hand_action = policy_action[:, OUTPUT_RIGHT_KEYPOINTS].reshape(num_timesteps, -1, 3)
    head_action = policy_action[:, OUTPUT_HEAD_EEF]

    head_mat = np.eye(4)
    head_mat[0:3, 3] = head_action[:, 0:3]
    head_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(head_action[:, 3:9])).numpy()

    rel_left_wrist_mat = np.eye(4)
    rel_left_wrist_mat[0:3, 3] = left_wrist_action[:, 0:3]
    rel_left_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(left_wrist_action[:, 3:9])).numpy()

    rel_right_wrist_mat = np.eye(4)
    rel_right_wrist_mat[0:3, 3] = right_wrist_action[:, 0:3]
    rel_right_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(right_wrist_action[:, 3:9])).numpy()

    rel_left_hand_keypoints = np.zeros((num_timesteps, 25, 3))
    rel_right_hand_keypoints = np.zeros((num_timesteps, 25, 3))
    rel_left_hand_keypoints[:, RETARGETTING_INDICES] = left_hand_action
    rel_right_hand_keypoints[:, RETARGETTING_INDICES] = right_hand_action
    
    return head_mat.squeeze(), rel_left_wrist_mat.squeeze(), rel_right_wrist_mat.squeeze(), rel_left_hand_keypoints.squeeze(), rel_right_hand_keypoints.squeeze()


def safe_as_quat(rotation):
    # Check SciPy version
    version = scipy.__version__.split('.')
    major, minor = int(version[0]), int(version[1])
    
    if major > 1 or (major == 1 and minor >= 14):  # scalar_first introduced in 1.14.0
        return rotation.as_quat(scalar_first=True)
    else:
        # update the order here for older versions manually
        q = rotation.as_quat()
        return np.array([q[3], q[0], q[1], q[2]])
    
def safe_from_quat(quat):
    # Check SciPy version
    version = scipy.__version__.split('.')
    major, minor = int(version[0]), int(version[1])
    
    if major > 1 or (major == 1 and minor >= 14):  # scalar_first introduced in 1.14.0
        return Rotation.from_quat(quat, scalar_first=True)
    else:
        # update the order here for older versions manually
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])