"""
Forward Kinematics Solver for hand glove (all five fingers).

This module provides forward kinematics computation for all finger motions
using DH parameters, supporting both left and right hands.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from tiny_fk import TinyFK
from utils import deg2rad, mat2quat, quat2mat

HAND_TYPE = 'right'        # 'right' or 'left'

class GloveFKSolver:
    """
    Forward Kinematics Solver for hand glove (all fingers).
    
    Computes finger joint poses using DH parameters for all finger joints.
    Supports both left and right hand configurations.
    """
    
    FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def __init__(self, hand_type='right'):
        """
        Initialize FK solver with DH parameters and fixed transformations.
        
        Args:
            hand_type: 'right' or 'left' to specify hand type
            
        Raises:
            ValueError: If hand_type is invalid
        """
        if hand_type not in ['right', 'left']:
            raise ValueError("hand_type must be 'right' or 'left'")
        
        self.hand_type = hand_type
        
        # Initialize DH parameters and transformations for all fingers
        self._init_finger_configs()
        
        # Create FK solvers for each finger
        self.finger_fk_solvers = {}
        for finger_name in self.FINGERS:
            dh_params = self.finger_configs[finger_name]['dh_params']
            self.finger_fk_solvers[finger_name] = TinyFK(dh_params)
    
    def _init_finger_configs(self):
        """Initialize DH parameters, transformation matrices, and landmark indices."""
        self.finger_configs = self._get_finger_configs(self.hand_type)
        # a₁≠0 使 FK[1](MCP) 随横摆角变化，不适合做稳定锚点。
        # 仍用 FK[0] 作根部（位置固定），跳过 FK[1]；在 generate_landmarks
        # 中对 FK[0] 沿世界 z 偏移 a₁，得到固定的 MCP landmark 位置。
        self.landmark_indices = {
            'thumb':  [0, 2, 4, 5],
            'index':  [0, 2, 3, 4],
            'middle': [0, 2, 3, 4],
            'ring':   [0, 2, 3, 4],
            'pinky':  [0, 2, 3, 4],
        }

    def _get_finger_configs(self, hand_type):
        """Get DH parameters and transformation matrices for all fingers."""
        hand_sign = -1 if hand_type == 'right' else 1
        configs = {}

        # ── Thumb (5 DOF) ──
        configs['thumb'] = {
            'dh_params': [
                [0,                  0,          0,                0],
                [hand_sign * deg2rad(90),    0,          hand_sign * 4.25 / 1000,  deg2rad(45)],
                [-hand_sign * deg2rad(30),   55.25/1000, -hand_sign * 10.2 / 1000, 0],
                [hand_sign * deg2rad(90),    0,          0,                0],
                [-hand_sign * deg2rad(90),   33.15/1000, 0,                0],
            ],
            'T_HandBase_FingerBase': np.array([
                [ 0, 1, 0, 8.5 / 1000],
                [-hand_sign, 0, 0, -hand_sign * 22.1 / 1000],
                [ 0, 0, hand_sign, 18 / 1000],
                [ 0, 0, 0, 1],
            ]),
            'T_FingerEnd_Tip': np.array([
                [0, 0, 1, 21 / 1000],
                [1, 0, 0,  8 / 1000],
                [0, 1, 0,  0],
                [0, 0, 0,  1],
            ]),
        }

        # ── 四指 (4 DOF) ──
        # 基座 (x, y, z) mm; 连杆 (a2, a3) mm; θ偏移 (θ2_off, θ3_off) °; 指尖 (tip_z) mm
        # y 通过 hand_sign 自动镜像适配左右手
        specs = {
            'index':  ((-0.14,   25.86, 48.12), 38.25, 25.5,    0,      0,   24),
            'middle': ((-1.14,    5.56, 51.12), 38.25, 25.5,    0,      0,   24),
            'ring':   (( 0.36,  -14.78, 46.12), 38.25, 25.5,    0,      0,   24),
            'pinky':  (( 3.17,  -32.09, 45.77), 36.38, 15.0, 15.3, -15.3,   15),
        }
        for fn, ((x, y, z), a2, a3, th2, th3, tz) in specs.items():
            configs[fn] = {
                'dh_params': [
                    [0,                 0,        0, 0],
                    [-hand_sign * deg2rad(90),  39.31/1000, 0, deg2rad(th2)],
                    [0,                 a2/1000,  0, deg2rad(th3)],
                    [0,                 a3/1000,  0, 0],
                ],
                'T_HandBase_FingerBase': np.array([
                    [0, 0, -hand_sign, x / 1000],
                    [0, hand_sign,  0, -hand_sign * y / 1000],
                    [1, 0,  0, z / 1000],
                    [0, 0,  0, 1],
                ]),
                'T_FingerEnd_Tip': np.array([
                    [0, 0, 1, tz / 1000],
                    [1, 0, 0, 9 / 1000],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]),
            }

        return configs

    def finger_fk(self, finger_name, joint_angles, return_all_joints=True):
        """
        Compute forward kinematics for a specific finger.
        
        Args:
            finger_name: Name of the finger ('thumb', 'index', 'middle', 'ring', 'pinky')
            joint_angles: List of joint angles (in radians) for the finger
            return_all_joints: If True, return all joint poses; if False, only return tip pose
            
        Returns:
            If return_all_joints is True:
                list of lists: [[x, y, z, qw, qx, qy, qz], ...] for each joint in HandBase frame
            If return_all_joints is False:
                list: [x, y, z, qw, qx, qy, qz] for finger tip in HandBase frame
            
        Raises:
            ValueError: If finger_name is invalid or joint_angles size mismatch
        """
        if finger_name not in self.FINGERS:
            raise ValueError(f"Invalid finger name: {finger_name}. Must be one of {self.FINGERS}")
        
        finger_base_poses = self.finger_fk_solvers[finger_name].fk(joint_angles, return_all=return_all_joints)
        config = self.finger_configs[finger_name]
        
        if return_all_joints:
            hand_base_poses = []
            for pose in finger_base_poses:
                T_FingerBase_Joint = quat2mat(pose)
                T_HandBase_Joint = config['T_HandBase_FingerBase'] @ T_FingerBase_Joint
                hand_base_poses.append(mat2quat(T_HandBase_Joint))
            
            T_FingerBase_End = quat2mat(finger_base_poses[-1])
            T_HandBase_Tip = config['T_HandBase_FingerBase'] @ T_FingerBase_End @ config['T_FingerEnd_Tip']
            hand_base_poses.append(mat2quat(T_HandBase_Tip))
            
            return hand_base_poses
        else:
            T_FingerBase_End = quat2mat(finger_base_poses)
            T_HandBase_Tip = config['T_HandBase_FingerBase'] @ T_FingerBase_End @ config['T_FingerEnd_Tip']
            return mat2quat(T_HandBase_Tip)
    
    def hand_fk(self, joint_angles_dict, return_all_joints=True):
        """
        Compute forward kinematics for all fingers (whole hand).
        
        Args:
            joint_angles_dict: Dictionary mapping finger names to joint angles
                              e.g., {'thumb': [0,0,0,0,0], 'index': [0,0,0,0], ...}
            return_all_joints: If True, return all joint poses; if False, only return tip poses
            
        Returns:
            dict: Dictionary mapping finger names to pose lists
                  If return_all_joints is True: each value is list of poses for all joints + tip
                  If return_all_joints is False: each value is single tip pose
        """
        results = {}
        for finger_name in self.FINGERS:
            if finger_name in joint_angles_dict:
                results[finger_name] = self.finger_fk(
                    finger_name, 
                    joint_angles_dict[finger_name],
                    return_all_joints=return_all_joints
                )
        return results
    
    def generate_landmarks(self, joint_angles_dict):
        """
        Generate 25x7 landmark array for the hand.
        
        Layout (FK[0]+a1*z 偏移得固定 MCP，avg midpoint + FK[0,2,3,4]):
            [0]     : wrist
            [1-4]   : thumb  FK[0,2,4,5]
            [5-9]   : index  avg(wrist,FK[0]) + (FK[0]+a1*z), FK[2], FK[3], FK[4]
            [10-14] : middle ...
            [15-19] : ring   ...
            [20-24] : pinky  (FK[2..4] 额外补偿 θ2 offset 引起的弯折)
        
        Args:
            joint_angles_dict: Dictionary mapping finger names to joint angles (radians)
        
        Returns:
            np.ndarray: (25, 7) array, each row is [x, y, z, qw, qx, qy, qz]
        """
        landmarks = np.zeros((25, 7))
        r0 = Rotation.from_euler('ZX', [-90, 180], degrees=True)
        q0 = r0.as_quat()  # [qx, qy, qz, qw]
        landmarks[0] = [0, 0, 0, q0[3], q0[0], q0[1], q0[2]]

        all_poses = self.hand_fk(joint_angles_dict, return_all_joints=True)

        offset = 1
        for finger_name in self.FINGERS:
            fk_indices = self.landmark_indices[finger_name]
            has_avg = (finger_name != 'thumb')

            if finger_name not in all_poses:
                offset += len(fk_indices) + (1 if has_avg else 0)
                continue

            fk_poses = all_poses[finger_name]
            chain_start = offset

            if has_avg:
                fk0 = np.array(fk_poses[0])
                pos = fk0[:3] / 2.0
                quat = landmarks[0, 3:] + fk0[3:]
                landmarks[offset] = np.concatenate([pos, quat / np.linalg.norm(quat)])
                offset += 1

            for i, idx in enumerate(fk_indices):
                landmarks[offset] = fk_poses[idx]
                if has_avg and i == 0:
                    a1 = self.finger_configs[finger_name]['dh_params'][1][1]
                    landmarks[offset, 2] += a1
                elif has_avg and i > 0:
                    th2_off = self.finger_configs[finger_name]['dh_params'][1][3]
                    if abs(th2_off) > 1e-6:
                        a2 = self.finger_configs[finger_name]['dh_params'][2][1]
                        landmarks[offset, 0] -= a2 * np.sin(th2_off)
                        landmarks[offset, 2] += a2 * (1 - np.cos(th2_off))
                offset += 1

            chain = list(range(chain_start, offset))
            for ci, li in enumerate(chain):
                prev_li = 0 if ci == 0 else chain[ci - 1]
                d = landmarks[prev_li, :3] - landmarks[li, :3]
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-10:
                    continue
                z_new = d / d_norm

                qw, qx, qy, qz = landmarks[li, 3:7]
                z_old = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()[:, 2]

                y_new = np.cross(z_new, -z_old)
                y_norm = np.linalg.norm(y_new)
                if y_norm < 1e-10:
                    continue
                y_new /= y_norm
                x_new = np.cross(y_new, z_new)

                q = Rotation.from_matrix(np.column_stack([x_new, y_new, z_new])).as_quat()
                landmarks[li, 3:7] = [q[3], q[0], q[1], q[2]]

            if has_avg:
                mcp_li = chain_start + 1
                qw, qx, qy, qz = landmarks[mcp_li, 3:7]
                r_mcp = Rotation.from_quat([qx, qy, qz, qw])
                r_mcp = r_mcp * Rotation.from_euler('z', -90, degrees=True)
                q = r_mcp.as_quat()
                landmarks[mcp_li, 3:7] = [q[3], q[0], q[1], q[2]]
                landmarks[chain_start, 3:7] = landmarks[mcp_li, 3:7]

            tip_li = chain[-1]
            qw, qx, qy, qz = landmarks[tip_li, 3:7]
            r_tip = Rotation.from_quat([qx, qy, qz, qw])
            r_new = r_tip * Rotation.from_euler('z', 90, degrees=True)
            q = r_new.as_quat()
            landmarks[tip_li, 3:7] = [q[3], q[0], q[1], q[2]]

        return landmarks

    def get_num_joints(self, finger_name):
        """
        Get the number of joints for a specific finger.
        
        Args:
            finger_name: Name of the finger
            
        Returns:
            int: Number of joints for the finger
        """
        if finger_name not in self.FINGERS:
            raise ValueError(f"Invalid finger name: {finger_name}. Must be one of {self.FINGERS}")
        return len(self.finger_configs[finger_name]['dh_params'])

    def parse_joint_angles(self, q_flat):
        """
        将平面关节向量转为每指 DH 输入（与 glove 内 DH 一致，与 HA4 机械手无关）。

        平面布局（弧度）须与 ``config/serial_parser_raw.JOINT_NAMES`` 一致：
        拇指 0–4，四指各 4 维，小指 17–20 为 MCP_Abd、MCP_Flex、PIP、DIP；
        若 22 维则末位为 Pinky_Invalid，去掉末位得到 21 维。

        Args:
            q_flat: 21 或 22 个关节弧度

        Returns:
            各指关节角列表，供 ``finger_fk`` / ``hand_fk`` 使用。
        """
        q = np.asarray(q_flat, dtype=float)
        if len(q) == 22:
            q = q[:-1].copy()
        elif len(q) != 21:
            raise ValueError(f"Expected 21 or 22 joint angles, got {len(q)}")

        n_pinky = self.get_num_joints('pinky')
        pinky_joints = q[17:17 + n_pinky].tolist()

        return {
            'thumb':  q[0:5].tolist(),
            'index':  q[5:9].tolist(),
            'middle': q[9:13].tolist(),
            'ring':   q[13:17].tolist(),
            'pinky':  pinky_joints,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    print("=" * 70)
    print(f"Testing {HAND_TYPE.upper()} Hand FK Solver")
    print("=" * 70)

    solver = GloveFKSolver(hand_type=HAND_TYPE)

    q_flat_21 = [0.0] * 21
    joint_angles = solver.parse_joint_angles(q_flat_21)
    print(f"Parsed 21D → pinky has {len(joint_angles['pinky'])} joints")

    landmarks = solver.generate_landmarks(joint_angles)
    print(f"Landmarks shape: {landmarks.shape}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    finger_colors = {
        'thumb': 'red',
        'index': 'orange',
        'middle': 'green',
        'ring': 'blue',
        'pinky': 'purple',
    }
    finger_slices = {
        'thumb':  (1, 5),
        'index':  (5, 10),
        'middle': (10, 15),
        'ring':   (15, 20),
        'pinky':  (20, 25),
    }

    def quat_to_axes(qw, qx, qy, qz):
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        return rot[:, 0], rot[:, 1], rot[:, 2]

    axis_len = 0.008
    axis_colors = ['red', 'green', 'blue']

    ax.scatter(*landmarks[0, :3], c='black', s=80, marker='o', label='wrist')
    ax.text(landmarks[0, 0], landmarks[0, 1], landmarks[0, 2], ' 0', fontsize=8)

    for i in range(25):
        pos = landmarks[i, :3]
        axes = quat_to_axes(*landmarks[i, 3:7])
        for a, c in zip(axes, axis_colors):
            ax.quiver(pos[0], pos[1], pos[2],
                      a[0], a[1], a[2],
                      length=axis_len, color=c, arrow_length_ratio=0.2, linewidth=1.2)

    for finger, (start, end) in finger_slices.items():
        pts = landmarks[start:end, :3]
        color = finger_colors[finger]
        chain = np.vstack([landmarks[0, :3], pts])
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], '-o', color=color,
                markersize=5, linewidth=2, label=finger)
        for i in range(start, end):
            ax.text(landmarks[i, 0], landmarks[i, 1], landmarks[i, 2],
                    f' {i}', fontsize=7, color=color)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Hand Landmarks ({HAND_TYPE.upper()} Hand)')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
