"""
Lightweight forward kinematics solver using DH parameters.

This module provides a simple FK solver based on Denavit-Hartenberg parameters
for serial manipulators.

Author: wangchiyu
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class TinyFK:
    """
    Lightweight Forward Kinematics solver using DH parameters.
    
    Computes end effector pose from joint angles using the Denavit-Hartenberg convention.
    """
    
    def __init__(self, dh_params):
        """
        Initialize FK solver with DH parameters.
        
        Args:
            dh_params: List of DH parameters [alpha_{i-1}, a_{i-1}, d_i, theta_i]
                      for each joint, where:
                      - alpha: link twist (rotation about x)
                      - a: link length (translation along x)
                      - d: link offset (translation along z)
                      - theta: joint angle (rotation about z)
        """
        self.dh_params = dh_params

    def fk(self, joint_angles, return_all=False):
        """
        Compute forward kinematics given joint angles.
        
        Args:
            joint_angles: List of joint angle values (radians)
            return_all: If True, return poses for all joints; if False (default), only return end effector
            
        Returns:
            If return_all is False (default):
                list: [x, y, z, qw, qx, qy, qz] representing end effector pose
            If return_all is True:
                list of lists: [[x, y, z, qw, qx, qy, qz], ...] for each joint from base to end
            
        Raises:
            ValueError: If number of joint angles doesn't match DH table size
        """
        if len(self.dh_params) != len(joint_angles):
            raise ValueError("DH table size is not equal to the number of joint angles")

        transform = np.eye(4)
        all_poses = []

        # Apply DH transformation for each joint
        for i in range(len(joint_angles)):
            alpha, a, d, theta = self.dh_params[i]
            theta += joint_angles[i]

            # DH transformation matrix
            transform_i = np.array([
                [np.cos(theta), -np.sin(theta), 0, a],
                [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                [0, 0, 0, 1]
            ])

            transform = np.dot(transform, transform_i)
            
            if return_all:
                # Convert current transform to pose representation
                pose = self._transform_to_pose(transform)
                all_poses.append(pose)

        if return_all:
            return all_poses
        else:
            # Return only the end effector pose (default behavior)
            return self._transform_to_pose(transform)

    def _transform_to_pose(self, transform):
        """
        Convert 4x4 transformation matrix to pose representation.
        
        Args:
            transform: 4x4 transformation matrix
            
        Returns:
            list: [x, y, z, qw, qx, qy, qz] representing pose
        """
        # Extract rotation and translation
        rotation_matrix = transform[:3, :3]
        translation_vector = transform[:3, 3]

        # Convert rotation matrix to quaternion [qx, qy, qz, qw]
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        qw, qx, qy, qz = quaternion[3], quaternion[0], quaternion[1], quaternion[2]

        return [translation_vector[0], translation_vector[1], translation_vector[2], qw, qx, qy, qz]




if __name__ == "__main__":
    print("=" * 70)
    print("Testing TinyFK - Default Mode (End Effector Only)")
    print("=" * 70)
    
    # Example usage - default behavior
    dh_params = [
        [0, 0, 0, 0],
        [np.pi / 2, 0, 0, 0],
        [0, 0.5, 0, 0]
    ]

    fk_solver = TinyFK(dh_params)

    joint_angles = [0.2, 0.3, 0.4]
    
    # Default mode - only return end effector
    result = fk_solver.fk(joint_angles)
    print("\nEnd effector pose [x, y, z, qw, qx, qy, qz]:")
    print(f"  Position: [{result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f}]")
    print(f"  Quaternion: [qw={result[3]:.4f}, qx={result[4]:.4f}, qy={result[5]:.4f}, qz={result[6]:.4f}]")
    
    # New feature - return all joint poses
    print("\n" + "=" * 70)
    print("Testing TinyFK - Extended Mode (All Joints)")
    print("=" * 70)
    
    all_poses = fk_solver.fk(joint_angles, return_all=True)
    print(f"\nNumber of joints: {len(all_poses)}")
    for i, pose in enumerate(all_poses):
        print(f"\nJoint {i+1} pose [x, y, z, qw, qx, qy, qz]:")
        print(f"  Position: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]")
        print(f"  Quaternion: [qw={pose[3]:.4f}, qx={pose[4]:.4f}, qy={pose[5]:.4f}, qz={pose[6]:.4f}]")
