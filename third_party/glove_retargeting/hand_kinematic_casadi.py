#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用纯Python实现完整的URDF加载和正运动学计算
包含完整的运动学链计算，与Pinocchio结果进行详细对比
"""

import logging
import numpy as np
import casadi as ca
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import os
from collections import defaultdict

class HandKinematicCasadi:
    """使用CasADi的手部运动学模型
      fkine是python函数，fkine_symbolic是CasADi符号化函数
    """
    
    def __init__(self, urdf_path, base_link='base_link', keypoint_links=None, keypoint_offsets=None, joint_names=None):
        """
        初始化python手部运动学模型
        
        Args:
            urdf_path: URDF文件路径
            base_link: 基础链接名称
            keypoint_links: 关键点链接名称列表
            keypoint_offsets: 关键点偏移量列表
            joint_names: 关节名称列表
        """
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.keypoint_links = keypoint_links or []
        self.keypoint_offsets = keypoint_offsets or []
        self.joint_names = joint_names or []
        
        # 性能优化：缓存
        self._transform_cache = {}
        self._last_qpos = None
        self._cached_result = None
        
        # 解析URDF文件
        self.parse_urdf(urdf_path)
        
        # 构建运动学树
        self.build_kinematic_tree()
        
        # 初始化关键点
        if keypoint_links:
            self.initialize_keypoints(keypoint_links, keypoint_offsets)

        self.fkine_symbolic_model, self.q_symbolic, self.fkine_symbolic_function = self.get_fkine_symbolic_model()
    
    def parse_urdf(self, urdf_path):
        """解析URDF文件，提取关节和链接信息"""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF文件不存在: {urdf_path}")
        
        logging.info(f"正在解析URDF文件: {urdf_path}")
        
        # 解析XML
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # 提取关节信息
        self.joints = {}
        self.links = {}
        self.joint_limits = {}
        
        # 处理关节
        for joint_elem in root.findall('.//joint'):
            joint_name = joint_elem.get('name')
            joint_type = joint_elem.get('type')
            
            # 处理所有类型的关节，包括fixed
            if joint_type in ['revolute', 'continuous', 'fixed']:
                # 获取关节轴
                axis_elem = joint_elem.find('axis')
                if axis_elem is not None:
                    axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()]
                else:
                    axis = [0, 0, 1]
                
                # 获取关节限制（只有revolute和continuous关节有）
                if joint_type in ['revolute', 'continuous']:
                    limit_elem = joint_elem.find('limit')
                    if limit_elem is not None:
                        lower = float(limit_elem.get('lower', '-3.14'))
                        upper = float(limit_elem.get('upper', '3.14'))
                    else:
                        lower, upper = -3.14, 3.14
                else:
                    # fixed关节没有限制
                    lower, upper = 0.0, 0.0
                
                # 获取父链接和子链接
                parent_link = joint_elem.find('parent').get('link')
                child_link = joint_elem.find('child').get('link')
                
                # 获取关节原点变换
                origin_elem = joint_elem.find('origin')
                if origin_elem is not None:
                    xyz = [float(x) for x in origin_elem.get('xyz', '0 0 0').split()]
                    rpy = [float(x) for x in origin_elem.get('rpy', '0 0 0').split()]
                else:
                    xyz = [0, 0, 0]
                    rpy = [0, 0, 0]
                
                self.joints[joint_name] = {
                    'type': joint_type,
                    'axis': axis,
                    'parent': parent_link,
                    'child': child_link,
                    'origin_xyz': xyz,
                    'origin_rpy': rpy,
                    'lower': lower,
                    'upper': upper
                }
                
                if joint_type in ['revolute', 'continuous']:
                    self.joint_limits[joint_name] = (lower, upper)
        
        # 处理链接
        for link_elem in root.findall('.//link'):
            link_name = link_elem.get('name')
            
            # 获取链接的几何信息（简化处理）
            visual_elem = link_elem.find('.//visual')
            if visual_elem is not None:
                geometry_elem = visual_elem.find('geometry')
                if geometry_elem is not None:
                    # 这里可以提取更多几何信息，但为了简化，我们只关注变换
                    pass
            
            self.links[link_name] = {
                'name': link_name
            }
        
        # print(f"Parsing completed: found {len(self.joints)} joints, {len(self.links)} links")
        # print(f"Joint type statistics: {dict([(joint_info['type'], sum(1 for j in self.joints.values() if j['type'] == joint_info['type'])) for joint_info in self.joints.values()])}")
    
    def build_kinematic_tree(self):
        """构建运动学树结构"""
        # 构建父子关系图
        self.parent_to_child = defaultdict(list)
        self.child_to_parent = {}
        self.joint_to_link = {}
        
        for joint_name, joint_info in self.joints.items():
            parent = joint_info['parent']
            child = joint_info['child']
            
            self.parent_to_child[parent].append((joint_name, child))
            self.child_to_parent[child] = (joint_name, parent)
            self.joint_to_link[joint_name] = child
        
        # 找到根链接（没有父链接的链接）
        all_links = set(self.links.keys())
        child_links = set(self.child_to_parent.keys())
        root_links = all_links - child_links
        
        if not root_links:
            logger = logging.getLogger(__name__)
            logger.warning("无法确定根链接")
            self.root_link = list(all_links)[0] if all_links else 'base_link'
        else:
            # 优先选择指定的base_link作为根链接
            if self.base_link in all_links:
                self.root_link = self.base_link
                # print(f"Using specified base_link as root link: {self.root_link}")
            else:
                self.root_link = list(root_links)[0]
                if len(root_links) > 1:
                    logging.warning(f"警告: 找到多个根链接: {root_links}")
                    logging.warning(f"选择: {self.root_link}")
        
        # print(f"Kinematic tree root link: {self.root_link}")
    
    def find_path_from_root_to_base(self):
        """找到从根链接到base_link的路径"""
        if self.root_link == self.base_link:
            return []
        
        # 从根链接开始，向下查找base_link
        path = []
        visited = set()
        queue = [(self.root_link, [])]
        
        while queue:
            current_link, current_path = queue.pop(0)
            if current_link in visited:
                continue
            
            visited.add(current_link)
            
            # 如果找到了base_link
            if current_link == self.base_link:
                return current_path
            
            # 继续向下查找
            for joint_name, child_link in self.parent_to_child.get(current_link, []):
                if child_link not in visited:
                    new_path = current_path + [(joint_name, child_link)]
                    queue.append((child_link, new_path))
        
        # 如果找不到路径，返回空列表
        return []
    
    def find_path_to_base(self, target_link):
        """简化的路径查找：从目标链接向上查找，直到找到base_link"""
        if target_link == self.base_link:
            return []
        
        path = []
        current_link = target_link
        visited = set()
        
        while current_link in self.child_to_parent and current_link not in visited:
            visited.add(current_link)
            joint_name, parent_link = self.child_to_parent[current_link]
            path.append((joint_name, parent_link))
            current_link = parent_link
            
            if current_link == self.base_link:
                break
        
        # 反转路径，使其从base_link到target_link
        path.reverse()
        return path
    
    def initialize_keypoints(self, keypoint_links, keypoint_offsets):
        """初始化关键点"""
        if len(keypoint_links) != len(keypoint_offsets):
            raise ValueError("关键点链接数量与偏移量数量不匹配")
        
        self.keypoint_links = keypoint_links
        self.keypoint_offsets = np.array(keypoint_offsets)
        
        # 为每个关键点计算到base_link的路径
        self.keypoint_paths = []
        for link_name in keypoint_links:
            if link_name in self.links:
                # 使用简化的路径查找
                path = self.find_path_to_base(link_name)
                self.keypoint_paths.append(path)
                # print(f"Keypoint {link_name} to {self.base_link} path: {[joint for joint, _ in path]}")
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"找不到链接 '{link_name}'")
                self.keypoint_paths.append([])
        
        # print(f"Initialized {len(keypoint_links)} keypoints")
    
    def get_joint_limit(self):
        """获取关节限制"""
        if not self.joint_names:
            return [], []
        
        limits = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_limits:
                limits.append(self.joint_limits[joint_name])
            else:
                limits.append((-3.14, 3.14))  # 默认限制
        
        lower = [limit[0] for limit in limits]
        upper = [limit[1] for limit in limits]
        return np.array(lower), np.array(upper)
    
    def get_n_dof(self):
        """获取自由度数量"""
        return len(self.joint_names)
    
    def get_joint_names(self):
        """获取关节名称列表"""
        return self.joint_names.copy()
    
    def create_transform_matrix(self, xyz, rpy):
        """创建4x4变换矩阵"""
        # 创建旋转矩阵
        rot = R.from_euler('xyz', rpy).as_matrix()
        
        # 创建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = xyz
        
        return transform
    
    def create_joint_transform(self, joint_name, joint_value):
        """创建关节变换矩阵"""
        if joint_name not in self.joints:
            return np.eye(4)
        
        joint = self.joints[joint_name]
        
        # 基础变换（关节原点）
        base_transform = self.create_transform_matrix(
            joint['origin_xyz'], 
            joint['origin_rpy']
        )
        
        # 关节旋转变换
        axis = np.array(joint['axis'])
        angle = joint_value
        
        # 使用Rodrigues公式计算旋转矩阵
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R_joint = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # 创建关节旋转的4x4变换矩阵
        joint_transform = np.eye(4)
        joint_transform[:3, :3] = R_joint
        
        # 组合变换
        return base_transform @ joint_transform
    
    def fkine(self, qpos):
        """优化的前向运动学计算"""
        if not hasattr(self, 'keypoint_paths'):
            raise ValueError("请先调用 initialize_keypoints 初始化关键点")
        
        # 性能优化：如果qpos相同，则从缓存加载结果
        if self._last_qpos is not None and np.allclose(self._last_qpos, qpos, atol=1e-10):
            return self._cached_result
        
        # 创建关节角度字典
        joint_values = {}
        for i, joint_name in enumerate(self.joint_names):
            if i < len(qpos):
                joint_values[joint_name] = qpos[i]
            else:
                joint_values[joint_name] = 0.0
        
        # 预计算所有关节的变换矩阵
        joint_transforms = {}
        for joint_name in joint_values:
            if joint_name in self.joints:
                joint_transforms[joint_name] = self.create_joint_transform(joint_name, joint_values[joint_name])
        
        # 计算每个关键点的位姿
        keypoint_poses = []
        
        for i, path in enumerate(self.keypoint_paths):
            if not path:
                # 无效路径，使用零位姿
                keypoint_poses.append(np.array([0, 0, 0, 1, 0, 0, 0]))
                continue
            
            # 从base_link开始，沿着路径计算变换
            current_transform = np.eye(4)
            
            for joint_name, _ in path:
                if joint_name in joint_transforms:
                    # 使用预计算的关节变换
                    current_transform = current_transform @ joint_transforms[joint_name]
                else:
                    # 关节不在驱动列表中，使用零角度
                    if joint_name not in joint_transforms:
                        joint_transforms[joint_name] = self.create_joint_transform(joint_name, 0.0)
                    current_transform = current_transform @ joint_transforms[joint_name]
            
            # 应用关键点偏移量
            offset = self.keypoint_offsets[i]
            if np.any(offset != 0):
                offset_transform = np.eye(4)
                offset_transform[:3, 3] = offset
                current_transform = current_transform @ offset_transform
            
            # 提取位置和四元数
            position = current_transform[:3, 3]
            rotation_matrix = current_transform[:3, :3]
            
            # 优化的四元数计算
            quaternion = self._matrix_to_quaternion_fast(rotation_matrix)
            
            # 转换为 [x, y, z, qw, qx, qy, qz] 格式
            pose_vector = np.concatenate([position, [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]])
            keypoint_poses.append(pose_vector)
        
        result = np.array(keypoint_poses)
        
        # 缓存结果
        self._last_qpos = qpos.copy()
        self._cached_result = result
        
        return result
    
    def get_fkine_symbolic_model(self, qpos_symbolic=None):
        """
        使用CasADi符号化计算前向运动学
        
        Args:
            qpos_symbolic: CasADi符号变量，如果为None则自动创建
            
        Returns:
            tuple: (符号化结果, 符号变量, 符号化函数)
        """
        if not hasattr(self, 'keypoint_paths'):
            raise ValueError("请先调用 initialize_keypoints 初始化关键点")
        
        # 创建符号变量
        if qpos_symbolic is None:
            n_dof = self.get_n_dof()
            qpos_symbolic = ca.SX.sym('q', n_dof)
        
        # 创建关节角度字典
        joint_values_symbolic = {}
        n_dof = self.get_n_dof()
        
        for i, joint_name in enumerate(self.joint_names):
            if i < n_dof:
                joint_values_symbolic[joint_name] = qpos_symbolic[i]
            else:
                joint_values_symbolic[joint_name] = 0.0
        
        # 预计算所有关节的符号化变换矩阵
        joint_transforms_symbolic = {}
        for joint_name in joint_values_symbolic:
            if joint_name in self.joints:
                joint_transforms_symbolic[joint_name] = self._create_joint_transform_symbolic(
                    joint_name, joint_values_symbolic[joint_name]
                )
        
        # 计算每个关键点的符号化位姿
        keypoint_poses_symbolic = []
        
        for i, path in enumerate(self.keypoint_paths):
            if not path:
                # 无效路径，使用零位姿
                zero_pose = ca.vertcat(0, 0, 0, 1, 0, 0, 0)
                keypoint_poses_symbolic.append(zero_pose)
                continue
            
            # 从base_link开始，沿着路径计算变换
            current_transform = ca.SX.eye(4)
            
            for joint_name, _ in path:
                if joint_name in joint_transforms_symbolic:
                    # 使用预计算的符号化关节变换
                    current_transform = current_transform @ joint_transforms_symbolic[joint_name]
                else:
                    # 关节不在驱动列表中，使用零角度
                    if joint_name not in joint_transforms_symbolic:
                        joint_transforms_symbolic[joint_name] = self._create_joint_transform_symbolic(
                            joint_name, 0.0
                        )
                    current_transform = current_transform @ joint_transforms_symbolic[joint_name]
            
            # 应用关键点偏移量
            offset = self.keypoint_offsets[i]
            if np.any(offset != 0):
                offset_transform = ca.SX.eye(4)
                offset_transform[:3, 3] = offset
                current_transform = current_transform @ offset_transform
            
            # 提取位置和四元数
            position = current_transform[:3, 3]
            rotation_matrix = current_transform[:3, :3]
            
            # 符号化四元数计算
            quaternion = self._matrix_to_quaternion_symbolic(rotation_matrix)
            
            # 转换为 [x, y, z, qw, qx, qy, qz] 格式
            pose_vector = ca.vertcat(
                position, 
                quaternion[3], quaternion[0], quaternion[1], quaternion[2]
            )
            keypoint_poses_symbolic.append(pose_vector)
        
        # 组合所有关键点的符号化结果
        if keypoint_poses_symbolic:
            # 确保每个关键点都是列向量
            result_symbolic = ca.horzcat(*keypoint_poses_symbolic).T
        else:
            # 如果没有关键点，创建空的符号化结果
            result_symbolic = ca.SX.zeros(0, 7)
        
        # print(f"   Debug info: Symbolic result shape = {result_symbolic.shape}")
        # print(f"   Debug info: Number of keypoints = {len(keypoint_poses_symbolic)}")
        
        # 创建符号化函数
        fkine_function = ca.Function('fkine_symbolic', [qpos_symbolic], [result_symbolic])
        
        return result_symbolic, qpos_symbolic, fkine_function
    
    def fkine_symbolic(self, qpos_list):
        return self.fkine_symbolic_function(qpos_list).full()

    def _create_joint_transform_symbolic(self, joint_name, joint_value_symbolic):
        """创建符号化关节变换矩阵"""
        if joint_name not in self.joints:
            return ca.SX.eye(4)
        
        joint = self.joints[joint_name]
        
        # 基础变换（关节原点）
        base_transform = self._create_transform_matrix_symbolic(
            joint['origin_xyz'], 
            joint['origin_rpy']
        )
        
        # 关节旋转变换
        axis = np.array(joint['axis'])
        
        # 使用Rodrigues公式计算旋转矩阵（符号化版本）
        K = ca.SX([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
        
        # 符号化三角函数
        sin_angle = ca.sin(joint_value_symbolic)
        cos_angle = ca.cos(joint_value_symbolic)
        
        R_joint = ca.SX.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
        
        # 创建关节旋转的4x4变换矩阵
        joint_transform = ca.SX.eye(4)
        joint_transform[:3, :3] = R_joint
        
        # 组合变换
        return base_transform @ joint_transform
    
    def _create_transform_matrix_symbolic(self, xyz, rpy):
        """创建符号化4x4变换矩阵"""
        # 创建旋转矩阵（符号化版本）
        rot = self._euler_to_matrix_symbolic(rpy)
        
        # 创建4x4变换矩阵
        transform = ca.SX.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = xyz
        
        return transform
    
    def _euler_to_matrix_symbolic(self, rpy):
        """符号化欧拉角到旋转矩阵转换"""
        roll, pitch, yaw = rpy
        
        # 符号化三角函数
        cr, sr = ca.cos(roll), ca.sin(roll)
        cp, sp = ca.cos(pitch), ca.sin(pitch)
        cy, sy = ca.cos(yaw), ca.sin(yaw)
        
        # 旋转矩阵
        R = ca.SX.zeros(3, 3)
        
        R[0, 0] = cy * cp
        R[0, 1] = cy * sp * sr - sy * cr
        R[0, 2] = cy * sp * cr + sy * sr
        R[1, 0] = sy * cp
        R[1, 1] = sy * sp * sr + cy * cr
        R[1, 2] = sy * sp * cr - cy * sr
        R[2, 0] = -sp
        R[2, 1] = cp * sr
        R[2, 2] = cp * cr
        
        return R
    
    def _matrix_to_quaternion_symbolic(self, R):
        """符号化矩阵到四元数转换"""
        # 使用更稳定的四元数计算方法（符号化版本）
        trace = ca.trace(R)
        
        # 创建符号化四元数变量
        quat = ca.SX.zeros(4)
        
        # 使用更稳定的方法，避免除零问题
        # 计算四元数分量
        quat[0] = (R[2, 1] - R[1, 2]) / 4  # x
        quat[1] = (R[0, 2] - R[2, 0]) / 4  # y
        quat[2] = (R[1, 0] - R[0, 1]) / 4  # z
        quat[3] = (trace + 1) / 4  # w
        
        # 归一化四元数
        norm = ca.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
        
        # 避免除零，使用条件表达式
        quat_normalized = ca.if_else(norm > 1e-10, quat / norm, quat)
        
        return quat_normalized
    
    def _matrix_to_quaternion_fast(self, R):
        """快速矩阵到四元数转换（避免scipy的开销）"""
        # 使用更高效的四元数计算方法
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return np.array([x, y, z, w])

def get_ha4_r_config():
    """获取HA4-R的默认配置"""
    # 获取URDF文件路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    urdf_path = os.path.join(script_dir, "./urdf/right_sharpa_ha4/right_sharpa_ha4.urdf")
    
    # 关节名称（按顺序，根据HA4 URDF中的关节定义）
    joint_names = [
        "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
        "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
        "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
        "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
        "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP"
    ]
    
    # 关键点链接名称（根据HA4 URDF中的链接定义）
    keypoint_links = [
        "right_thumb_CMC_VL", "right_thumb_MCP_VL", "right_thumb_DP", "right_thumb_fingertip",
        "right_index_MCP_VL", "right_index_MP", "right_index_DP", "right_index_fingertip",
        "right_middle_MCP_VL", "right_middle_MP", "right_middle_DP", "right_middle_fingertip",
        "right_ring_MCP_VL", "right_ring_MP", "right_ring_DP", "right_ring_fingertip",
        "right_pinky_MC", "right_pinky_MP", "right_pinky_DP", "right_pinky_fingertip"
    ]
    
    # 关键点偏移量（相对于链接中心的偏移）
    keypoint_offsets = [
        [0.0, 0.0, 0.0],  # right_thumb_CMC_VL
        [0.0, 0.0, 0.0],  # right_thumb_MCP_VL
        [0.0, 0.0, 0.0],  # right_thumb_DP
        [0.0, 0.0, 0.0],  # right_thumb_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # right_index_MCP_VL
        [0.0, 0.0, 0.0],  # right_index_PP
        [0.0, 0.0, 0.0],  # right_index_DP
        [0.0, 0.0, 0.0],  # right_index_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # right_middle_MCP_VL
        [0.0, 0.0, 0.0],  # right_middle_PP
        [0.0, 0.0, 0.0],  # right_middle_DP
        [0.0, 0.0, 0.0],  # right_middle_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # right_ring_MCP_VL
        [0.0, 0.0, 0.0],  # right_ring_PP
        [0.0, 0.0, 0.0],  # right_ring_DP
        [0.0, 0.0, 0.0],  # right_ring_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # right_pinky_MC
        [0.0, 0.0, 0.0],  # right_pinky_PP
        [0.0, 0.0, 0.0],  # right_pinky_DP
        [0.0, 0.0, 0.0],  # right_pinky_fingertip (根据fingertip_joint的origin)
    ]
    
    return {
        'urdf_path': urdf_path,
        'base_link': 'virtual_hand_base',  # ,使用虚拟掌心，原来是 'right_hand_C_MC'
        'joint_names': joint_names,
        'keypoint_links': keypoint_links,
        'keypoint_offsets': keypoint_offsets
    }

def get_ha4_l_config():
    """获取HA4-L的默认配置"""
    # 获取URDF文件路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    urdf_path = os.path.join(script_dir, "./urdf/left_sharpa_ha4/left_sharpa_ha4.urdf")
    
    # 关节名称（按顺序，根据HA4 URDF中的关节定义）
    joint_names = [
        "left_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
        "left_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
        "left_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
        "left_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
        "left_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP"
    ]

    # 关键点链接名称（根据HA4 URDF中的链接定义）
    keypoint_links = [
        "left_thumb_CMC_VL", "left_thumb_MCP_VL", "left_thumb_DP", "left_thumb_fingertip",
        "left_index_MCP_VL", "left_index_MP", "left_index_DP", "left_index_fingertip",
        "left_middle_MCP_VL", "left_middle_MP", "left_middle_DP", "left_middle_fingertip",
        "left_ring_MCP_VL", "left_ring_MP", "left_ring_DP", "left_ring_fingertip",
        "left_pinky_MC", "left_pinky_MP", "left_pinky_DP", "left_pinky_fingertip"
    ]
    
    # 关键点偏移量（相对于链接中心的偏移）
    keypoint_offsets = [
        [0.0, 0.0, 0.0],  # left_thumb_CMC_VL   
        [0.0, 0.0, 0.0],  # left_thumb_MCP_VL
        [0.0, 0.0, 0.0],  # left_thumb_DP
        [0.0, 0.0, 0.0],  # left_thumb_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # left_index_MCP_VL
        [0.0, 0.0, 0.0],  # left_index_MP
        [0.0, 0.0, 0.0],  # left_index_DP
        [0.0, 0.0, 0.0],  # left_index_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # left_middle_MCP_VL
        [0.0, 0.0, 0.0],  # left_middle_MP
        [0.0, 0.0, 0.0],  # left_middle_DP
        [0.0, 0.0, 0.0],  # left_middle_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # left_ring_MCP_VL
        [0.0, 0.0, 0.0],  # left_ring_MP
        [0.0, 0.0, 0.0],  # left_ring_DP
        [0.0, 0.0, 0.0],  # left_ring_fingertip (根据fingertip_joint的origin)
        [0.0, 0.0, 0.0],  # left_pinky_MC
        [0.0, 0.0, 0.0],  # left_pinky_MP
        [0.0, 0.0, 0.0],  # left_pinky_DP
        [0.0, 0.0, 0.0],  # left_pinky_fingertip (根据fingertip_joint的origin)
    ]   

    return {
        'urdf_path': urdf_path,
        'base_link': 'virtual_hand_base',  # 根据HA4 URDF中的base link
        'joint_names': joint_names,
        'keypoint_links': keypoint_links,
        'keypoint_offsets': keypoint_offsets
    }

