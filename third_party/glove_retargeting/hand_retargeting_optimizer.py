import numpy as np
import time
from rich.console import Console
import json

from hand_kinematic_casadi import HandKinematicCasadi, get_ha4_r_config, get_ha4_l_config
from surjection_mapper import SurjectionMapper
import casadi as ca

# 添加双进程优化相关的导入
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import atexit
from dataclasses import dataclass
from typing import Optional
import traceback
import logging

# 设置日志
logger = logging.getLogger(__name__)

class HandRetargetingOptimizer:
    def __init__(self, hand_model, hand_type=None, mode='teleop', surjection_config_path=None):
        """
        统一手部重定向优化器，支持 teleop 和 ego 两种模式
        Args:
            hand_model: HandKinematic模型实例
            hand_type: 'left' 或 'right'
            mode: 'teleop' 或 'ego'
            surjection_config_path: 满射映射 JSON 路径（仅 teleop 模式使用）
        """
        self.hand_model = hand_model
        self.n_dof = hand_model.get_n_dof()
        self.last_success = True
        self.hand_type = hand_type
        self.mode = mode

        # HA4 模型关键点索引
        self.model_roots = [0, 4, 8, 12, 16]
        self.model_tips = [3, 7, 11, 15, 19]

        # raw_skeleton 关键点索引 (25 个关节)
        self.raw_finger_roots = [1, 6, 11, 16, 21]
        self.raw_finger_tips = [4, 9, 14, 19, 24]

        self.last_iter_count = 0
        self.last_return_status = 'Unknown'

        self.console = Console()

        self.casadi_initialized = False
        self.casadi_cost_function = None
        self.casadi_solver = None
        self.q_symbolic = self.hand_model.q_symbolic
        self.fkine_symbolic_model = self.hand_model.fkine_symbolic_model

        self.raw_keypoints_symbolic = None
        self.keypoints_last_symbolic = None
        self.q_last_symbolic = None
        self.dq_last_symbolic = None
        self.raw_joint_angles_symbolic = None
        self.q_opt = None

        self.last_solution = {'x': None, 'lam_x': None, 'lam_g': None}
        self.dq_last = np.zeros(self.n_dof)

        if self.mode == 'teleop':
            self.surjection_mapper = SurjectionMapper(config_path=surjection_config_path)
        else:
            self.surjection_mapper = None
          
    def apply_surjection_mapping(self, optimized_q, raw_joint_angles):
        """满射映射后处理（仅 teleop 模式）"""
        if self.surjection_mapper is None:
            return optimized_q
        return self.surjection_mapper.apply(optimized_q, raw_joint_angles)
        
    def _initialize_casadi_optimizer(self):
        """初始化CasADi优化器（只在第一次调用时执行）"""
        if not isinstance(self.hand_model, HandKinematicCasadi):
            raise ValueError("CasADi方法需要HandKinematicCasadi模型")
        
        self.raw_keypoints_symbolic = ca.SX.sym('raw_keypoints', 25, 7)
        self.q_last_symbolic = ca.SX.sym('q_last', self.n_dof)
        self.keypoints_last_symbolic = ca.SX.sym('keypoints_last', 25, 7)
        self.dq_last_symbolic = ca.SX.sym('dq_last', self.n_dof)
        self.raw_joint_angles_symbolic = ca.SX.sym('raw_joint_angles', self.n_dof)
        
        start_time = time.time()
        total_loss_sym = self._build_casadi_cost_function(
            self.q_symbolic, 
            self.raw_keypoints_symbolic, 
            self.q_last_symbolic, 
            self.keypoints_last_symbolic,
            self.dq_last_symbolic,
            self.raw_joint_angles_symbolic
        )
        end_time = time.time()
        logger.info(f"build_casadi_cost_function time: {end_time - start_time}")
        
        start_time = time.time()
        input_list = [self.q_symbolic, self.raw_keypoints_symbolic, self.q_last_symbolic, self.keypoints_last_symbolic, self.dq_last_symbolic, self.raw_joint_angles_symbolic]
        
        self.casadi_cost_function = ca.Function('cost', input_list, [total_loss_sym])
        
        joint_lower, joint_upper = self.hand_model.get_joint_limit()
        
        x = ca.SX.sym('x', self.n_dof)
        p = ca.SX.sym('p', 25*7 + self.n_dof + 25*7 + self.n_dof + self.n_dof)
        
        raw_keypoints_flat = ca.reshape(p[:25*7], (25, 7))
        q_last_flat = p[25*7:25*7+self.n_dof]
        keypoints_last_flat = ca.reshape(p[25*7+self.n_dof:25*7+self.n_dof+25*7], (25, 7))
        dq_last_flat = p[25*7+self.n_dof+25*7:25*7+self.n_dof+25*7+self.n_dof]
        raw_joint_angles_flat = p[25*7+self.n_dof+25*7+self.n_dof:]
        
        objective = self.casadi_cost_function(x, raw_keypoints_flat, q_last_flat, keypoints_last_flat, dq_last_flat, raw_joint_angles_flat)
        
        nlp = {'x': x, 'p': p, 'f': objective}
        
        solver_opts = {
            'print_time': False,
            'error_on_fail': False, 
            'ipopt': {
                'print_level': 0,
                'max_iter': 5,
                'tol': 1e-3,
                'acceptable_tol': 1e-3,  
                'acceptable_iter': 1,
                'hessian_approximation': 'limited-memory',
                'warm_start_init_point': 'yes',
                'warm_start_bound_push': 1e-5,
                'warm_start_mult_bound_push': 1e-5,
                'mu_init': 1e-3,
                # 'output_file': 'ipopt_debug.log',
                # 'output_file': 'ipopt_debug.log',
                # 'constr_viol_tol': 1e-3,  # 约束违反容差
                # 'bound_push': 1e-4,       # 边界推进因子
                # 'bound_frac': 1e-4,       # 边界分数
            }
        }
        
        self.casadi_solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)
        
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper
        
        self.casadi_initialized = True
        end_time = time.time()
        logger.info(f"initialize_casadi_optimizer time: {end_time - start_time}")

    def _build_casadi_cost_function(self, q, raw_keypoints, q_last=None, keypoints_last=None, dq_last=None, raw_joint_angles=None):
        """构建CasADi符号化损失函数"""
        
        # 定义CasADi数值检测函数
        def mx_is_nan(x):
            """检测CasADi符号变量是否为NaN"""
            return x != x

        def mx_is_inf(x):
            """检测CasADi符号变量是否为无穷大"""
            return ca.logic_or(ca.eq(x,  ca.inf),
                               ca.eq(x, -ca.inf))

        def mx_is_finite(x):
            """检测CasADi符号变量是否为有限值"""
            return ca.logic_not(ca.logic_or(mx_is_nan(x), mx_is_inf(x)))
        
                # 定义数值稳定的归一化函数
        def safe_normalize(v, eps=1e-8):
            """数值稳定的向量归一化"""
            norm = ca.norm_2(v)
            return ca.if_else(norm > eps, v / norm, v / eps)  # 避免除零
        
        # 定义数值稳定的点积函数
        def safe_dot(v1, v2, eps=1e-8):
            """数值稳定的点积，确保结果在合理范围内"""
            dot_val = ca.dot(v1, v2)
            # 限制点积结果在[-1, 1]范围内（对于归一化向量）
            return ca.fmax(-1.0, ca.fmin(1.0, dot_val))
        
        # ============= 输入数据有效性检查 =============
        # 检查并清理raw_keypoints中的无效数据
        raw_keypoints_clean = ca.SX.zeros(raw_keypoints.shape)
        for i in range(raw_keypoints.shape[0]):
            for j in range(raw_keypoints.shape[1]):
                # 检查NaN和无穷大，替换为合理默认值
                val = raw_keypoints[i, j]
                # 位置数据 (前3列) 默认为0，四元数数据 (后4列) 需要特殊处理
                if j < 3:  # 位置数据
                    default_val = 0.0
                elif j == 3:  # 四元数w分量，默认为1
                    default_val = 1.0
                else:  # 四元数xyz分量，默认为0
                    default_val = 0.0
                
                # 使用条件表达式处理无效值
                cleaned_val = ca.if_else(mx_is_finite(val), val, default_val)
                raw_keypoints_clean[i, j] = cleaned_val
        
        # 检查并清理关节角度数据
        q_clean = ca.SX.zeros(q.shape)
        for i in range(q.shape[0]):
            val = q[i]
            # 关节角度限制在合理范围内 [-π, π]
            cleaned_val = ca.if_else(mx_is_finite(val), val, 0.0)
            q_clean[i] = cleaned_val
        
        # 检查并清理q_last数据
        if q_last is not None:
            q_last_clean = ca.SX.zeros(q_last.shape)
            for i in range(q_last.shape[0]):
                val = q_last[i]
                cleaned_val = ca.if_else(mx_is_finite(val), val, 0.0)
                q_last_clean[i] = cleaned_val
        else:
            q_last_clean = ca.SX.zeros(q.shape)
        
        # 检查并清理dq_last数据
        if dq_last is not None:
            dq_last_clean = ca.SX.zeros(dq_last.shape)
            for i in range(dq_last.shape[0]):
                val = dq_last[i]
                cleaned_val = ca.if_else(mx_is_finite(val), val, 0.0)
                dq_last_clean[i] = cleaned_val
        else:
            dq_last_clean = ca.SX.zeros(q.shape)
        
        # 检查并清理raw_joint_angles数据
        if raw_joint_angles is not None:
            raw_joint_angles_clean = ca.SX.zeros(raw_joint_angles.shape)
            for i in range(raw_joint_angles.shape[0]):
                val = raw_joint_angles[i]
                cleaned_val = ca.if_else(mx_is_finite(val), val, 0.0)
                raw_joint_angles_clean[i] = cleaned_val
        else:
            raw_joint_angles_clean = ca.SX.zeros(q.shape)
        
        # 使用清理后的数据
        raw_keypoints = raw_keypoints_clean
        q = q_clean
        q_last = q_last_clean
        dq_last = dq_last_clean
        raw_joint_angles = raw_joint_angles_clean
        # ============= 输入数据有效性检查结束 =============
        
        model_keypoints = self.fkine_symbolic_model
        
        model_tip_idx = self.model_tips
        raw_tip_idx = self.raw_finger_tips
        model_root_idx = self.model_roots
        raw_root_idx = self.raw_finger_roots
        
        model_tips_pose = model_keypoints[model_tip_idx, :]
        raw_tips_pose = raw_keypoints[raw_tip_idx, :]

        # ===== 指尖朝向损失 (Tip Orientation Loss) =====
        # 拇指使用四元数距离，其余四指仅对比 z 轴朝向
        tip_ori_loss = 0
        w_tip_ori = [2.0, 1.0, 1.0, 1.0, 1.0]

        for i, (raw_idx, model_idx) in enumerate(zip(raw_tip_idx, model_tip_idx)):
            if i == 0:  # 拇指
                raw_quat = raw_keypoints[raw_idx, 3:7]
                model_quat = model_keypoints[model_idx, 3:7]
                raw_quat_ca = ca.reshape(raw_quat, (1, 4))
                raw_quat_norm = safe_normalize(raw_quat_ca)
                model_quat_norm = safe_normalize(model_quat)
                dot = safe_dot(raw_quat_norm, model_quat_norm)
                tip_ori_loss += w_tip_ori[0] * (1 - ca.fabs(dot))
            else:
                raw_quat = raw_keypoints[raw_idx, 3:7]
                model_quat = model_keypoints[model_idx, 3:7]
                raw_quat_norm = safe_normalize(raw_quat)
                model_quat_norm = safe_normalize(model_quat)
                raw_w, raw_x, raw_y, raw_z = raw_quat_norm[0,0], raw_quat_norm[0,1], raw_quat_norm[0,2], raw_quat_norm[0,3]
                model_w, model_x, model_y, model_z = model_quat_norm[0], model_quat_norm[1], model_quat_norm[2], model_quat_norm[3]
                raw_z_axis = ca.vertcat(
                    2 * (raw_x * raw_z + raw_w * raw_y),
                    2 * (raw_y * raw_z - raw_w * raw_x),
                    1 - 2 * (raw_x * raw_x + raw_y * raw_y)
                )
                model_z_axis = ca.vertcat(
                    2 * (model_x * model_z + model_w * model_y),
                    2 * (model_y * model_z - model_w * model_x),
                    1 - 2 * (model_x * model_x + model_y * model_y)
                )
                raw_z_axis_norm = safe_normalize(raw_z_axis.T)
                model_z_axis_norm = safe_normalize(model_z_axis.T)
                dot = safe_dot(raw_z_axis_norm, model_z_axis_norm)
                tip_ori_loss += w_tip_ori[i] * (1 - dot)

        # ===== 关节增量平滑损失 (Joint Smoothness Loss) =====
        if self.mode == 'teleop':
            wdq = ca.DM.ones(self.n_dof) * 0.01
        else:
            wdq = ca.DM.ones(self.n_dof) * 1.0
        dq = q - q_last
        dq_loss = ca.sqrt(ca.sumsqr(dq * wdq) + 1e-12)
        
        # ===== 关节边界软约束损失 (Margin Loss) =====
        # ===== BEGIN: margin loss，软边界惩罚，提前远离上下限 =====
        # 关节限位（常量）
        jl, ju = self.hand_model.get_joint_limit()
        jl = ca.DM(jl)
        ju = ca.DM(ju)

        # 边界“缓冲区”大小（弧度），可调：5°左右通常就能明显减抖
        deg2rad = ca.pi / 180.0
        margin = 3.0 * deg2rad   # 5°

        # 软 ReLU：平滑、可导，避免非光滑点
        def softplus(z, k=20.0):
            return (1.0 / k) * ca.log(1 + ca.exp(k * z))

        # 只有在进入 margin 区才产生代价：
        # q < (jl + margin) 或 q > (ju - margin) 时，代价随“侵入深度”平滑增长
        lower_th = jl + margin
        upper_th = ju - margin

        lower_violation = softplus(z=lower_th - q,k=20.0)   # 低端越越界越大
        upper_violation = softplus(z=q - upper_th,k=20.0)   # 高端越越界越大

        # 归一化一下让尺度更稳定
        margin_loss = ca.sumsqr((lower_violation + upper_violation) / (margin + 1e-8))
        # ===== END: margin loss =====

        # ===== 线性约束边界损失 (Linear Constraint Margin Loss) =====
        # ===== BEGIN: linear-constraint margin loss (keep away from linear bounds) =====
        deg2rad = ca.pi / 180.0
        eps_deg = 3.0                # 距离上界3°开始给惩罚，可调
        eps = eps_deg * deg2rad

        lin_losses = []

        # 1) 拇指 CMC 两条线性上界
        #   g1 = thumb_cmc_aa - 0.33*thumb_cmc_fe <= 3.3°
        #   g2 = thumb_cmc_aa + 0.66*thumb_cmc_fe <= 72.6°
        thumb_cmc_fe = q[0]
        thumb_cmc_aa = q[1]

        g1 = thumb_cmc_aa - 0.33 * thumb_cmc_fe
        ub1 = 3.3 * deg2rad
        lin_losses.append( softplus(z=g1 - (ub1 - eps),k=20.0) )

        g2 = thumb_cmc_aa + 0.66 * thumb_cmc_fe
        ub2 = 72.6 * deg2rad
        lin_losses.append( softplus(z=g2 - (ub2 - eps),k=20.0) )

        # 2) 四指 MCP 两条线性上界（每指两条）
        #   gl = -mcp_aa + 0.22*mcp_fe <= 29.8°
        #   gu =  mcp_aa + 0.22*mcp_fe <= 29.8°
        finger_mcp_pairs = [(5,6), (9,10), (13,14), (18,19)]
        ub_mcp = 29.8 * deg2rad

        for fe_idx, aa_idx in finger_mcp_pairs:
            mcp_fe = q[fe_idx]
            mcp_aa = q[aa_idx]

            gl = -mcp_aa + 0.22 * mcp_fe
            gu =  mcp_aa + 0.22 * mcp_fe

            lin_losses.append( softplus(gl - (ub_mcp - eps)) )
            lin_losses.append( softplus(gu - (ub_mcp - eps)) )

        # 汇总（平方一下让量纲更稳定）
        lincon_margin_loss = ca.sumsqr(ca.vertcat(*lin_losses))
        # ===== END: linear-constraint margin loss =====

        # ===== 参考关节角度损失 (Reference Joint Angles Loss) =====
        reference_joint_loss = 0.0

        if self.mode == 'teleop':
            # teleop: 手套→HA4 小指关节重映射
            raw_ref = ca.SX.zeros(self.n_dof)
            for i in range(17):
                raw_ref[i] = raw_joint_angles[i]
            raw_ref[17] = 0                        # HA4[17] 无手套对应
            raw_ref[18] = raw_joint_angles[18]     # Pinky_MCP_Flex → MCP_FE
            raw_ref[19] = raw_joint_angles[17]     # Pinky_MCP_Abd  → MCP_AA
            raw_ref[20] = raw_joint_angles[19]     # Pinky_PIP_Flex → PIP
            raw_ref[21] = raw_joint_angles[20]     # Pinky_DIP_Flex → DIP

            w_ref_joint = ca.DM.ones(self.n_dof) * 1.0
            w_ref_joint[14] = 10.0      # 无名指 MCP_AA 高权重
            w_ref_joint[17] = 0.0       # HA4[17] 无手套对应，不约束
            w_ref_joint[19] = 10.0      # 小指 MCP_AA（横摆）高权重
        else:
            # ego: 直接使用 raw_joint_angles，不做重映射
            raw_ref = raw_joint_angles
            w_ref_joint = ca.DM.ones(self.n_dof) * 1.0
            w_ref_joint[14] = 1.0
            w_ref_joint[17:22] = 0.0
            w_ref_joint[19] = 1.0

        joint_diff = q - raw_ref
        weighted_diff = joint_diff * w_ref_joint
        reference_joint_loss = ca.sqrt(ca.sumsqr(weighted_diff) + 1e-12)

        # ===== 拇指到多指向量匹配损失 (Thumb-to-Fingers Vector Matching Loss) =====
        # 目的：匹配拇指到其他手指（食指2、中指3、无名指4）的向量关系
        # 损失包含：平移（向量长度）和旋转（向量方向）两部分
        # 可调权重：每个手指的平移/旋转权重，以及手指间的向量权重（2-3, 3-4, 2-4）
        finger_vector_loss = 0.0
        
        # 定义手指索引：1=拇指(0), 2=食指(1), 3=中指(2), 4=无名指(3), 5=小指(4)
        thumb_idx = 0
        index_idx = 1
        middle_idx = 2
        ring_idx = 3
        
        # === 拇指到其他手指的向量损失（1-2, 1-3, 1-4）===
        # 平移损失权重（每个手指可调）
        w_trans_1_to_2 = 2.0  # 拇指到食指的平移权重
        w_trans_1_to_3 = 2.0  # 拇指到中指的平移权重
        w_trans_1_to_4 = 2.0  # 拇指到无名指的平移权重
        
        # 旋转损失权重（每个手指可调）
        w_rot_1_to_2 = 1.0  # 拇指到食指的旋转权重
        w_rot_1_to_3 = 1.0  # 拇指到中指的旋转权重
        w_rot_1_to_4 = 1.0  # 拇指到无名指的旋转权重
        
        # 计算拇指到其他手指的向量
        finger_pairs = [
            (thumb_idx, index_idx, w_trans_1_to_2, w_rot_1_to_2),
            (thumb_idx, middle_idx, w_trans_1_to_3, w_rot_1_to_3),
            (thumb_idx, ring_idx, w_trans_1_to_4, w_rot_1_to_4)
        ]

        scale_factor = 0.85
        
        for from_idx, to_idx, w_trans, w_rot in finger_pairs:
            # raw数据：从from_finger到to_finger的向量
            raw_vec = raw_tips_pose[to_idx, :3] - raw_tips_pose[from_idx, :3]
            raw_vec = ca.reshape(raw_vec, (1, 3))
            raw_vec_norm = safe_normalize(raw_vec)
            raw_vec_len = ca.norm_2(raw_vec)
            
            # model数据：从from_finger到to_finger的向量
            model_vec = model_tips_pose[to_idx, :3] - model_tips_pose[from_idx, :3]
            model_vec = ca.reshape(model_vec, (1, 3))
            model_vec_norm = safe_normalize(model_vec)
            model_vec_len = ca.norm_2(model_vec)
            
            # 平移损失：向量长度差异
            trans_loss = w_trans * ca.fabs(model_vec_len * scale_factor - raw_vec_len)
            
            # 旋转损失：向量方向差异
            # 先计算点积
            dot = safe_dot(raw_vec_norm, model_vec_norm)
            
            angle = ca.arccos(ca.fmax(-0.999, ca.fmin(0.999, dot)))
            delta = 0.1
            rot_loss = w_rot * ca.if_else(angle < delta, 0.5 * angle**2, delta * (angle - 0.5 * delta))
            
            finger_vector_loss += trans_loss + rot_loss

        # ===== 指尖位姿损失 (Fingertip Pose Loss) =====
        fingertip_pose_loss_pos = 0.0
        fingertip_pose_loss_ori = 0.0
        w_ftp_pos = [100.0, 100.0, 100.0, 100.0, 100.0]
        w_ftp_ori = [1.0, 1.0, 1.0, 1.0, 1.0]

        for i, (raw_idx, model_idx) in enumerate(zip(raw_tip_idx, model_tip_idx)):
            raw_tip_pos = raw_keypoints[raw_idx, :3]
            model_tip_pos = model_keypoints[model_idx, :3]
            pos_diff = model_tip_pos - raw_tip_pos
            pos_loss_i = ca.sumsqr(pos_diff)

            raw_tip_quat = ca.reshape(raw_keypoints[raw_idx, 3:7], (1, 4))
            model_tip_quat = ca.reshape(model_keypoints[model_idx, 3:7], (1, 4))
            raw_tip_quat_n = safe_normalize(raw_tip_quat)
            model_tip_quat_n = safe_normalize(model_tip_quat)
            quat_dot = safe_dot(raw_tip_quat_n, model_tip_quat_n)
            ori_loss_i = 1 - ca.fabs(quat_dot)

            fingertip_pose_loss_pos += w_ftp_pos[i] * pos_loss_i
            fingertip_pose_loss_ori += w_ftp_ori[i] * ori_loss_i

        fingertip_pose_loss = fingertip_pose_loss_pos + fingertip_pose_loss_ori

        # ============================================================
        # 总损失函数 (Total Loss Function) - 按 mode 组装
        # ============================================================
        if self.mode == 'teleop':
            total_loss = (
                tip_ori_loss * 2.5 + 
                reference_joint_loss * 0.02 +
                dq_loss * 3.0 +
                margin_loss * 1.0 +
                lincon_margin_loss * 1.0 +
                finger_vector_loss * 2.0
            )
        else:
            total_loss = fingertip_pose_loss * 1.0

        return total_loss

    def check_violations(self, q: np.ndarray, tol_deg: float = 0.1):
        """
        诊断当前关节角是否越界（线性约束和关节上下界）。
        tol_deg: 报告阈值（角度），小于该阈值的不报警。
        返回一个字典，包含最大越界和具体索引。
        """
        q = np.asarray(q).reshape(-1)
        rad = np.deg2rad
        deg = np.rad2deg

        # ---- 关节上下界（注意：你现在仍然在 nlpsol 调用里设置了 lbx/ubx，这是硬边界）----
        jl = np.asarray(self.joint_lower)
        ju = np.asarray(self.joint_upper)

        low_vi  = np.maximum(0.0, jl - q)    # q < jl 的越界量
        high_vi = np.maximum(0.0, q - ju)    # q > ju 的越界量
        max_bound_vi_rad = max(low_vi.max(initial=0.0), high_vi.max(initial=0.0))

        # ---- 线性约束（手工重算 g(x) <= ub）----
        g_vals = []
        g_ubs  = []

        # 拇指 CMC： y - 0.33x <= 3.3°,  y + 0.66x <= 72.6°
        g_vals += [q[1] - 0.33*q[0],  q[1] + 0.66*q[0]]
        g_ubs  += [rad(3.3),          rad(72.6)]

        # 四指 MCP： -aa + 0.22*fe <= 29.8°,  aa + 0.22*fe <= 29.8°
        for fe, aa in [(5,6), (9,10), (13,14), (18,19)]:
            g_vals += [-q[aa] + 0.22*q[fe],  q[aa] + 0.22*q[fe]]
            g_ubs  += [rad(29.8),            rad(29.8)]

        g_vals = np.array(g_vals, dtype=float)
        g_ubs  = np.array(g_ubs,  dtype=float)
        lin_vi = np.maximum(0.0, g_vals - g_ubs)  # 只看超过上界的量
        max_lin_vi_rad = lin_vi.max(initial=0.0)

        # 超过阈值的索引（用于定位）
        tol = rad(tol_deg)
        joint_low_idx  = np.where(low_vi  > tol)[0].tolist()
        joint_high_idx = np.where(high_vi > tol)[0].tolist()
        lin_idx        = np.where(lin_vi  > tol)[0].tolist()

        return {
            "joint_bounds": {
                "max_violation_deg": float(deg(max_bound_vi_rad)),
                "lower_indices": joint_low_idx,
                "upper_indices": joint_high_idx,
            },
            "linear_constraints": {
                "max_violation_deg": float(deg(max_lin_vi_rad)),
                "violated_indices": lin_idx,
            }
        }

    def print_constraint_report(self, q: np.ndarray, tol_deg: float = 0.1):
        """便捷打印：只在超过阈值时输出报警信息。"""
        rep = self.check_violations(q, tol_deg=tol_deg)
        jb = rep["joint_bounds"]
        lc = rep["linear_constraints"]
        if jb["max_violation_deg"] > tol_deg or lc["max_violation_deg"] > tol_deg:
            print(
                f"[VIOL] joint max={jb['max_violation_deg']:.2f}° "
                f"(low idx: {jb['lower_indices']}, high idx: {jb['upper_indices']}), "
                f"linear max={lc['max_violation_deg']:.2f}° "
                f"(lin idx: {lc['violated_indices']})"
            )

    def optimize_with_casadi(self, initial_q=None, raw_keypoints=None, keypoints_last=None, raw_joint_angles=None):
        """使用CasADi的非线性优化求解关节角度"""
        if initial_q is None or self.last_success == False:
            initial_q = np.zeros(self.n_dof)
        if raw_keypoints is None:
            raise ValueError("raw_keypoints 不能为空")
        if keypoints_last is None:
            keypoints_last = np.zeros((25,7))
        if raw_joint_angles is None:
            raw_joint_angles = np.zeros(self.n_dof)
        
        if np.linalg.norm(raw_keypoints - keypoints_last) < 1e-6:
            if self.last_success: 
                return initial_q
        
        if not self.casadi_initialized:
            logger.info("First call, initializing CasADi optimizer...")
            self._initialize_casadi_optimizer()
            logger.info("CasADi optimizer initialization completed")
        
        start_time = time.time()
        
        raw_keypoints_flat = raw_keypoints.flatten(order='F')
        keypoints_last_flat = keypoints_last.flatten(order='F')
        params = np.concatenate([raw_keypoints_flat, initial_q, keypoints_last_flat, self.dq_last, raw_joint_angles])

        if (self.last_solution is None or 
            self.last_solution['x'] is None or 
            self.last_solution['lam_x'] is None or 
            self.last_solution['lam_g'] is None):
            x0 = initial_q
            lam_x0 = np.zeros(self.n_dof)
        else:
            x0 = self.last_solution['x']
            lam_x0 = self.last_solution['lam_x']
        
        try:
            sol = self.casadi_solver(
                x0=x0,
                lam_x0=lam_x0,
                p=params,
                lbx=self.joint_lower,
                ubx=self.joint_upper,
            )
            
            stats = self.casadi_solver.stats()
            return_status = stats.get('return_status', 'Unknown')
            success = stats.get('success', False)
            
            self.last_iter_count = stats.get('iter_count', 0)
            self.last_return_status = return_status
            
            optimized_q = np.array(sol['x']).flatten()

            lam_x = sol.get('lam_x')
            lam_g = sol.get('lam_g')
            if lam_x is not None and lam_g is not None:
                self.last_solution = {
                    'x': optimized_q.copy(),
                    'lam_x': lam_x,
                    'lam_g': lam_g
                }
            
            if success:
                self.last_success = True
            elif return_status == 'Maximum_Iterations_Exceeded':
                self.last_success = True
            else:
                self.last_success = False
                print(f"Warning: Optimization status {return_status}")
            
            if self.mode == 'teleop':
                optimized_q = self.apply_surjection_mapping(optimized_q, raw_joint_angles)
            
            self.dq_last = optimized_q - initial_q
                
            return optimized_q
                
        except Exception as e:
            print(f"CasADi solver error: {str(e)}")
            self.last_success = False
            self.last_iter_count = 0
            self.last_return_status = "Error"
            if self.last_solution['x'] is not None:
                return self.last_solution['x']
            else:
                return initial_q


@dataclass
class OptimizationResult:
    """优化结果数据类（统一 teleop/ego 超集）"""
    joint_angles: Optional[np.ndarray] = None
    filtered_angles: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    cost_value: float = 0.0
    optimization_time: float = 0.0
    frame_index: int = 0
    success: bool = False
    iter_count: int = 0
    status_code: int = -1  # 0: 最优收敛, 1: 可接受收敛, 2: 达最大迭代, -1: 失败


class FirstOrderLowPassFilter:
    """
    一阶低通滤波器类，用于平滑关节角度数据
    """
    def __init__(self, alpha=0.3, initial_value=None):
        """
        初始化滤波器
        Args:
            alpha: 滤波系数 (0-1)，值越小滤波越强，但响应越慢
            initial_value: 初始值，如果为None则使用第一个输入值作为初始值
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)  # 确保alpha在[0,1]范围内
        self.initialized = False
        self.filtered_value = None
        self.initial_value = initial_value
        
    def filter(self, input_value):
        """
        对输入值进行滤波
        Args:
            input_value: 输入值（可以是标量或数组）
        Returns:
            filtered_value: 滤波后的值
        """
        input_value = np.array(input_value)
        
        # 如果是第一次调用，初始化滤波器
        if not self.initialized:
            if self.initial_value is not None:
                self.filtered_value = np.array(self.initial_value)
            else:
                self.filtered_value = input_value.copy()
            self.initialized = True
            return self.filtered_value.copy()
        
        # 一阶低通滤波公式: y[n] = α * x[n] + (1-α) * y[n-1]
        self.filtered_value = self.alpha * input_value + (1 - self.alpha) * self.filtered_value
        
        return self.filtered_value.copy()
    
    def reset(self, new_initial_value=None):
        """
        重置滤波器状态
        Args:
            new_initial_value: 新的初始值，如果为None则保持当前滤波值
        """
        if new_initial_value is not None:
            self.filtered_value = np.array(new_initial_value)
        self.initialized = False
    
    def get_current_value(self):
        """
        获取当前滤波值
        Returns:
            current_value: 当前滤波值
        """
        if not self.initialized:
            return None
        return self.filtered_value.copy()


class JointAngleFilter:
    """
    关节角度滤波器，为每个关节创建独立的一阶低通滤波器
    """
    def __init__(self, n_joints, alpha=0.3, initial_values=None):
        """
        初始化关节角度滤波器
        Args:
            n_joints: 关节数量
            alpha: 滤波系数 (0-1)
            initial_values: 初始关节角度值，如果为None则使用零向量
        """
        self.n_joints = n_joints
        self.alpha = alpha
        
        # 为每个关节创建独立的滤波器
        self.filters = []
        for i in range(n_joints):
            initial_value = None
            if initial_values is not None and i < len(initial_values):
                initial_value = initial_values[i]
            self.filters.append(FirstOrderLowPassFilter(alpha=alpha, initial_value=initial_value))
    
    def filter_joint_angles(self, joint_angles):
        """
        对关节角度进行滤波
        Args:
            joint_angles: 关节角度数组 (n_joints,)
        Returns:
            filtered_angles: 滤波后的关节角度
        """
        if len(joint_angles) != self.n_joints:
            raise ValueError(f"关节角度数量不匹配: 期望{self.n_joints}，实际{len(joint_angles)}")
        
        filtered_angles = np.zeros(self.n_joints)
        for i in range(self.n_joints):
            filtered_angles[i] = self.filters[i].filter(joint_angles[i])
        
        return filtered_angles
    
    def reset(self, new_initial_values=None):
        """
        重置所有滤波器
        Args:
            new_initial_values: 新的初始值数组
        """
        for i, filter_obj in enumerate(self.filters):
            initial_value = None
            if new_initial_values is not None and i < len(new_initial_values):
                initial_value = new_initial_values[i]
            filter_obj.reset(initial_value)
    
    def set_alpha(self, new_alpha):
        """
        设置新的滤波系数
        Args:
            new_alpha: 新的滤波系数 (0-1)
        """
        self.alpha = np.clip(new_alpha, 0.0, 1.0)
        for filter_obj in self.filters:
            filter_obj.alpha = self.alpha


class SharedMemoryManager:
    """管理多进程共享内存的类"""
    
    def __init__(self, hand_type, input_keypoints_shape=(25, 7), output_keypoints_shape=(20, 7), joint_angles_shape=(22,)):
        self.hand_type = hand_type
        self.input_keypoints_shape = input_keypoints_shape
        self.output_keypoints_shape = output_keypoints_shape
        self.joint_angles_shape = joint_angles_shape
        
        self.input_lock = mp.Lock()
        self.result_lock = mp.Lock()
        
        self.input_keypoints_size = np.prod(input_keypoints_shape)    # 175
        self.output_keypoints_size = np.prod(output_keypoints_shape)  # 140
        self.joint_angles_size = np.prod(joint_angles_shape)          # 22
        
        self.frame_index_offset = 0
        self.has_new_data_offset = self.frame_index_offset + 1
        self.is_processing_offset = self.has_new_data_offset + 1
        self.input_keypoints_offset = self.is_processing_offset + 1
        self.raw_joint_angles_offset = self.input_keypoints_offset + self.input_keypoints_size
        self.joint_angles_offset = self.raw_joint_angles_offset + self.joint_angles_size
        self.filtered_angles_offset = self.joint_angles_offset + self.joint_angles_size
        self.output_keypoints_offset = self.filtered_angles_offset + self.joint_angles_size
        self.cost_value_offset = self.output_keypoints_offset + self.output_keypoints_size
        self.optimization_time_offset = self.cost_value_offset + 1
        self.success_offset = self.optimization_time_offset + 1
        self.result_frame_index_offset = self.success_offset + 1
        self.iter_count_offset = self.result_frame_index_offset + 1
        self.status_code_offset = self.iter_count_offset + 1
        
        self.total_size = self.status_code_offset + 1
        
        size = self.total_size * 8
        self.shm = SharedMemory(create=True, size=size)
        self.arr = np.ndarray((self.total_size,), dtype=np.float64, buffer=self.shm.buf)
        self.arr[:] = 0.0
        
    def attach_shared_memory(self, shm_name, input_lock=None, result_lock=None):
        if hasattr(self, 'shm') and self.shm is not None:
            self.shm.close()
        self.shm = SharedMemory(name=shm_name)
        self.arr = np.ndarray((self.total_size,), dtype=np.float64, buffer=self.shm.buf)
        if input_lock is not None: self.input_lock = input_lock
        if result_lock is not None: self.result_lock = result_lock
        self.arr[self.is_processing_offset] = 0.0
        
    def write_input_data(self, keypoints, frame_index, raw_joint_angles=None):
        with self.input_lock:
            if self.arr[self.is_processing_offset] == 1.0:
                return False
            self.arr[self.frame_index_offset] = frame_index
            self.arr[self.input_keypoints_offset:self.input_keypoints_offset + self.input_keypoints_size] = keypoints.flatten()
            if raw_joint_angles is not None:
                self.arr[self.raw_joint_angles_offset:self.raw_joint_angles_offset + self.joint_angles_size] = raw_joint_angles.flatten()
            else:
                self.arr[self.raw_joint_angles_offset:self.raw_joint_angles_offset + self.joint_angles_size] = 0.0
            self.arr[self.has_new_data_offset] = 1.0
            return True
        
    def read_input_data(self):
        with self.input_lock:
            if self.arr[self.has_new_data_offset] == 0.0:
                return None, None, None
            self.arr[self.is_processing_offset] = 1.0
            frame_index = int(self.arr[self.frame_index_offset])
            keypoints = self.arr[self.input_keypoints_offset:self.input_keypoints_offset + self.input_keypoints_size].reshape(self.input_keypoints_shape)
            raw_joint_angles = self.arr[self.raw_joint_angles_offset:self.raw_joint_angles_offset + self.joint_angles_size].reshape(self.joint_angles_shape)
            self.arr[self.has_new_data_offset] = 0.0
            return keypoints.copy(), frame_index, raw_joint_angles.copy()
        
    def mark_data_processed(self):
        with self.input_lock:
            self.arr[self.is_processing_offset] = 0.0
        
    def write_result(self, result: OptimizationResult):
        with self.result_lock:
            if result.joint_angles is not None:
                self.arr[self.joint_angles_offset:self.joint_angles_offset + self.joint_angles_size] = result.joint_angles.flatten()
            if result.filtered_angles is not None:
                self.arr[self.filtered_angles_offset:self.filtered_angles_offset + self.joint_angles_size] = result.filtered_angles.flatten()
            if result.keypoints is not None:
                self.arr[self.output_keypoints_offset:self.output_keypoints_offset + self.output_keypoints_size] = result.keypoints.flatten()
            
            self.arr[self.cost_value_offset] = result.cost_value
            self.arr[self.optimization_time_offset] = result.optimization_time
            self.arr[self.success_offset] = 1.0 if result.success else 0.0
            self.arr[self.result_frame_index_offset] = float(result.frame_index)
            self.arr[self.iter_count_offset] = float(result.iter_count)
            self.arr[self.status_code_offset] = float(result.status_code)
        
    def read_result(self):
        with self.result_lock:
            joint_angles = self.arr[self.joint_angles_offset:self.joint_angles_offset + self.joint_angles_size].reshape(self.joint_angles_shape).copy()
            filtered_angles = self.arr[self.filtered_angles_offset:self.filtered_angles_offset + self.joint_angles_size].reshape(self.joint_angles_shape).copy()
            output_keypoints = self.arr[self.output_keypoints_offset:self.output_keypoints_offset + self.output_keypoints_size].reshape(self.output_keypoints_shape).copy()
            
            cost_value = float(self.arr[self.cost_value_offset])
            optimization_time = float(self.arr[self.optimization_time_offset])
            success = bool(self.arr[self.success_offset] > 0.5)
            frame_index = int(self.arr[self.result_frame_index_offset])
            iter_count = int(self.arr[self.iter_count_offset])
            status_code = int(self.arr[self.status_code_offset])
            
            return OptimizationResult(
                joint_angles=joint_angles,
                filtered_angles=filtered_angles,
                keypoints=output_keypoints,
                cost_value=cost_value,
                optimization_time=optimization_time,
                frame_index=frame_index,
                success=success,
                iter_count=iter_count,
                status_code=status_code
            )
        
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()



def optimization_worker_multiprocess(hand_type, shm_name, input_lock, result_lock, control_queue, result_queue, filter_alpha, hand_serial, mode='teleop', surjection_config_path=None):
    """统一多进程优化工作函数"""
    try:
        hand_models = init_hand_model(hand_serial)
        hand_model = hand_models[hand_type]
        
        shm_manager = SharedMemoryManager(hand_type)
        shm_manager.attach_shared_memory(shm_name, input_lock, result_lock)
        
        optimizer = HandRetargetingOptimizer(
            hand_model, hand_type=hand_type, mode=mode,
            surjection_config_path=surjection_config_path
        )
        optimizer._initialize_casadi_optimizer()
        initial_q = np.zeros(hand_model.get_n_dof())
        frame_last_raw = None
        last_processed_frame = -1
        
        joint_filter = JointAngleFilter(n_joints=hand_model.get_n_dof(), alpha=filter_alpha)
        logger.info(f'{hand_type} hand optimization worker initialized (mode={mode})')
        result_queue.put(f"{hand_type}_worker_started")
        
        while True:
            try:
                try:
                    control_signal = control_queue.get_nowait()
                    if control_signal == "stop":
                        break
                except:
                    pass
                
                frame_raw, current_frame, raw_joint_angles = shm_manager.read_input_data()
                if frame_raw is not None:
                    start_time = time.time()
                    
                    optimized_q = optimizer.optimize_with_casadi(
                        initial_q=initial_q,
                        raw_keypoints=frame_raw,
                        keypoints_last=frame_last_raw,
                        raw_joint_angles=raw_joint_angles
                    )
                    
                    status_str = optimizer.last_return_status
                    if status_str == 'Solve_Succeeded': status_code = 0
                    elif status_str == 'Solved_To_Acceptable_Level': status_code = 1
                    elif status_str == 'Maximum_Iterations_Exceeded': status_code = 2
                    else: status_code = -1
                    
                    filtered_q = joint_filter.filter_joint_angles(optimized_q) if mode == 'teleop' else optimized_q
                    optimized_keypoints = hand_model.fkine_symbolic(filtered_q)
                    
                    optimization_time = time.time() - start_time
                    
                    result = OptimizationResult(
                        joint_angles=optimized_q.copy(),
                        filtered_angles=filtered_q.copy(),
                        keypoints=optimized_keypoints.copy(),
                        optimization_time=optimization_time,
                        frame_index=int(current_frame),
                        success=optimizer.last_success,
                        iter_count=optimizer.last_iter_count,
                        status_code=status_code,
                    )
                    
                    shm_manager.write_result(result)
                    shm_manager.mark_data_processed()
                    
                    initial_q = filtered_q
                    frame_last_raw = frame_raw
                    last_processed_frame = current_frame
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"{hand_type} hand optimization worker error: {e}")
                raise e
                
    except Exception as e:
        result_queue.put(f"{hand_type}_worker_init_error: {e}")
        logger.error(f"{hand_type} hand optimization worker initialization failed: {e}\n{traceback.format_exc()}")
    finally:
        try:
            shm_manager.shm.close()
            shm_manager.shm.unlink()
        except:
            pass
        result_queue.put(f"{hand_type}_worker_stopped")


def init_hand_model(hand_serial='HA4'):
    if hand_serial != 'HA4':
        raise ValueError(f"Unsupported hand model: {hand_serial}, only HA4 is supported")
    
    left_config = get_ha4_l_config()
    right_config = get_ha4_r_config()

    left_hand_model = HandKinematicCasadi(
        urdf_path=left_config['urdf_path'],
        base_link=left_config['base_link'],
        joint_names=left_config['joint_names'],
        keypoint_links=left_config['keypoint_links'],
        keypoint_offsets=left_config['keypoint_offsets']
    )
    
    right_hand_model = HandKinematicCasadi(
        urdf_path=right_config['urdf_path'],
        base_link=right_config['base_link'],
        joint_names=right_config['joint_names'],
        keypoint_links=right_config['keypoint_links'],
        keypoint_offsets=right_config['keypoint_offsets']
    )
    
    return {'left': left_hand_model, 'right': right_hand_model}


class MultiprocessOptimizationManager:
    """统一多进程优化管理器"""
    
    def __init__(self, hand_models, filter_alpha=0.3, hand_serial=None, mode='teleop', surjection_config_path=None):
        self.hand_models = hand_models
        self.filter_alpha = filter_alpha
        self.hand_serial = hand_serial
        self.mode = mode
        self.surjection_config_path = surjection_config_path
        self.left_shm_manager = None
        self.right_shm_manager = None
        self.processes = []
        self.control_queues = {}
        self.result_queues = {}
        
    def start(self):
        try:
            if self.hand_models.get('left') is not None:
                self.left_shm_manager = SharedMemoryManager('left')
            if self.hand_models.get('right') is not None:
                self.right_shm_manager = SharedMemoryManager('right')
            
            for hand_type, hand_model in self.hand_models.items():
                if hand_model is None:
                    continue
                control_queue = mp.Queue()
                result_queue = mp.Queue()
                self.control_queues[hand_type] = control_queue
                self.result_queues[hand_type] = result_queue
                
                shm_manager = self.left_shm_manager if hand_type == 'left' else self.right_shm_manager
                
                process = mp.Process(
                    target=optimization_worker_multiprocess,
                    args=(hand_type, shm_manager.shm.name, shm_manager.input_lock, shm_manager.result_lock,
                          control_queue, result_queue, self.filter_alpha, self.hand_serial,
                          self.mode, self.surjection_config_path),
                    daemon=False
                )
                
                process.start()
                self.processes.append(process)
                
            logger.info(f"Started {len(self.processes)} optimization processes (mode={self.mode})")
            
        except Exception as e:
            logger.error(f"Failed to start multiprocess optimization: {e}")
            self.cleanup()
            
    def update_process_keypoints(self, hand_type, keypoints, frame_index, raw_joint_angles=None):
        success = False
        if hand_type == 'left' and self.left_shm_manager:
            success = self.left_shm_manager.write_input_data(keypoints, frame_index, raw_joint_angles)
        elif hand_type == 'right' and self.right_shm_manager:
            success = self.right_shm_manager.write_input_data(keypoints, frame_index, raw_joint_angles)
        return success
            
    def get_result(self, hand_type):
        if hand_type == 'left' and self.left_shm_manager:
            return self.left_shm_manager.read_result()
        elif hand_type == 'right' and self.right_shm_manager:
            return self.right_shm_manager.read_result()
        return None
        
    def stop(self):
        for hand_type, control_queue in self.control_queues.items():
            try:
                control_queue.put("stop")
            except:
                pass
                
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                logger.warning(f"Force terminating process {process.pid}")
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    
        self.processes.clear()
        
    def cleanup(self):
        self.stop()
        for mgr in (self.left_shm_manager, self.right_shm_manager):
            if mgr:
                try:
                    mgr.cleanup()
                except:
                    pass
        self.left_shm_manager = None
        self.right_shm_manager = None
