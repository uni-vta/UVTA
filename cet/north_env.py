from abc import ABC, abstractmethod

import threading
import time
import numpy as np
import csv
import os
import cv2
from turbojpeg import TurboJPEG
from typing import Dict, Any, Optional
from proto import north_pb2 as pb
from zenoh_wrapper import ZenohWrapper, MessageEncoding
import copy
# import transforms3d as t3d
import multiprocessing
import sys
import orjson
import logging

from collections import deque


def normalize_vector(x):
    return x / np.linalg.norm(x, axis=-1)


def rotation_6d_to_matrix_np(d6: np.array) -> np.array:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if np.sum(np.abs(d6)) < 1e-10:
        return np.eye(3)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize_vector(a1)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = normalize_vector(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def get_root_logger():
    # raise NotImplementedError
    return logging.Logger(name='ailab')


class Env(ABC):
    def __init__(self, action_dim=12, **kwargs):
        self.logger = get_root_logger()

        self.action_dim = action_dim
        if kwargs.get("using_zmq", True):
            raise NotImplementedError('ZMQ Env is deprecated. Please use eclipse-zenoh.')

        self.dumper = None 
        if kwargs.get("dumper", None) is not None:
            raise NotImplementedError('Dumper is not implemented yet.')
            self.dumper = Dumper(**kwargs["dumper"])

class NorthEnv(Env):
    """    
    发送: UhrActionBundle (动作数据)
    接收: NorthObservation (观察数据)
    使用Zenoh进行通信
    """
    
    def __init__(self, 
                 observation_topic="north_observation",
                 action_topic="inference/action",
                 eef_action_topic="teleop/teleop_info",
                 recv_dt=1/30.0,
                 max_obs_buffer_length=100,
                 max_action_buffer_length=10,
                 zenoh_config=None,
                 using_eef_control=False,
                 static_lowbody=True,
                 action_output=[],
                 action_pub_duration=0.01666,
                 inference_dt=0.1,
                 enable_obs_robot_state=True,
                 enable_tactile=False,
                 enable_obs_vision=True,
                 enable_obs_mode=True,
                 action_log_interval_s=None,
                 **kwargs):
        super().__init__(**kwargs, using_zmq=False)
        self.logger.info(f"{kwargs}")
        self.action_pub_key = action_output
        self.action_pub_duration = action_pub_duration
        self.inference_dt = inference_dt
        self.enable_obs_robot_state = enable_obs_robot_state
        self.enable_tactile = enable_tactile
        self.enable_obs_vision = enable_obs_vision
        self.enable_obs_mode = enable_obs_mode
        self.action_log_interval_s = (
            None
            if action_log_interval_s is None
            else max(0.0, float(action_log_interval_s))
        )
        self._last_action_log_time = float("-inf")

        self.last_step_obs_time = None

        self.observation_topic = observation_topic
        self.action_topic = action_topic if not using_eef_control else eef_action_topic
        
        # Zenoh相关 - 使用统一的 ZenohWrapper
        self.zenoh_wrapper = None
        self.zenoh_config = zenoh_config or {}  # 保留配置以备将来扩展
        self.using_eef_control = using_eef_control
        self.static_lowbody = static_lowbody
        self.lowbody_motor_cache = None

        # 状态管理
        # self.last_observation = None
        self.is_connected = False
        self.turbo_jpeg = TurboJPEG()

        # 初始化Zenoh连接
        self._init_zenoh_connections()

        # 线程控制相关
        self._action_thread_stop_event = threading.Event()
        self._action_thread = None
        self._action_lock = threading.Lock()
        # 动作缓冲区
        self.max_action_buffer_length = max_action_buffer_length
        self._action_buffer = deque(maxlen=self.max_action_buffer_length)

        self.max_obs_buffer_length = max_obs_buffer_length
        self._normal_obs_buffer = deque(maxlen=self.max_obs_buffer_length)
        self.recv_dt = recv_dt
        
        self._last_recv_time = None
        self._obs_recv_lock = threading.Lock()
        
        # 触觉传感器映射关系
        self.tactile_sensor_map = {
            "0": "R_LITTLE",
            "1": "R_RING", 
            "2": "R_MIDDLE",
            "3": "R_INDEX",
            "4": "R_THUMB",
            "5": "L_LITTLE",
            "6": "L_RING",
            "7": "L_MIDDLE", 
            "8": "L_INDEX",
            "9": "L_THUMB"
        }
        
        # 启动常驻发送线程
        self._start_action_thread()

    def _init_zenoh_connections(self):
        """初始化Zenoh连接（使用ZenohWrapper）"""
        try:
            # 创建 ZenohWrapper 实例
            self.zenoh_wrapper = ZenohWrapper()
            
            # 连接到 Zenoh 网络
            if not self.zenoh_wrapper.connect():
                raise Exception("Failed to connect to Zenoh network")
            
            # 订阅观察数据（使用Protobuf编码）
            if not self.zenoh_wrapper.subscribe(
                self.observation_topic,
                self._on_observation_received_wrapper,
                encoding=MessageEncoding.PROTOBUF.value,
                message_type=pb.NorthObservation
            ):
                raise Exception(f"Failed to subscribe to {self.observation_topic}")
            
            # 动作发布者和debug发布者不需要提前声明
            # ZenohWrapper.publish() 会自动处理
            
            self.is_connected = True
            self.logger.info(f"Zenoh initialized (using ZenohWrapper) - obs: {self.observation_topic}, action: {self.action_topic}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Zenoh connections: {e}")
            self.is_connected = False
            if self.zenoh_wrapper:
                self.zenoh_wrapper.disconnect()
                self.zenoh_wrapper = None
            return False

    def _on_observation_received_wrapper(self, topic, north_obs):
        """ZenohWrapper回调适配器：接收观察数据"""
        try:
            # ZenohWrapper 已经自动反序列化为 NorthObservation 对象
            if north_obs is None:
                self.logger.warning("Received None observation")
                return
            
            # 转换为字典格式
            obs_dict = self._convert_north_observation_to_dict(north_obs)
            # 更新缓冲区
            
            # self.last_observation = obs_dict
            # Keep both clocks.  North's message timestamp is producer-side;
            # these values describe when this process received the bundle.
            # Rollouts schedule on the monotonic clock (immune to NTP / wall
            # clock jumps) and use wall time only to map a synchronized
            # producer timestamp onto that monotonic timeline.
            push_buffer_ts = time.time()
            push_buffer_mono = time.monotonic()
            obs_dict["_north_receive_wall_time_s"] = push_buffer_ts
            obs_dict["_north_receive_monotonic_s"] = push_buffer_mono
            data_ts = push_buffer_ts
            with self._obs_recv_lock:
                self._normal_obs_buffer.append([data_ts, obs_dict])

            loop_time = time.time()
            if self._last_recv_time is not None:
                if abs((loop_time - self._last_recv_time) - self.recv_dt) > 0.01:
                    self.logger.warning(f"recv timing error: {loop_time - self._last_recv_time} / {self.recv_dt}")
            self._last_recv_time = loop_time
            
        except Exception as e:
            self.logger.error(f"Failed to receive observation: {e}")

    def _parse_robot_state(self, robot_state):
        obs_dict = {}
        if robot_state.HasField('left_arm'):
            left_arm = robot_state.left_arm
            # # 获取TCP位姿
            # tcp_pose = left_arm.tcp_pose
            # px, py, pz = tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z
            # qw, qx, qy, qz = (
            #     tcp_pose.orientation.w,
            #     tcp_pose.orientation.x, 
            #     tcp_pose.orientation.y,
            #     tcp_pose.orientation.z
            # )
            # # 四元数转轴角
            # axis, angle = t3d.quaternions.quat2axangle([qw, qx, qy, qz])
            # axis = np.array(axis) / np.linalg.norm(axis)
            # ax, ay, az = axis[0] * angle, axis[1] * angle, axis[2] * angle
            # # 设置输出状态
            # obs_dict['/state/left_arm/tcp_pose'] = [px, py, pz, ax, ay, az]
            obs_dict['/state/left_arm/joint_angle'] = list(left_arm.joint.position)
            obs_dict['/state/left_arm/joint_effort'] = list(left_arm.joint.effort)
            force = [left_arm.wrench.force.x, left_arm.wrench.force.y, left_arm.wrench.force.z]
            torque = [left_arm.wrench.torque.x, left_arm.wrench.torque.y, left_arm.wrench.torque.z]
            obs_dict['/state/left_arm/tcp_forces'] = force+torque
            # self.logger.info(f"left_arm_forces:{obs_dict['/state/left_arm/tcp_forces']}")
            
        # 右臂状态
        if robot_state.HasField('right_arm'):
            right_arm = robot_state.right_arm
            # tcp_pose = right_arm.tcp_pose
            # px, py, pz = tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z
            # qw, qx, qy, qz = (
            #     tcp_pose.orientation.w,
            #     tcp_pose.orientation.x, 
            #     tcp_pose.orientation.y,
            #     tcp_pose.orientation.z
            # )
            # axis, angle = t3d.quaternions.quat2axangle([qw, qx, qy, qz])
            # axis = np.array(axis) / np.linalg.norm(axis)
            # ax, ay, az = axis[0] * angle, axis[1] * angle, axis[2] * angle
            # # 设置输出状态
            # obs_dict['/state/right_arm/tcp_pose'] = [px, py, pz, ax, ay, az]
            obs_dict['/state/right_arm/joint_angle'] = list(right_arm.joint.position)
            obs_dict['/state/right_arm/joint_effort'] = list(right_arm.joint.effort)
            force = [right_arm.wrench.force.x, right_arm.wrench.force.y, right_arm.wrench.force.z]
            torque = [right_arm.wrench.torque.x, right_arm.wrench.torque.y, right_arm.wrench.torque.z]
            obs_dict['/state/right_arm/tcp_forces'] = force+torque
            # self.logger.info(f"right_arm_forces:{obs_dict['/state/right_arm/tcp_forces']}")
            
        # 左手状态
        if robot_state.HasField('left_hand'):
            left_hand = robot_state.left_hand
            obs_dict['/state/left_hand/joint_angle'] = list(left_hand.joint.position)
            obs_dict['/state/left_hand/effort'] = list(left_hand.joint.effort)
            
        # 右手状态
        if robot_state.HasField('right_hand'):
            right_hand = robot_state.right_hand
            obs_dict['/state/right_hand/joint_angle'] = list(right_hand.joint.position)
            obs_dict['/state/right_hand/effort'] = list(right_hand.joint.effort)
        # 脖子状态
        if robot_state.HasField('neck'):
            neck = robot_state.neck
            obs_dict['/state/neck/joint_angle'] = list(neck.joint.position)
        
        # 电机状态
        if robot_state.HasField('motor'):
            motor_status = robot_state.motor
            # 提取电机位置信息
            motor_positions = []
            for motor in motor_status.motors:
                motor_positions.append(motor.position)
            obs_dict['/state/motor/joint_angle'] = motor_positions
            if self.static_lowbody and self.lowbody_motor_cache is None:
                self.logger.info(f"Static lowbody motor cache is None, "\
                                 f"set it to the current motor positions {motor_positions}")
                self.lowbody_motor_cache = motor_positions
            
            # 提取电机速度信息
            motor_velocities = []
            for motor in motor_status.motors:
                motor_velocities.append(motor.velocity)
            obs_dict['/state/motor/joint_velocity'] = motor_velocities
            
            # 提取电机力矩信息
            motor_torques = []
            for motor in motor_status.motors:
                motor_torques.append(motor.torque)
            obs_dict['/state/motor/joint_effort'] = motor_torques
        return obs_dict

    def _convert_north_observation_to_dict(self, north_obs):
        """将NorthObservation protobuf消息转换为字典格式"""
        obs_dict = {
            "timestamp": north_obs.timestamp,
            "reward": north_obs.reward,
            "on_sleep": north_obs.on_sleep,
        }
        
        # 处理机器人状态数据
        if self.enable_obs_robot_state and north_obs.HasField('robot_state'):
            robot_state = north_obs.robot_state
            
            robot_state_dict = self._parse_robot_state(robot_state)
            obs_dict.update(robot_state_dict)
                
             # 触觉数据
            if self.enable_tactile:
                for tactile in robot_state.tactile:
                    # 解析触觉传感器键值
                    sensor_key = self._parse_tactile_key(tactile.header.key)
                    
                    # 根据传感器键值生成h5路径
                    h5_base_path = self._get_tactile_h5_path(sensor_key)
                    
                    # 添加force6d数据到obs_dict
                    obs_dict[f"{h5_base_path}/force6d"] = [tactile.force6d.force.x, tactile.force6d.force.y, tactile.force6d.force.z, tactile.force6d.torque.x, tactile.force6d.torque.y, tactile.force6d.torque.z]
                    
                    # 添加raw_image数据（如果存在）
                    if tactile.HasField('raw_image'):
                        raw_image_data = tactile.raw_image.data
                        if raw_image_data:
                            # 解码raw_image数据
                            encoded_image = np.frombuffer(raw_image_data, np.uint8)
                            try:
                                raw_image = self.turbo_jpeg.decode(encoded_image)
                                # 添加到obs_dict
                                obs_dict[f"{h5_base_path}/raw"] = raw_image
                                self.logger.debug(f"Successfully decoded raw_image for tactile {sensor_key}")
                            except Exception as e:
                                self.logger.warning(f"Failed to decode raw_image for tactile {sensor_key}: {e}")
                    
                    # 添加deform_image数据（如果存在）
                    if tactile.HasField('deform_image'):
                        deform_image_data = tactile.deform_image.data
                        if deform_image_data:
                            # 解码deform_image数据
                            encoded_image = np.frombuffer(deform_image_data, np.uint8)
                            try:
                                deform_image = self.turbo_jpeg.decode(encoded_image)
                                # 添加到obs_dict
                                obs_dict[f"{h5_base_path}/deform"] = deform_image
                                self.logger.debug(f"Successfully decoded deform_image for tactile {sensor_key}")
                            except Exception as e:
                                self.logger.warning(f"Failed to decode deform_image for tactile {sensor_key}: {e}")
        
        # 处理视觉数据
        if self.enable_obs_vision and north_obs.HasField('vision'):
            vision = north_obs.vision
            
            if vision.HasField('image_left'):
                
                encoded_image = np.frombuffer(vision.image_left.data, np.uint8)
                image = cv2.cvtColor(self.turbo_jpeg.decode(encoded_image), cv2.COLOR_BGR2RGB)
                obs_dict["/observe/vision/head/stereo/lefteye/rgb"] = image
            if vision.HasField('image_right'):
                encoded_image = np.frombuffer(vision.image_right.data, np.uint8)
                image = cv2.cvtColor(self.turbo_jpeg.decode(encoded_image), cv2.COLOR_BGR2RGB)
                obs_dict["/observe/vision/head/stereo/righteye/rgb"] = image
            if vision.HasField('fish_left'):
                encoded_image = np.frombuffer(vision.fish_left.data, np.uint8)
                image = cv2.cvtColor(self.turbo_jpeg.decode(encoded_image), cv2.COLOR_BGR2RGB)
                obs_dict["/observe/vision/left_wrist/fisheye/rgb"] = image
            if vision.HasField('fish_right'):
                encoded_image = np.frombuffer(vision.fish_right.data, np.uint8)
                image = cv2.cvtColor(self.turbo_jpeg.decode(encoded_image), cv2.COLOR_BGR2RGB)
                obs_dict["/observe/vision/right_wrist/fisheye/rgb"] = image
        
        # 处理模式信息
        if self.enable_obs_mode and north_obs.HasField('mode'):
            mode = north_obs.mode
            obs_dict["mode"] = {
                "operation_mode": mode.operation_mode,
                "state": mode.state,
                "sub_state": mode.sub_state
            }
            obs_dict["/mode/act"] = mode.state
            obs_dict["/mode/sub_act"] = mode.sub_state
        
        return obs_dict

    def send_action(self, action_dict: Dict[str, Any], immediate: bool=False) -> None:
        """添加动作到缓冲区"""
        with self._action_lock:
            if len(self._action_buffer) == self.max_action_buffer_length:
                self.logger.warning("Action buffer is full, overwriting oldest action")

            if immediate:
                # 立即发送动作
                self._action_buffer.clear()
            self._action_buffer.append(action_dict)

    def get_latest_observation(self) -> Optional[Dict[str, Any]]:
        """获取最新的观察数据"""
        with self._obs_recv_lock:
            if len(self._normal_obs_buffer) == 0:
                self.logger.warning("No observations in buffer")
                return None
            return copy.deepcopy(self._normal_obs_buffer[-1][1])

    def _start_action_thread(self):
        """启动常驻动作发送线程"""
        self._action_thread_stop_event.clear()
        self._action_thread = threading.Thread(
            target=self._action_sender_loop,
            daemon=True
        )
        self._action_thread.start()

    def _action_sender_loop(self):
        last_send_time = time.time() - self.action_pub_duration
        
        while not self._action_thread_stop_event.is_set():
            # 计算距离上次发送的时间
            current_time = time.time()
            time_since_last = current_time - last_send_time
            
            # 如果还没到发送间隔，sleep到间隔时间
            if time_since_last < self.action_pub_duration:
                sleep_time = last_send_time + self.action_pub_duration - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"action pub duration is small than publish cost, sleep_time: {sleep_time}")
                continue
            
            # 获取并发送下一个动作
            with self._action_lock:
                if not self._action_buffer:  # 检查列表是否为空
                    self.logger.debug("Zmq action buffer is empty, waiting for actions...")
                    last_send_time = time.time()
                    continue
                
                raw_action = self._action_buffer.popleft()  # 取出第一个动作
                action_dict = self._prepare_action_dict(raw_action)

            if self.action_log_interval_s is None:
                print("Send Action")
            elif (
                self.action_log_interval_s > 0.0
                and current_time - self._last_action_log_time
                >= self.action_log_interval_s
            ):
                print("Send Action")
                self._last_action_log_time = current_time
            
            self._send_action(action_dict)
            # print('==========================', action_dict)
            # print('=========================*', raw_action)
            # 记录action历史（使用原始数据）
            last_send_time = time.time()

    def _parse_xyz_rotation_6d(self, xyz_rotation_6d):
        """解析xyz_rotation_6d为xyz和rotation_6d"""
        xyz = xyz_rotation_6d[:3]
        rotation_6d = xyz_rotation_6d[3:]
        rotation_matrix = rotation_6d_to_matrix_np(np.array(rotation_6d))
        qw, qx, qy, qz = t3d.quaternions.mat2quat(rotation_matrix)
        return {
            "position": {
              "x": xyz[0],
              "y": xyz[1],
              "z": xyz[2]
            },
            "orientation": {
                "x": qx,
                "y": qy,
                "z": qz,
                "w": qw
            }
        }

    def _prepare_action_dict(self, action_values):
        """将原始数据格式转换为发送用的动作字典"""
        action_dict = {}
        if self.using_eef_control:
            raise NotImplementedError("EEF control action preparation not implemented yet.")
            action_dict["right_hand/joints"] = dict(position=action_values["/action/right_hand/joint_angle"])
            action_dict["left_hand/joints"] = dict(position=action_values["/action/left_hand/joint_angle"])
            action_dict["right_hand/pose"] = self._parse_xyz_rotation_6d(action_values["/action/right_hand_base_link"])
            action_dict["left_hand/pose"] = self._parse_xyz_rotation_6d(action_values["/action/left_hand_base_link"])
            action_dict["head_pose"] = self._parse_xyz_rotation_6d(action_values["/action/head_base_link"])
            action_dict["torso_pose"] = self._parse_xyz_rotation_6d(action_values["/action/torso_base_link"])
        else:
            for key in self.action_pub_key:
                parts = key.split('/')
                if len(parts) < 4:
                    continue
                actuator = parts[2]
                action_type = parts[3]
                
                if key in action_values:
                    if actuator not in action_dict:
                        action_dict[actuator] = {}
                    action_dict[actuator]["position"] = action_values[key]
        
        return action_dict

    def _send_go_home_signal(self):
        """发送归位信号（使用ZenohWrapper服务调用）"""
        try:
            # 使用ZenohWrapper的query来调用服务
            result = self.zenoh_wrapper.query(
                self.go_home_service,
                data=None,  # 不需要发送数据
                timeout=5.0,
                encoding=MessageEncoding.JSON.value
            )
            
            if result is not None:
                self.logger.info("go home 触发成功!")
            else:
                self.logger.warning("go home 触发超时或无响应")
        except Exception as e:
            self.logger.error(f"go home 触发失败: {e}")

    def shutdown(self):
        """停止线程"""
        self._action_thread_stop_event.set()
        if self._action_thread is not None:
            self._action_thread.join()

    def _send_action(self, action_dict):
        """发送动作数据（使用ZenohWrapper）"""
        if not self.is_connected or not self.zenoh_wrapper:
            self.logger.warning("Zenoh not connected, cannot send action")
            return False
            
        try:
            # 创建UhrActionBundle消息
            if self.using_eef_control:
                current_time = time.time()
                action_bundle = pb.NorthTeleopInfo()
                if "left_hand/joints" in action_dict:
                    left_glove_joints = action_bundle.left_hand_joints
                    self._fill_joint_data(left_glove_joints, action_dict["left_hand/joints"])
                    
                # 处理右手套动作
                if "right_hand/joints" in action_dict:
                    right_glove_joints = action_bundle.right_hand_joints
                    self._fill_joint_data(right_glove_joints, action_dict["right_hand/joints"])
                    
                left_hand_pose = action_bundle.T_RobotRoot_LeftRobotHand
                self._fill_pose_data(left_hand_pose, action_dict["left_hand/pose"])
                    
                right_hand_pose = action_bundle.T_RobotRoot_RightRobotHand
                self._fill_pose_data(right_hand_pose, action_dict["right_hand/pose"])

                head_pose = action_bundle.T_RobotRoot_RobotHead
                self._fill_pose_data(head_pose, action_dict["head_pose"])

                torso_pose = action_bundle.T_RobotRoot_RobotUpperTrunk
                self._fill_pose_data(torso_pose, action_dict["torso_pose"])

                self._set_header_timestamp(action_bundle.header, current_time)
                action_bundle.is_direct = False

            else:
                action_bundle = pb.UhrActionBundle()
                
                # 设置时间戳
                current_time = time.time()
                
                # 处理左手套动作
                if "left_hand" in action_dict:
                    left_glove = action_bundle.left_glove
                    self._fill_joint_data(left_glove.joint, action_dict["left_hand"])
                    self._set_header_timestamp(left_glove.header, current_time)
                    
                # 处理右手套动作
                if "right_hand" in action_dict:
                    right_glove = action_bundle.right_glove
                    self._fill_joint_data(right_glove.joint, action_dict["right_hand"])
                    self._set_header_timestamp(right_glove.header, current_time)
                    
                # 处理左臂动作
                if "left_arm" in action_dict:
                    left_arm = action_bundle.left_arm
                    self._fill_joint_data(left_arm.joint, action_dict["left_arm"])
                    self._set_header_timestamp(left_arm.header, current_time)
                    
                # 处理右臂动作
                if "right_arm" in action_dict:
                    right_arm = action_bundle.right_arm
                    self._fill_joint_data(right_arm.joint, action_dict["right_arm"])
                    self._set_header_timestamp(right_arm.header, current_time)
                    
                # 处理颈部动作
                if "neck" in action_dict:
                    neck = action_bundle.neck
                    self._fill_joint_data(neck.joint, action_dict["neck"])
                    self._set_header_timestamp(neck.header, current_time)
                
                # 处理电机动作
                if "motor" in action_dict:
                    motor = action_bundle.motor
                    motor.timestamp = int(current_time * 1000)
                    if self.static_lowbody:
                        assert self.lowbody_motor_cache is not None
                        action_dict["motor"]["position"][0] = self.lowbody_motor_cache[0]
                        action_dict["motor"]["position"][1] = self.lowbody_motor_cache[1]
                    self._fill_motor_commands(motor, action_dict["motor"])
            
            # 使用ZenohWrapper发送Protobuf消息
            success = self.zenoh_wrapper.publish(
                self.action_topic,
                action_bundle,
                encoding=MessageEncoding.PROTOBUF.value
            )

            # print("-=-=-=-=-=-=-=-=", action_bundle.right_glove.joint)
            
            if success:
                self.logger.debug(f"Sent action bundle to {self.action_topic}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send action: {e}")
            return False

    def _fill_joint_data(self, joint_proto, joint_dict):
        """填充关节数据到protobuf消息"""
        if "position" in joint_dict:
            joint_proto.position[:] = joint_dict["position"]
        if "velocity" in joint_dict:
            joint_proto.velocity[:] = joint_dict["velocity"]
        if "effort" in joint_dict:
            joint_proto.effort[:] = joint_dict["effort"]
        if "name" in joint_dict:
            joint_proto.name[:] = joint_dict["name"]

    def _fill_pose_data(self, pose_proto, pose_dict):
        """填充关节数据到protobuf消息"""
        pose_proto.position.x = pose_dict["position"]["x"]
        pose_proto.position.y = pose_dict["position"]["y"]
        pose_proto.position.z = pose_dict["position"]["z"]
        pose_proto.orientation.x = pose_dict["orientation"]["x"]
        pose_proto.orientation.y = pose_dict["orientation"]["y"]
        pose_proto.orientation.z = pose_dict["orientation"]["z"]
        pose_proto.orientation.w = pose_dict["orientation"]["w"]

    def _fill_motor_commands(self, motor_proto, motor_dict):
        """填充电机命令到protobuf消息"""
        if "position" in motor_dict:
            positions = motor_dict["position"]
            for i, pos in enumerate(positions):
                motor_cmd = motor_proto.commands.add()
                motor_cmd.motor_id = i + 1  # motor ID从1开始
                motor_cmd.command = "position"
                motor_cmd.value = pos

    def _set_header_timestamp(self, header, timestamp):
        """设置header时间戳"""
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec

    def __del__(self):
        """析构函数，清理资源"""
        self.cleanup()

    def cleanup(self):
        """清理Zenoh资源"""
        try:
            # 停止动作发送线程
            self._action_thread_stop_event.set()
            if hasattr(self, '_action_thread') and self._action_thread is not None:
                self._action_thread.join(timeout=2.0)
            
            # 断开ZenohWrapper连接（会自动清理所有订阅和发布）
            if self.zenoh_wrapper:
                self.zenoh_wrapper.disconnect()
                self.zenoh_wrapper = None
            
            self.is_connected = False
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {e}") 
    
    def _parse_tactile_key(self, key):
        """解析触觉数据的键值，将数字键转换为可读的传感器名称"""
        # 如果键值已经在映射中，直接返回
        if key in self.tactile_sensor_map.values():
            return key
        
        # 尝试从数字键获取传感器名称
        if key in self.tactile_sensor_map:
            return self.tactile_sensor_map[key]
        
        # 如果无法解析，返回原始键值
        return key
    
    def _get_tactile_h5_path(self, sensor_key):
        """
        根据传感器键值生成h5路径
        
        Args:
            sensor_key: 传感器键值，如 'R_THUMB', 'L_INDEX' 等
        
        Returns:
            h5基础路径，如 '/observe/tactile/right_thumb'
        """
        # 传感器键值到h5路径的映射
        sensor_to_h5_map = {
            "R_LITTLE": "/observe/tactile/right_little",
            "R_RING": "/observe/tactile/right_ring",
            "R_MIDDLE": "/observe/tactile/right_middle",
            "R_INDEX": "/observe/tactile/right_index",
            "R_THUMB": "/observe/tactile/right_thumb",
            "L_LITTLE": "/observe/tactile/left_little",
            "L_RING": "/observe/tactile/left_ring",
            "L_MIDDLE": "/observe/tactile/left_middle",
            "L_INDEX": "/observe/tactile/left_index",
            "L_THUMB": "/observe/tactile/left_thumb"
        }
        
        return sensor_to_h5_map.get(sensor_key, f"/observe/tactile/{sensor_key.lower()}")

if __name__ == "__main__":
    
    env = NorthEnv(
        enable_tactile=True,
        action_output = [
            "/action/right_arm/joint_angle",
            "/action/right_hand/joint_angle",
            "/action/left_arm/joint_angle",
            "/action/left_hand/joint_angle",
            "/action/motor/joint_angle",
        ],
    )
    while True:
        latest_obs = env.get_latest_observation()
        if latest_obs is None:
            time.sleep(0.1)
            print("waiting for obs...")
            continue
        
        print(latest_obs.keys())
        # dict_keys(['timestamp', 'reward', 'on_sleep',
        # '/state/left_arm/joint_angle',
        # '/state/left_arm/joint_effort',
        # '/state/left_arm/tcp_forces',
        # '/state/right_arm/joint_angle',
        # '/state/right_arm/joint_effort',
        # '/state/right_arm/tcp_forces',
        # '/state/left_hand/joint_angle',
        # '/state/left_hand/effort',
        # '/state/right_hand/joint_angle',
        # '/state/right_hand/effort',
        # '/state/motor/joint_angle',
        # '/state/motor/joint_velocity',
        # '/state/motor/joint_effort',
        # '/observe/tactile/right_little/force6d',
        # '/observe/tactile/right_little/raw',
        # '/observe/tactile/right_little/deform',
        # '/observe/tactile/right_ring/force6d',
        # '/observe/tactile/right_ring/raw',
        # '/observe/tactile/right_ring/deform',
        # '/observe/tactile/right_middle/force6d',
        # '/observe/tactile/right_middle/raw',
        # '/observe/tactile/right_middle/deform',
        # '/observe/tactile/right_index/force6d',
        # '/observe/tactile/right_index/raw',
        # '/observe/tactile/right_index/deform',
        # '/observe/tactile/right_thumb/force6d',
        # '/observe/tactile/right_thumb/raw',
        # '/observe/tactile/right_thumb/deform',
        # '/observe/tactile/left_little/force6d',
        # '/observe/tactile/left_little/raw',
        # '/observe/tactile/left_little/deform',
        # '/observe/tactile/left_ring/force6d',
        # '/observe/tactile/left_ring/raw',
        # '/observe/tactile/left_ring/deform',
        # '/observe/tactile/left_middle/force6d',
        # '/observe/tactile/left_middle/raw',
        # '/observe/tactile/left_middle/deform',
        # '/observe/tactile/left_thumb/force6d',
        # '/observe/tactile/left_thumb/raw',
        # '/observe/tactile/left_thumb/deform',
        # '/observe/tactile/left_index/force6d',
        # '/observe/tactile/left_index/raw',
        # '/observe/tactile/left_index/deform',
        # '/observe/vision/head/stereo/lefteye/rgb',
        # '/observe/vision/head/stereo/righteye/rgb',
        # '/observe/vision/left_wrist/fisheye/rgb',
        # '/observe/vision/right_wrist/fisheye/rgb',
        # 'mode', '/mode/act', '/mode/sub_act'])

        lefteye_rgb = latest_obs['/observe/vision/head/stereo/lefteye/rgb']  # TODO: bgr to rgb
        print(lefteye_rgb.dtype, lefteye_rgb.shape, lefteye_rgb.max(), lefteye_rgb.min())
        cv2.imshow('lefteye_rgb', lefteye_rgb)


        tactile_raw = latest_obs['/observe/tactile/right_thumb/raw']
        print(tactile_raw.dtype, tactile_raw.shape, tactile_raw.max(), tactile_raw.min())
        cv2.imshow('tactile_raw', tactile_raw)

        cv2.waitKey(1)

        actions={}
        for key,value in latest_obs.items():
            if 'state' in key:
                print(key)
                actions[key.replace("/state/", "/action/")]=value
        current_value = actions['/action/right_hand/joint_angle'][2]
        actions['/action/right_hand/joint_angle'][2]+=0.002
        target_value = actions['/action/right_hand/joint_angle'][2]
        print("current", current_value, "target_value", target_value)

        # print(actions)
        env.send_action(actions, immediate=True)

        # env._action_buffer = [
        #     {
        #         '/action/left_arm/joint_angle': 
        #         '/action/left_arm/joint_effort',
        #         '/action/left_arm/tcp_forces',
        #         '/action/right_arm/joint_angle',
        #         '/action/right_arm/joint_effort',
        #         '/action/right_arm/tcp_forces',
        #         '/action/left_hand/joint_angle',
        #         '/action/left_hand/effort',
        #         '/action/right_hand/joint_angle',
        #         '/action/right_hand/effort',
        #         '/action/motor/joint_angle',
        #         '/action/motor/joint_velocity',
        #         '/action/motor/joint_effort',
        #     }
        # ]

        # action_dict = {
        #     ""
        # }
