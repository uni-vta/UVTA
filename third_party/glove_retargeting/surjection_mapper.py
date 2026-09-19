"""
满射映射器 (Surjection Mapper)

将raw关节角度的有限范围平滑映射到model关节的完整范围（0-90度）
只针对四指（食指、中指、无名指、小指）的弯曲关节
"""
import os
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SurjectionMapper:
    """满射映射器：将有限范围的传感器数据映射到完整的关节范围"""
    
    def __init__(self, config_path=None):
        """
        初始化满射映射器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config = self._load_config(config_path)
        self.enabled = self.config is not None
        
        if self.enabled:
            # 从配置中获取参数
            self.threshold_lower = self.config.get("threshold_lower", 0.15)
            self.threshold_upper = self.config.get("threshold_upper", 0.85)
            self.buffer_size = self.config.get("buffer_size", 0.1)
            
            # 预处理关节配置（按手指分组）
            self.finger_configs = self._parse_finger_configs()
            
            logger.info(f"SurjectionMapper initialized: "
                       f"lower={self.threshold_lower:.2f}, "
                       f"upper={self.threshold_upper:.2f}, "
                       f"buffer={self.buffer_size:.2f}")
        else:
            logger.warning("SurjectionMapper disabled (config not found)")
    
    def _load_config(self, config_path=None):
        """加载配置文件"""
        if config_path is None:
            # 默认配置文件路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config", "surjection_mapping_config.json")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config: {e}")
            return None
    
    def _parse_finger_configs(self):
        """解析并缓存手指配置"""
        finger_configs = {}
        
        for finger_name, finger_data in self.config.get("joints", {}).items():
            finger_configs[finger_name] = {}
            
            for joint_name, joint_config in finger_data.items():
                if not joint_config.get("enabled", True):
                    continue
                
                # 提取并验证参数
                model_idx = joint_config.get("model_idx")
                raw_idx = joint_config.get("raw_idx")
                raw_lower = joint_config.get("raw_lower")
                raw_upper = joint_config.get("raw_upper")
                model_lower = joint_config.get("model_lower")
                model_upper = joint_config.get("model_upper")
                
                if any(x is None for x in [model_idx, raw_idx, raw_lower, raw_upper, model_lower, model_upper]):
                    logger.warning(f"Incomplete config for {finger_name}.{joint_name}")
                    continue
                
                finger_configs[finger_name][joint_name] = {
                    'model_idx': model_idx,
                    'raw_idx': raw_idx,
                    'raw_lower_rad': np.deg2rad(raw_lower),
                    'raw_upper_rad': np.deg2rad(raw_upper),
                    'model_lower_rad': np.deg2rad(model_lower),
                    'model_upper_rad': np.deg2rad(model_upper)
                }
        
        return finger_configs
    
    def apply(self, optimized_q, raw_joint_angles):
        """
        应用满射映射
        
        Args:
            optimized_q: 优化后的关节角度 (n_dof,) numpy数组
            raw_joint_angles: 原始关节角度 (n_dof,) numpy数组
            
        Returns:
            modified_q: 修正后的关节角度 (n_dof,) numpy数组
        """
        # 如果未启用，直接返回原值
        if not self.enabled:
            return optimized_q
        
        # 复制一份，避免修改原数组
        modified_q = optimized_q.copy()
        
        # 计算缓冲区边界
        lower_buffer_end = self.threshold_lower + self.buffer_size
        upper_buffer_start = self.threshold_upper - self.buffer_size
        
        # 对每根手指进行映射
        for finger_name, joints in self.finger_configs.items():
            if not joints:
                continue
            
            # 计算该手指所有关节的归一化值
            normalized_values = []
            joint_infos = []
            
            for joint_name, config in joints.items():
                raw_angle = raw_joint_angles[config['raw_idx']]
                
                # 归一化到 [0, 1]
                range_diff = config['raw_upper_rad'] - config['raw_lower_rad']
                if abs(range_diff) < 1e-6:
                    continue
                
                normalized = (raw_angle - config['raw_lower_rad']) / range_diff
                normalized = np.clip(normalized, 0.0, 1.0)
                
                normalized_values.append(normalized)
                joint_infos.append({
                    'model_idx': config['model_idx'],
                    'model_lower_rad': config['model_lower_rad'],
                    'model_upper_rad': config['model_upper_rad']
                })
            
            if not normalized_values:
                continue
            
            # 使用平均归一化值判断映射方向（避免突变）
            avg_normalized = np.mean(normalized_values)
            
            # 下界映射
            if avg_normalized <= lower_buffer_end:
                # 计算权重
                if avg_normalized <= self.threshold_lower:
                    weight = 1.0
                else:
                    weight = 1.0 - (avg_normalized - self.threshold_lower) / self.buffer_size
                    weight = np.clip(weight, 0.0, 1.0)
                
                # 应用映射
                for info in joint_infos:
                    idx = info['model_idx']
                    target = info['model_lower_rad']
                    modified_q[idx] = (1.0 - weight) * modified_q[idx] + weight * target
            
            # 上界映射
            elif avg_normalized >= upper_buffer_start:
                # 计算权重
                if avg_normalized >= self.threshold_upper:
                    weight = 1.0
                else:
                    weight = (avg_normalized - upper_buffer_start) / self.buffer_size
                    weight = np.clip(weight, 0.0, 1.0)
                
                # 应用映射
                for info in joint_infos:
                    idx = info['model_idx']
                    target = info['model_upper_rad']
                    modified_q[idx] = (1.0 - weight) * modified_q[idx] + weight * target
        
        return modified_q


# 简单使用示例
if __name__ == "__main__":
    # 创建映射器
    mapper = SurjectionMapper()
    
    # 示例数据
    optimized_q = np.random.rand(22) * 1.5  # 假设的优化结果
    raw_joint_angles = np.random.rand(22) * 1.0  # 假设的原始角度
    
    # 应用映射
    result = mapper.apply(optimized_q, raw_joint_angles)
    
    print(f"映射器状态: {'已启用' if mapper.enabled else '未启用'}")
    if mapper.enabled:
        print(f"配置参数: lower={mapper.threshold_lower}, "
              f"upper={mapper.threshold_upper}, buffer={mapper.buffer_size}")
        print(f"手指数量: {len(mapper.finger_configs)}")

