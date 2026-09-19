#!/usr/bin/env python3
"""
Zenoh Wrapper Class
提供简洁的zenoh pub、sub和queryable接口，支持JSON和Protobuf消息类型
"""

import os
import socket
import time
import zenoh
import json
import logging
import os
import socket
import threading
import queue
from typing import Callable, Optional, Any, Dict, List, Type
from enum import Enum
import google.protobuf.message as pb_message
from google.protobuf import json_format

try:
    from zenoh import handlers as _zenoh_handlers
except ImportError:  # older zenoh-python without handlers module
    _zenoh_handlers = None

logger = logging.getLogger(__name__)


# =========================================================
# 连接配置（环境变量 + 机器人 endpoint 自动发现）
# =========================================================
_DEFAULT_ROBOT_IP = os.environ.get("SHARPA_NORTH_ROBOT_IP", "192.168.6.66")
_DEFAULT_ROBOT_ZENOH_PORT = os.environ.get("SHARPA_NORTH_ZENOH_PORT", "7449")


def _env_endpoints(name: str) -> List[str]:
    """解析逗号分隔的 endpoint 环境变量，如 'tcp/192.168.6.66:7449'。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        logger.warning(f"{name}={raw!r} 不是整数，使用默认值 {default}")
        return default


def _local_ip_for(remote_ip: str) -> Optional[str]:
    """根据到 remote_ip 的路由推断本机出口 IP（UDP connect 不会真正发包）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((remote_ip, 1))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def _auto_robot_endpoints() -> List[str]:
    """本机确实在机器人网段上时，自动生成机器人 Zenoh endpoint。"""
    robot_ip = _DEFAULT_ROBOT_IP
    local_ip = _local_ip_for(robot_ip)
    if local_ip is None:
        return []
    # 同网段判断：前三段一致（默认 /24，如 192.168.6.x）
    if local_ip.rsplit(".", 1)[0] == robot_ip.rsplit(".", 1)[0]:
        return [f"tcp/{robot_ip}:{_DEFAULT_ROBOT_ZENOH_PORT}"]
    return []


def _configured_connect_endpoints() -> List[str]:
    """优先环境变量显式指定，其次自动发现机器人 endpoint。"""
    endpoints = _env_endpoints("SHARPA_ZENOH_CONNECT_ENDPOINTS")
    if endpoints:
        return endpoints
    return _auto_robot_endpoints()


def _env_mode(has_connect_endpoints: bool) -> str:
    """有机器人 connect endpoint 时默认 client 直连，否则回退 peer。"""
    mode = os.environ.get("SHARPA_ZENOH_MODE", "").strip().lower()
    if mode in ("client", "peer"):
        return mode
    return "client" if has_connect_endpoints else "peer"


def _auto_listen_endpoint() -> Optional[str]:
    """本机到机器人那张网卡上的监听 endpoint，无法确定时返回 None。"""
    local_ip = _local_ip_for(_DEFAULT_ROBOT_IP)
    if not local_ip:
        return None
    return f"tcp/{local_ip}:0"

class MessageEncoding(Enum):
    """消息编码类型"""
    JSON = "application/json"
    PROTOBUF = "application/x-protobuf"
    TEXT = "text/plain"

class ZenohWrapper:
    """
    Zenoh包装类
    提供简洁的pub、sub和queryable接口，支持JSON和Protobuf消息类型
    """
    
    def __init__(self):
        """
        初始化Zenoh包装器
        
        完全自动化配置：
        - 本地节点间：自动使用共享内存（SHM）
        - 远程节点间：自动使用网络（TCP/UDP）
        - 混合场景：自动为每个连接选择最优方式
        - 无需任何配置！
        """
        self.session: Optional[zenoh.Session] = None
        self.publishers: Dict[str, zenoh.Publisher] = {}
        self.subscribers: Dict[str, zenoh.Subscriber] = {}
        self.queryables: Dict[str, zenoh.Queryable] = {}
        self.is_seed_node = False  # 标记是否为种子节点
        self.connection_info = {}  # 存储连接详细信息
        # 订阅 drain 线程管理（topic -> (thread, stop_event)）
        self._subscriber_threads: Dict[str, tuple] = {}

        
    def connect(self) -> bool:
        """
        连接到Zenoh网络（完全自动配置）
        
        自动特性：
        - 本地通信：共享内存（零拷贝，微秒级延迟）
        - 远程通信：网络传输（自动发现，自动连接）
        - 智能路由：Zenoh自动为每个订阅者选择最优路径
        
        Returns:
            bool: 连接是否成功
        """
        # ========== 优先路径：显式连接机器人 Zenoh endpoint ==========
        # 依赖 multicast/peer 自动发现在多网卡机器上不可靠：ping 通不等于
        # scouting 能发现发布端。只要能确定机器人 endpoint（环境变量显式
        # 指定，或本机在机器人网段上），就用 client mode 直连。
        connect_endpoints = _configured_connect_endpoints()
        mode = _env_mode(bool(connect_endpoints))
        if connect_endpoints and mode == "client":
            return self._connect_direct(connect_endpoints)

        try:
            # 创建zenoh配置
            zenoh_config = zenoh.Config()
            
            # 走到这里说明 client 直连不适用（没解析出 endpoint，或用
            # SHARPA_ZENOH_MODE=peer 显式要求 peer），所以这里只保留 peer 模式，
            # 机器人 endpoint 仅作为 peer 的后备连接目标使用。
            robot_connect_endpoints = connect_endpoints
            zenoh_config.insert_json5("mode", '"peer"')
            
            # 配置优化（避免触发Zenoh内部错误）
            try:
                # 设置合理的缓冲区大小，避免Position error
                zenoh_config.insert_json5("transport/unicast/qos/enabled", "true")
            except Exception as config_err:
                logger.debug(f"配置QoS失败（忽略）: {config_err}")
                pass
            
            # ========== 自动化配置：本地 + 网络 ==========
            
            # 显式连接机器人时默认关闭 multicast/SHM，减少多网卡环境的不确定性
            multicast_enabled = _env_bool(
                "SHARPA_ZENOH_MULTICAST_ENABLED", not robot_connect_endpoints)
            shared_memory_enabled = _env_bool(
                "SHARPA_ZENOH_SHARED_MEMORY_ENABLED", not robot_connect_endpoints)
            
            # ===== 本地/局域网自动发现（peer 模式）=====
            logger.info("🚀 自动配置Zenoh (稳定模式: SHM + 网络)")
            
            # 1. 共享内存
            zenoh_config.insert_json5(
                "transport/shared_memory/enabled",
                "true" if shared_memory_enabled else "false")
            
            # 2. 发现机制配置（禁用gossip避免panic bug）
            zenoh_config.insert_json5(
                "scouting/multicast/enabled",
                "true" if multicast_enabled else "false")   # 网络发现
            zenoh_config.insert_json5("scouting/gossip/enabled", "false")     # 禁用gossip（有bug）
            
            # 3. 智能监听：尝试成为种子节点，失败则连接现有节点
            try:
                listen_endpoints = _env_endpoints("SHARPA_ZENOH_LISTEN_ENDPOINTS") or [
                    "tcp/127.0.0.1:7447",
                    _auto_listen_endpoint() or "tcp/0.0.0.0:0",
                ]
                zenoh_config.insert_json5(
                    "listen/endpoints", json.dumps(listen_endpoints))
                self.session = zenoh.open(zenoh_config)
                self.is_seed_node = True
                self.connection_info = {
                    "role": "SEED",
                    "local_endpoint": ",".join(listen_endpoints),
                    "network_endpoint": "0.0.0.0:dynamic",
                    "shm_enabled": shared_memory_enabled,
                    "multicast_enabled": multicast_enabled,
                    "connects_to": None,
                }
                logger.info("="*70)
                logger.info("✅ 连接成功 - 种子节点")
                logger.info("角色: 🌟 种子节点 (帮助节点发现)")
                logger.info("本地: 127.0.0.1:7447 + SHM (共享内存，零拷贝)")
                logger.info("网络: 0.0.0.0:动态端口 (多播发现)")
                logger.info("传输: Zenoh自动为每个连接选择最优方式")
                logger.info("="*70)
                return True
                
            except Exception as e1:
                if "Address already in use" in str(e1) or "already in use" in str(e1).lower():
                    logger.info("种子节点已存在，加入网络...")
                    # 重新配置：动态端口 + 连接种子节点（优先连机器人 endpoint）
                    fallback_connect = robot_connect_endpoints or ["tcp/127.0.0.1:7447"]
                    fallback_listen = _env_endpoints("SHARPA_ZENOH_LISTEN_ENDPOINTS") or [
                        "tcp/127.0.0.1:0",
                        _auto_listen_endpoint() or "tcp/0.0.0.0:0",
                    ]
                    zenoh_config = zenoh.Config()
                    zenoh_config.insert_json5("mode", '"peer"')
                    zenoh_config.insert_json5(
                        "transport/shared_memory/enabled",
                        "true" if shared_memory_enabled else "false")
                    zenoh_config.insert_json5(
                        "scouting/multicast/enabled",
                        "true" if multicast_enabled else "false")
                    zenoh_config.insert_json5("scouting/gossip/enabled", "false")  # 禁用gossip
                    zenoh_config.insert_json5(
                        "listen/endpoints", json.dumps(fallback_listen))
                    zenoh_config.insert_json5(
                        "connect/endpoints", json.dumps(fallback_connect))
                    self.is_seed_node = False
                    self.connection_info = {
                        "role": "PEER",
                        "local_endpoint": ",".join(fallback_listen),
                        "network_endpoint": "0.0.0.0:dynamic",
                        "shm_enabled": shared_memory_enabled,
                        "multicast_enabled": multicast_enabled,
                        "connects_to": ",".join(fallback_connect),
                    }
                else:
                    raise
            
            # 创建会话（对于加入节点）
            if self.session is None:
                self.session = zenoh.open(zenoh_config)
            
            logger.info("="*70)
            logger.info("✅ 连接成功 - P2P节点")
            logger.info("角色: 🔗 P2P节点 (已连接到种子节点)")
            logger.info("本地: TCP(127.0.0.1:7447) + SHM (共享内存)")
            logger.info("网络: 多播发现远程节点")
            logger.info("传输: Zenoh自动选择SHM或NET")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"Zenoh连接失败: {e}")
            logger.error(f"详细错误信息: {type(e).__name__}: {str(e)}")
            return False
    
    def _connect_direct(self, connect_endpoints: List[str]) -> bool:
        """client mode 直连机器人 Zenoh endpoint（不依赖 multicast 发现）。

        默认关闭 multicast scouting 与 shared memory（显式直连时这两条
        路径不稳定且无必要），可用环境变量覆盖：
          SHARPA_ZENOH_MULTICAST_ENABLED / SHARPA_ZENOH_SHARED_MEMORY_ENABLED
        """
        multicast = _env_bool("SHARPA_ZENOH_MULTICAST_ENABLED", False)
        shared_memory = _env_bool("SHARPA_ZENOH_SHARED_MEMORY_ENABLED", False)
        listen_endpoints = _env_endpoints("SHARPA_ZENOH_LISTEN_ENDPOINTS")
        try:
            zenoh_config = zenoh.Config()
            zenoh_config.insert_json5("mode", '"client"')
            zenoh_config.insert_json5(
                "connect/endpoints", json.dumps(connect_endpoints))
            if listen_endpoints:
                zenoh_config.insert_json5(
                    "listen/endpoints", json.dumps(listen_endpoints))
            zenoh_config.insert_json5(
                "scouting/multicast/enabled", "true" if multicast else "false")
            zenoh_config.insert_json5("scouting/gossip/enabled", "false")
            zenoh_config.insert_json5(
                "transport/shared_memory/enabled",
                "true" if shared_memory else "false")

            self.session = zenoh.open(zenoh_config)
            self.is_seed_node = False
            self.connection_info = {
                "role": "CLIENT",
                "connects_to": connect_endpoints,
                "listen": listen_endpoints,
                "shm_enabled": shared_memory,
                "multicast_enabled": multicast,
            }
            # 这条日志是确认走机器人 endpoint 的关键证据
            logger.info(
                f"Zenoh session opened role=client mode=client "
                f"listen={listen_endpoints} connect={connect_endpoints} "
                f"multicast={multicast} shared_memory={shared_memory}"
            )
            return True
        except Exception as e:
            logger.error(f"Zenoh client 直连失败 {connect_endpoints}: {e}")
            self.session = None
            return False

    def _close_subscriber(self, topic: str):
        """停止 drain 线程并 undeclare subscriber，避免订阅线程残留。"""
        thread_entry = self._subscriber_threads.pop(topic, None)
        if thread_entry is not None:
            thread, stop_event = thread_entry
            stop_event.set()
            thread.join(timeout=2.0)
        subscriber = self.subscribers.pop(topic, None)
        if subscriber is not None:
            try:
                subscriber.undeclare()
            except Exception as e:
                logger.debug(f"清理subscriber时出错: {e}")

    def disconnect(self):
        """断开Zenoh连接并清理所有资源"""
        try:
            # 清理所有资源
            for queryable in list(self.queryables.values()):
                try:
                    queryable.undeclare()
                except Exception as e:
                    logger.debug(f"清理queryable时出错: {e}")
            
            for topic in list(self.subscribers.keys()):
                self._close_subscriber(topic)
            
            # Publishers不需要显式清理，会自动清理
            
            if self.session:
                try:
                    self.session.close()
                except Exception as e:
                    logger.debug(f"关闭session时出错: {e}")
                self.session = None
            
            self.publishers.clear()
            self.subscribers.clear()
            self.queryables.clear()
            
            logger.info("Zenoh连接已断开")
            
        except Exception as e:
            logger.error(f"断开连接时出错: {e}")
    
    def _serialize_message(self, data: Any, encoding: str):
        """
        序列化消息
        
        Args:
            data: 要序列化的数据
            encoding: 编码格式
            
        Returns:
            str or bytes: 序列化后的数据
        """
        try:
            if encoding == MessageEncoding.JSON.value:
                if isinstance(data, pb_message.Message):
                    # Protobuf消息转换为JSON
                    return json_format.MessageToJson(data)
                else:
                    return json.dumps(data, ensure_ascii=False)
            
            elif encoding == MessageEncoding.PROTOBUF.value:
                if isinstance(data, pb_message.Message):
                    # Protobuf消息序列化为二进制数据
                    return data.SerializeToString()  # 返回binary string
                else:
                    raise ValueError("Protobuf编码需要protobuf消息对象")
            
            else:  # TEXT或其他格式
                if isinstance(data, pb_message.Message):
                    # Protobuf消息转换为文本
                    return json_format.MessageToJson(data)
                else:
                    return str(data)
                    
        except Exception as e:
            logger.error(f"序列化消息失败: {e}")
            raise
    
    def _deserialize_message(self, payload: Any, encoding: str, message_type: Optional[Type[pb_message.Message]] = None) -> Any:
        """
        反序列化消息
        
        Args:
            payload: 要反序列化的数据
            encoding: 编码格式
            message_type: Protobuf消息类型（仅用于protobuf编码）
            
        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 检查payload是否为空
            if payload is None:
                return None
            
            # 处理不同类型的payload
            if hasattr(payload, '__class__') and payload.__class__.__name__ == 'ZBytes':
                # 将ZBytes转换为bytes
                try:
                    payload_bytes = bytes(payload)
                    logger.debug(f"ZBytes转换为bytes，长度: {len(payload_bytes)}")
                except Exception as e:
                    logger.error(f"ZBytes转换失败: {e}")
                    return None
            elif isinstance(payload, bytes):
                payload_bytes = payload
            elif isinstance(payload, str):
                payload_bytes = payload.encode('utf-8')
            else:
                payload_bytes = str(payload).encode('utf-8')
            
            if encoding == MessageEncoding.JSON.value:
                try:
                    # JSON需要先转换为字符串
                    payload_str = payload_bytes.decode('utf-8')
                    if not payload_str.strip():
                        return None
                    return json.loads(payload_str)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logger.error(f"JSON解析失败: {e}")
                    return None
            
            elif encoding == MessageEncoding.PROTOBUF.value:
                if message_type is None:
                    raise ValueError("Protobuf反序列化需要指定message_type")
                
                try:
                    # Protobuf直接使用二进制数据
                    message = message_type()
                    message.ParseFromString(payload_bytes)
                    return message
                except Exception as e:
                    logger.error(f"Protobuf解析失败: {e}")
                    return None
            
            else:  # TEXT或其他格式
                try:
                    return payload_bytes.decode('utf-8')
                except UnicodeDecodeError as e:
                    logger.error(f"文本解码失败: {e}")
                    return None
                    
        except Exception as e:
            logger.error(f"反序列化消息失败: {e}")
            return None  # 返回None而不是抛出异常
    
    def publish(self, topic: str, data: Any, encoding: str = MessageEncoding.JSON.value, 
                add_metadata: bool = False) -> bool:
        """
        发布消息到指定主题（Zenoh自动选择传输方式）
        
        Args:
            topic: 主题名称
            data: 要发布的数据（支持JSON对象、字符串或Protobuf消息）
            encoding: 编码格式
            add_metadata: 是否添加元数据（用于传输方式追踪）
            
        Returns:
            bool: 发布是否成功
            
        自动传输选择：
        - 本地订阅者: Zenoh自动使用共享内存（SHM）
        - 远程订阅者: Zenoh自动使用网络（NET）
        - 混合场景: Zenoh为每个订阅者选择最优方式
        """
        try:
            if not self.session:
                logger.error("Zenoh会话未连接")
                return False
            
            # 获取或创建发布者
            if topic not in self.publishers:
                self.publishers[topic] = self.session.declare_publisher(topic)
                logger.debug(f"创建发布者: {topic}")
            
            # 添加元数据（包含时间戳用于传输路径检测）
            if add_metadata and isinstance(data, dict):
                import socket
                import time
                data['_zenoh_meta'] = {
                    'pub_hostname': socket.gethostname(),
                    'pub_timestamp': time.perf_counter(),  # 高精度时间戳
                    'pub_role': self.connection_info.get('role', 'UNKNOWN')
                }
            
            # 序列化数据
            payload = self._serialize_message(data, encoding)
            
            # 发布消息 - Zenoh自动为每个订阅者选择最优传输方式
            self.publishers[topic].put(payload)
            logger.debug(f"📤 发布到 {topic} (Zenoh自动路由)")
            return True
            
        except Exception as e:
            logger.error(f"发布消息失败 {topic}: {e}")
            return False
    
    @staticmethod
    def _try_recv(receiver):
        """非阻塞取一个 sample，兼容不同 zenoh-python 版本；空/关闭返回 None。"""
        try:
            return receiver.try_recv()
        except Exception:
            return None

    def _drain_loop(self, topic: str, subscriber, callback, encoding,
                    message_type, stop_event: threading.Event):
        """drain 线程：把 RingChannel 里积压的样本全部取出，只反序列化并
        回调最新的一条。丢弃发生在反序列化之前，代价接近零，因此处理再慢
        也只会降低帧率（跳帧），不会造成延迟累积。"""
        # 不同版本的 subscriber handler 获取方式不同
        receiver = getattr(subscriber, "handler", subscriber)
        while not stop_event.is_set():
            sample = self._try_recv(receiver)
            if sample is None:
                time.sleep(0.001)
                continue
            # 队列里若已有更新的样本，全部取出只留最后一条
            while True:
                newer = self._try_recv(receiver)
                if newer is None:
                    break
                sample = newer

            try:
                topic_str = str(sample.key_expr)
            except Exception:
                topic_str = topic
            try:
                data = self._deserialize_message(
                    sample.payload, encoding, message_type)
            except Exception as deser_err:
                logger.error(f"反序列化失败 {topic_str}: {deser_err}")
                continue
            if data is None:
                continue
            try:
                callback(topic_str, data)
            except Exception as callback_err:
                logger.error(f"❌ 回调错误 {topic_str}: {callback_err}")

    def subscribe(self, topic: str, callback: Callable[[str, Any], None], 
                 encoding: str = MessageEncoding.JSON.value,
                 message_type: Optional[Type[pb_message.Message]] = None) -> bool:
        """
        订阅指定主题（保证回调处理的是最新样本）

        实现：subscriber 挂 RingChannel（有界环形队列，满了自动丢最旧的
        样本），独立 drain 线程从队列取样本、只把最新一条交给反序列化和
        上层 callback。旧实现把 callback 直接注册给 Zenoh，反序列化在
        Zenoh 回调线程里逐样本执行，处理速度跟不上发布速度时 FIFO 持续
        积压，上层永远在消费旧样本。

        队列容量由 SHARPA_ZENOH_SUBSCRIBER_RING_CAPACITY 控制（默认 8）。

        Args:
            topic: 主题名称
            callback: 回调函数，接收(topic, data)参数
            encoding: 编码格式
            message_type: Protobuf消息类型（仅用于protobuf编码）

        Returns:
            bool: 订阅是否成功
        """
        try:
            if not self.session:
                logger.error("Zenoh会话未连接")
                return False

            if _zenoh_handlers is None:
                logger.error(
                    "zenoh.handlers 不可用（zenoh-python 版本过旧），"
                    "无法使用 RingChannel 最新帧订阅；请升级 zenoh-python。"
                )
                return False

            capacity = _env_int("SHARPA_ZENOH_SUBSCRIBER_RING_CAPACITY", 8)
            subscriber = self.session.declare_subscriber(
                topic, _zenoh_handlers.RingChannel(capacity))
            self.subscribers[topic] = subscriber

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._drain_loop,
                args=(topic, subscriber, callback, encoding, message_type,
                      stop_event),
                name=f"zenoh-drain-{topic}",
                daemon=True,
            )
            self._subscriber_threads[topic] = (thread, stop_event)
            thread.start()

            logger.info(f"✓ 订阅成功: {topic} (RingChannel capacity={capacity})")
            return True

        except Exception as e:
            logger.error(f"❌ 订阅失败 {topic}: {e}")
            logger.exception("订阅异常详情:")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        取消订阅指定主题
        
        Args:
            topic: 主题名称
            
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            if topic in self.subscribers:
                self._close_subscriber(topic)
                logger.info(f"取消订阅: {topic}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"取消订阅失败 {topic}: {e}")
            return False
    
    def create_queryable(self, service_name: str, 
                        handler: Callable[[str, Any], Any],
                        encoding: str = MessageEncoding.JSON.value,
                        message_type: Optional[Type[pb_message.Message]] = None) -> bool:
        """
        创建查询服务
        
        Args:
            service_name: 服务名称
            handler: 处理函数，接收query_data，返回response_data
            encoding: 编码格式
            message_type: Protobuf消息类型（仅用于protobuf编码）
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if not self.session:
                logger.error("Zenoh会话未连接")
                return False
            
            def query_handler(sample):
                try:
                    # 反序列化查询数据
                    query_data = self._deserialize_message(sample.payload, encoding, message_type) if sample.payload else {}
                    logger.debug(f"查询服务收到数据: {sample.payload}, 反序列化后: {query_data}")
                    
                    # 调用处理函数
                    response_data = handler(sample.key_expr, query_data)
                    logger.debug(f"查询服务处理结果: {response_data}")
                    
                    # 只有当响应数据不为None时才发送响应
                    if response_data is not None:
                        # 序列化响应
                        response = self._serialize_message(response_data, encoding)
                        logger.debug(f"查询服务序列化响应: {response}")
                        
                        # 发送响应
                        sample.reply(sample.key_expr, response)
                        logger.debug(f"查询服务响应 {service_name}: {response}")
                        logger.debug(f"响应类型: {type(response)}, 长度: {len(response) if isinstance(response, str) else 'N/A'}")
                    else:
                        logger.warning(f"查询服务返回空响应: {service_name}")
                    
                except Exception as e:
                    logger.error(f"处理查询时出错 {service_name}: {e}")
            
            # 创建queryable
            queryable = self.session.declare_queryable(service_name, query_handler)
            self.queryables[service_name] = queryable
            
            logger.info(f"创建查询服务: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建查询服务失败 {service_name}: {e}")
            return False
    
    def query(self, service_name: str, data: Any = None, 
              timeout: float = 2.0, encoding: str = MessageEncoding.JSON.value,
              message_type: Optional[Type[pb_message.Message]] = None) -> Optional[Any]:
        """
        查询服务（支持超时）
        
        Args:
            service_name: 服务名称
            data: 查询数据（可选）
            timeout: 超时时间（秒），默认5秒
            encoding: 编码格式
            message_type: Protobuf消息类型（仅用于protobuf编码）
            
        Returns:
            Optional[Any]: 查询结果，失败或超时返回None
        """
        try:
            if not self.session:
                logger.error("Zenoh会话未连接")
                return None
            
            # 序列化查询数据
            query_payload = None
            if data is not None:
                query_payload = self._serialize_message(data, encoding)
            
            # 使用队列和线程实现超时机制
            result_queue = queue.Queue(maxsize=1)
            exception_holder = [None]
            
            def query_worker():
                """在独立线程中执行查询"""
                try:
                    # 发送查询
                    if query_payload is not None:
                        replies = self.session.get(service_name, payload=query_payload)
                    else:
                        replies = self.session.get(service_name)
                    
                    logger.debug(f"发送查询到 {service_name}")
                    
                    # 等待响应并处理
                    reply_count = 0
                    for reply in replies:
                        reply_count += 1
                        logger.debug(f"收到回复 #{reply_count}")
                        
                        # 检查回复是否有效
                        if reply and hasattr(reply, 'ok') and reply.ok:
                            if hasattr(reply.ok, 'payload') and reply.ok.payload:
                                response_data = self._deserialize_message(
                                    reply.ok.payload, encoding, message_type
                                )
                                if response_data is not None:
                                    logger.debug(f"查询成功 {service_name}")
                                    result_queue.put(response_data)
                                    return
                                else:
                                    logger.warning(f"查询响应反序列化失败: {service_name}")
                            else:
                                logger.warning(f"查询响应payload为空: {service_name}")
                        elif reply and hasattr(reply, 'err') and reply.err:
                            logger.error(f"查询错误: {service_name} - {reply.err}")
                            result_queue.put(None)
                            return
                    
                    logger.warning(f"查询无响应: {service_name}, 收到 {reply_count} 个回复")
                    result_queue.put(None)
                    
                except Exception as e:
                    logger.error(f"查询线程异常 {service_name}: {e}")
                    exception_holder[0] = e
                    result_queue.put(None)
            
            # 启动查询线程
            query_thread = threading.Thread(target=query_worker, daemon=True)
            query_thread.start()
            
            # 等待结果，带超时
            try:
                result = result_queue.get(timeout=timeout)
                if exception_holder[0] is not None:
                    logger.error(f"查询过程中发生异常: {service_name} - {exception_holder[0]}")
                return result
            except queue.Empty:
                logger.warning(f"查询超时: {service_name} (timeout={timeout}s)")
                return None
            except Exception as e:
                logger.error(f"等待查询结果时出错: {service_name} - {e}")
                return None
            
        except Exception as e:
            logger.error(f"查询失败 {service_name}: {e}")
            return None
    
    def get_active_topics(self) -> List[str]:
        """
        获取活跃的主题列表
        
        Returns:
            List[str]: 主题列表
        """
        return list(self.publishers.keys())
    
    def get_active_services(self) -> List[str]:
        """
        获取活跃的服务列表
        
        Returns:
            List[str]: 服务列表
        """
        return list(self.queryables.keys())
    
    def is_connected(self) -> bool:
        """
        检查是否已连接
        
        Returns:
            bool: 是否已连接
        """
        return self.session is not None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            "connected": self.is_connected(),
            "is_seed_node": self.is_seed_node,
            **self.connection_info
        }
    
    def get_active_peers(self) -> Dict[str, Any]:
        """
        获取当前活跃的peer连接信息
        
        Returns:
            Dict: 包含本地和远程peer数量的信息
        """
        if not self.session:
            return {"local_peers": 0, "remote_peers": 0, "total_peers": 0, "peers": []}
        
        try:
            # 获取 session 的 peer 信息
            # Zenoh 的 session.info() 可以获取路由器和peer信息
            peers_info = []
            local_count = 0
            remote_count = 0
            
            # 尝试获取 peers (这个API可能因版本而异)
            try:
                # 获取所有 peers 的 ZenohId
                info = self.session.info()
                
                # 尝试获取 peers 信息
                # 注意：具体API可能需要根据zenoh-python版本调整
                peers_zid = getattr(info, 'peers_zid', lambda: [])()
                
                for peer_id in peers_zid:
                    peer_str = str(peer_id)
                    peers_info.append(peer_str)
                    
                    # 简单启发式判断：本地peer通常通过127.0.0.1或共享内存
                    # 这是一个粗略的判断，实际可能需要更复杂的逻辑
                    
            except Exception as e:
                logger.debug(f"获取peer信息时出错: {e}")
            
            total = len(peers_info)
            
            return {
                "local_peers": local_count,
                "remote_peers": remote_count,
                "total_peers": total,
                "peers": peers_info
            }
        except Exception as e:
            logger.debug(f"获取活跃peers失败: {e}")
            return {"local_peers": 0, "remote_peers": 0, "total_peers": 0, "peers": []}
    
    def print_connection_status(self, show_peers: bool = True):
        """打印当前连接状态"""
        if not self.session:
            print("\n❌ 未连接\n")
            return
        
        print("\n" + "="*70)
        print("📡 Zenoh 连接状态")
        print("="*70)
        
        role = self.connection_info.get('role', 'UNKNOWN')
        if role == 'SEED':
            print("角色: 🌟 种子节点")
            print(f"本地端点: {self.connection_info.get('local_endpoint', 'N/A')}")
            print(f"网络端点: {self.connection_info.get('network_endpoint', 'N/A')}")
        else:
            print("角色: 🔗 P2P节点")
            print(f"本地端点: {self.connection_info.get('local_endpoint', 'N/A')}")
            print(f"网络端点: {self.connection_info.get('network_endpoint', 'N/A')}")
            if 'connects_to' in self.connection_info:
                print(f"连接到: {self.connection_info['connects_to']}")
        
        print(f"共享内存: {'✅ 已启用' if self.connection_info.get('shm_enabled') else '❌ 未启用'}")
        print(f"多播发现: {'✅ 已启用' if self.connection_info.get('multicast_enabled') else '❌ 未启用'}")
        
        # 显示peer信息
        if show_peers:
            print("-" * 70)
            print("🔗 连接的Peers:")
            peers = self.get_active_peers()
            if peers['total_peers'] > 0:
                for i, peer in enumerate(peers['peers'], 1):
                    print(f"  {i}. {peer}")
            else:
                print("  (暂无其他peer，或等待发现中...)")
        
        print("="*70 + "\n")
    
    def check_subscriber_types(self, topic: str) -> Dict[str, Any]:
        """
        检查指定主题的订阅者类型（实验性功能）
        
        注意：Zenoh可能不直接提供这个API，这个方法提供一个框架
        
        Args:
            topic: 主题名称
            
        Returns:
            Dict: 订阅者类型信息
        """
        if not self.session:
            return {"error": "未连接"}
        
        # 这是一个占位实现
        # Zenoh的具体版本可能提供或不提供这种细粒度的订阅者信息
        return {
            "topic": topic,
            "note": "Zenoh通常不暴露订阅者的详细位置信息（隐私设计）",
            "suggestion": "通过peer连接数量和配置类型来推断"
        }

# 便捷函数
def create_zenoh_client() -> ZenohWrapper:
    """
    创建并连接Zenoh客户端
    
    Returns:
        ZenohWrapper: 已连接的Zenoh包装器
        
    Raises:
        ConnectionError: 如果连接失败
    """
    wrapper = ZenohWrapper()
    if wrapper.connect():
        return wrapper
    else:
        raise ConnectionError("无法连接到Zenoh网络")

# 使用示例
def example_usage():
    """使用示例"""
    # 创建客户端
    zenoh_client = create_zenoh_client()
    
    try:
        # 发布JSON消息
        zenoh_client.publish("test/json", {"message": "Hello Zenoh!"})
        
        # 发布文本消息
        zenoh_client.publish("test/text", "Hello Text!", encoding=MessageEncoding.TEXT.value)
        
        # 订阅消息
        def message_callback(topic, data):
            logger.info(f"收到消息 {topic}: {data}")
        
        zenoh_client.subscribe("test/json", message_callback)
        
        # 创建查询服务
        def query_handler(topic, data):
            return {"response": f"处理了: {data}"}
        
        zenoh_client.create_queryable("test/service", query_handler, encoding=MessageEncoding.JSON.value)
        
        # 等待服务启动
        import time
        time.sleep(1)
        
        # 查询服务
        result = zenoh_client.query("test/service", {"request": "test"}, encoding=MessageEncoding.JSON.value)
        logger.info(f"查询结果: {result}")
        
        # 保持运行
        time.sleep(10)
        
    finally:
        zenoh_client.disconnect()

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行示例
    # example_usage() 

    zenoh_wrapper = ZenohWrapper()

    if not zenoh_wrapper.connect():
        raise Exception("Failed to connect to Zenoh network")
    
    import time
    
    while True:
        zenoh_wrapper.get_active_peers()
        zenoh_wrapper.print_connection_status()
        time.sleep(0.1)
