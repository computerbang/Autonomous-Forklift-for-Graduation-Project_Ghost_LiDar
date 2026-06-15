"""
client_step5.py — [Step 5] TB3 ROS2 클라이언트

RGB 이미지 + PointCloud2 점군 토픽을 실시간으로 GPU 서버에 전송.

기존 Step4:
    /rgb → JPEG → GPU 서버

Step5:
    /rgb → JPEG
    /cloud PointCloud2 → raw data bytes
    두 데이터를 ZMQ multipart로 GPU 서버 전송

전송 구조:
[0] metadata JSON
[1] RGB JPEG bytes
[2] PointCloud2 raw data bytes

주의:
- 기존 client_step4.py는 건드리지 않음.
- 이 파일은 server_step5.py와 같이 사용해야 함.
- GPU 서버 포트는 5556 사용.
"""

import json
import time

import zmq
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist


# ── 서버 / ROS 설정 ──────────────────────────────────────────────
SERVER = "tcp://192.168.0.204:5556"

RGB_TOPIC = "/rgb"

# 실제 점군이 발행되는 토픽.
# 확인 결과 /cloud 가 Publisher count: 1 이었으므로 /cloud 사용.
POINT_TOPIC = "/cloud"

CMD_TOPIC = "/cmd_vel"

JPEG_QUALITY = 80

# 너무 자주 보내면 네트워크가 무거워짐.
# 0.20초 = 최대 약 5Hz
SEND_INTERVAL_SEC = 0.2

# 오래된 점군이면 보내지 않음.
MAX_CLOUD_AGE_SEC = 1.0

SERVER_TIMEOUT_MS = 1500


def msg_to_bgr(msg: Image):
    """sensor_msgs/Image → OpenCV BGR."""
    arr = np.frombuffer(msg.data, np.uint8)

    n = msg.height * msg.width
    ch = len(msg.data) // n if n else 3
    enc = msg.encoding.lower()

    if ch >= 3:
        arr = arr.reshape(msg.height, msg.width, ch)[:, :, :3]

        if enc.startswith("rgb"):
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return arr

    arr = arr.reshape(msg.height, msg.width)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def stamp_to_float(stamp):
    """ROS stamp → float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_stamp_to_float(msg: Image):
    try:
        return stamp_to_float(msg.header.stamp)
    except Exception:
        return time.time()


def cloud_stamp_to_float(msg: PointCloud2):
    try:
        return stamp_to_float(msg.header.stamp)
    except Exception:
        return time.time()


def point_fields_to_meta(fields):
    """
    PointCloud2 fields를 서버가 numpy dtype으로 복원할 수 있는 형태로 변환.

    sensor_msgs/msg/PointField datatype:
        1 INT8
        2 UINT8
        3 INT16
        4 UINT16
        5 INT32
        6 UINT32
        7 FLOAT32
        8 FLOAT64
    """
    result = []

    for f in fields:
        result.append({
            "name": f.name,
            "offset": int(f.offset),
            "datatype": int(f.datatype),
            "count": int(f.count),
        })

    return result


class DockingClientStep5(Node):
    def __init__(self):
        super().__init__("docking_client_step5")

        self.ctx = zmq.Context()
        self.sock = None
        self.connect_socket()

        self.pub = self.create_publisher(Twist, CMD_TOPIC, 10)

        self.latest_cloud = None
        self.latest_cloud_time = None
        self.latest_cloud_recv_time = None

        self.rgb_sub = self.create_subscription(
            Image,
            RGB_TOPIC,
            self.on_image,
            qos_profile_sensor_data
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            POINT_TOPIC,
            self.on_cloud,
            qos_profile_sensor_data
        )

        self.frame_id = 0
        self.last_send_time = 0.0

        self.get_logger().info(f"client_step5 시작 → {SERVER}")
        self.get_logger().info(f"RGB_TOPIC={RGB_TOPIC}")
        self.get_logger().info(f"POINT_TOPIC={POINT_TOPIC}")
        self.get_logger().info(f"CMD_TOPIC={CMD_TOPIC}")

    def connect_socket(self):
        if self.sock is not None:
            try:
                self.sock.close(0)
            except Exception:
                pass

        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, SERVER_TIMEOUT_MS)
        self.sock.setsockopt(zmq.SNDTIMEO, SERVER_TIMEOUT_MS)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(SERVER)

    def on_cloud(self, msg: PointCloud2):
        """최신 PointCloud2를 메모리에 보관."""
        self.latest_cloud = msg
        self.latest_cloud_time = cloud_stamp_to_float(msg)
        self.latest_cloud_recv_time = time.time()

    def on_image(self, msg: Image):
        now = time.time()

        # 전송 주기 제한
        if now - self.last_send_time < SEND_INTERVAL_SEC:
            return

        self.last_send_time = now
        self.frame_id += 1

        try:
            # ── RGB JPEG 변환 ────────────────────────────────────
            frame = msg_to_bgr(msg)

            ok, jpg_buf = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not ok:
                self.get_logger().warn("RGB JPEG encode 실패")
                return

            rgb_bytes = jpg_buf.tobytes()
            h, w = frame.shape[:2]

            # ── PointCloud2 raw data 전송 비활성화 ───────────────
            # 점군 전송만 제거함. VPN 대역폭 절감 목적.
            # multipart 3-파트 구조는 서버 호환을 위해 그대로 유지하되
            # 3번째 프레임을 빈 bytes로 보냄.
            cloud_bytes = b""

            cloud_meta = {
                "present": False,
                "topic": POINT_TOPIC,
                "frame_id": None,
                "stamp": None,
                "age_sec": None,
                "height": None,
                "width": None,
                "point_step": None,
                "row_step": None,
                "is_bigendian": None,
                "is_dense": None,
                "fields": [],
                "data_size_bytes": 0,
                "error": "point cloud transmission disabled",
            }

            # ── metadata 구성 ────────────────────────────────────
            meta = {
                "type": "rgb_pointcloud2_raw_step5",
                "frame_id": self.frame_id,
                "time_unix": now,
                "rgb": {
                    "topic": RGB_TOPIC,
                    "encoding": "jpg",
                    "quality": JPEG_QUALITY,
                    "width": int(w),
                    "height": int(h),
                    "size_bytes": len(rgb_bytes),
                    "stamp": image_stamp_to_float(msg),
                },
                "cloud": cloud_meta,
            }

            # ── GPU 서버로 multipart 전송 ────────────────────────
            self.sock.send_multipart([
                json.dumps(meta).encode("utf-8"),
                rgb_bytes,
                cloud_bytes,
            ])

            r = json.loads(self.sock.recv_string())

        except Exception as e:
            self.get_logger().warn(f"frame skip / socket reset: {e}")
            self.connect_socket()
            return

        # ── 서버 응답 기반 cmd_vel publish ───────────────────────
        # 현재 server_step5.py는 확인 단계라 lin/ang를 0으로 보내는 상태.
        t = Twist()
        t.linear.x = float(r.get("lin", 0.0))
        t.angular.z = float(r.get("ang", 0.0))
        self.pub.publish(t)

        state = r.get("state", "?")
        d = r.get("dist_m")

        if cloud_meta["present"]:
            field_names = [f["name"] for f in cloud_meta["fields"]]
            cloud_status = (
                f"cloud=YES "
                f"{cloud_meta['width']}x{cloud_meta['height']} "
                f"{cloud_meta['data_size_bytes'] / 1024:.1f}KB "
                f"age={cloud_meta['age_sec']}s "
                f"fields={field_names}"
            )
        else:
            cloud_status = f"cloud=NO ({cloud_meta.get('error')})"

        self.get_logger().info(
            f"[{state:16}] lin {t.linear.x:+.3f}  "
            f"ang {t.angular.z:+.3f}  dist {d}  |  {cloud_status}"
        )

    def stop(self):
        self.pub.publish(Twist())

    def destroy_node(self):
        try:
            if self.sock is not None:
                self.sock.close(0)
            self.ctx.term()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = DockingClientStep5()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
