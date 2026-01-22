import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
import pandas as pd
import numpy as np
import math
import os
from transforms3d.euler import quat2euler

# ================= CONFIG =================
CSV_PATH = '/root/sim_ws/src/f1tenth_gym_ros/racelines/Spielberg_map.csv'
WB = 0.33
K_SOFT = 1.0

# Nominal tuning
K_BASE = 1.2
K_CURV = 2.0
V_MAX = 5.0
V_MIN = 1.0
V_ALPHA = 3.0

# RL correction limits
K_CORR_SCALE = 0.2     # ±20%
V_CORR_SCALE = 1.0     # ±100%

# Rate limits
DK_MAX = 0.03
DV_MAX = 0.30

STEER_LIMIT = 0.4

CURV_LOOKAHEAD_DIST = 3.0  # meters
CURV_DEADZONE = 0.07  # tune this
# =========================================

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')

        if not os.path.exists(CSV_PATH):
            self.get_logger().error("Missing raceline")
            return

        df = pd.read_csv(CSV_PATH)
        self.waypoints = df[['x', 'y', 'yaw']].values

        # RL corrections (Δk, Δv) in [-1, 1]
        self.dk = 0.0
        self.dv = 0.0

        self.k_prev = K_BASE
        self.v_prev = V_MIN

        self.curv_baseline = np.mean([
            self.lookahead_curvature(i)
            for i in range(0, len(self.waypoints), 10)
        ])


        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.cte_pub = self.create_publisher(Float32, '/rl/cte', 10)
        self.heading_pub = self.create_publisher(Float32, '/rl/heading', 10)
        self.curv_pub = self.create_publisher(Float32, '/rl/curvature', 10)

        # Subscribers
        self.create_subscription(Odometry, '/ego_racecar/odom', self.drive_cb, 10)
        self.create_subscription(Float32, '/rl/dk', self.dk_cb, 10)
        self.create_subscription(Float32, '/rl/dv', self.dv_cb, 10)

    def dk_cb(self, msg):
        self.dk = float(np.clip(msg.data, -1.0, 1.0))

    def dv_cb(self, msg):
        self.dv = float(np.clip(msg.data, -1.0, 1.0))
    
    
    def lookahead_curvature(self, start_idx):
        n = len(self.waypoints)

        dist_accum = 0.0
        idx = start_idx

        # start heading
        yaw0 = self.waypoints[idx % n][2]

        # walk forward until lookahead distance
        while dist_accum < CURV_LOOKAHEAD_DIST:
            p_curr = self.waypoints[idx % n][:2]
            p_next = self.waypoints[(idx + 1) % n][:2]
            dist_accum += np.linalg.norm(p_next - p_curr)
            idx += 1

        yaw1 = self.waypoints[idx % n][2]

        dtheta = abs(wrap_angle(yaw1 - yaw0))

        if dist_accum < 1e-3:
            return 0.0

        return dtheta / dist_accum


    def drive_cb(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])

        fx = pos.x + WB * math.cos(yaw)
        fy = pos.y + WB * math.sin(yaw)

        dists = np.sum((self.waypoints[:, :2] - [fx, fy]) ** 2, axis=1)
        idx = int(np.argmin(dists))
        wp = self.waypoints[idx]

        heading_err = wp[2] - yaw
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        dx = wp[0] - fx
        dy = wp[1] - fy
        cte = -math.sin(yaw) * dx + math.cos(yaw) * dy

        curvature = self.lookahead_curvature(idx)

        # Publish RL signals
        self.cte_pub.publish(Float32(data=cte))
        self.heading_pub.publish(Float32(data=heading_err))
        self.curv_pub.publish(Float32(data=curvature))

        # ---------- Nominal schedule ----------
        k_nom = K_BASE + K_CURV * curvature
        effective_curv = max(curvature - self.curv_baseline, 0.0)
        v_nom = V_MAX * math.exp(-V_ALPHA * effective_curv)
        v_nom = np.clip(v_nom, V_MIN, V_MAX)

        # ---------- RL corrections ----------
        k_cmd = k_nom * (1.0 + K_CORR_SCALE * self.dk)
        v_cmd = v_nom * (1.0 + V_CORR_SCALE * self.dv)

        # ---------- Rate limiting ----------
        k = self.k_prev + np.clip(k_cmd - self.k_prev, -DK_MAX, DK_MAX)
        # Faster accel on straights, slower in corners
        dv_max = DV_MAX * (1.0 + 2.0 * max(0.0, CURV_DEADZONE - curvature))
        v = self.v_prev + np.clip(v_cmd - self.v_prev, -dv_max, dv_max)

        self.k_prev = k
        self.v_prev = v

        steer = heading_err + math.atan2(k * cte, v + K_SOFT)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer
        drive.drive.speed = float(v)
        self.drive_pub.publish(drive)


def main():
    rclpy.init()
    node = StanleyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
