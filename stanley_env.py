import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pandas as pd



# =========================================================
# ROS2 INTERFACE NODE
# =========================================================

class StanleyLearningNode(Node):
    def __init__(self):
        super().__init__('rl_coach_node')

        self.dk_pub = self.create_publisher(Float32, '/rl/dk', 10)
        self.dv_pub = self.create_publisher(Float32, '/rl/dv', 10)

        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10
        )

        self.create_subscription(Float32, '/rl/cte', self.cte_cb, 10)
        self.create_subscription(Float32, '/rl/heading', self.heading_cb, 10)
        self.create_subscription(Float32, '/rl/curvature', self.curv_cb, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)

        self.cte = 0.0
        self.heading = 0.0
        self.curvature = 0.0
        self.speed = 0.0

    def cte_cb(self, msg):
        self.cte = msg.data

    def heading_cb(self, msg):
        self.heading = msg.data

    def curv_cb(self, msg):
        self.curvature = msg.data

    def odom_cb(self, msg):
        self.speed = msg.twist.twist.linear.x

    def send_action(self, dk, dv):
        self.dk_pub.publish(Float32(data=float(dk)))
        self.dv_pub.publish(Float32(data=float(dv)))

    def reset_car(self, x, y, z, w):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = float(z)
        msg.pose.pose.orientation.w = float(w)
        self.reset_pub.publish(msg)


# =========================================================
# GYM ENVIRONMENT
# =========================================================

class StanleyEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ros_node, env_cfg, obs_cfg, reward_cfg, command_cfg):
        super().__init__()

        # Load raceline for randomized resets
        csv_path = "/root/sim_ws/src/f1tenth_gym_ros/racelines/Spielberg_map.csv"
        df = pd.read_csv(csv_path)
        self.raceline = df[['x', 'y', 'yaw']].values

        self.ros = ros_node
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.steps = 0

    # -----------------------------------------------------

    def step(self, action):
        dk, dv = action
        self.ros.send_action(dk, dv)

        time.sleep(self.env_cfg["step_delay"])
        rclpy.spin_once(self.ros, timeout_sec=0.01)

        cte = abs(self.ros.cte)
        heading = abs(self.ros.heading)
        curvature = self.ros.curvature
        speed = self.ros.speed

        cte_n = cte / self.env_cfg["max_track_width"]
        heading_n = heading / np.pi
        curv_n = min(curvature / self.env_cfg["curvature_norm"], 1.0)

        obs = np.array([cte_n, heading_n, curv_n], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)

        r = self.reward_cfg["scales"]

        reward = (
            r["speed"] * speed
            - r["cte"] * cte
            - r["heading"] * heading
            - r["curvature_speed"] * speed * curvature
        )
        
        self.steps += 1

        done = cte > self.env_cfg["max_track_width"]
        truncated = self.steps >= self.env_cfg["max_episode_steps"]

        if done:
            reward -= r["collision"]


        return obs, reward, done, truncated, {}

    # -----------------------------------------------------

    '''reset for the ftmhalle map'''
    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)

    #     x0, y0 = self.env_cfg["init_pos"]
    #     noise = self.env_cfg["reset_noise"]

    #     x = x0 + np.random.uniform(-noise, noise)
    #     y = y0 + np.random.uniform(-noise, noise)

    #     z, w = self.env_cfg["init_quat"][2:]
    #     self.ros.reset_car(x, y, z, w)

    #     time.sleep(0.5)
    #     self.steps = 0

    #     return np.zeros(3, dtype=np.float32), {}

    '''reset for the Spielberg map'''
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # sample a random point along the raceline
        idx = np.random.randint(0, len(self.raceline))
        x, y, yaw = self.raceline[idx]

        # convert yaw to quaternion (z, w)
        z = np.sin(yaw * 0.5)
        w = np.cos(yaw * 0.5)

        self.ros.reset_car(x, y, z, w)

        time.sleep(0.3)
        self.steps = 0

        return np.zeros(3, dtype=np.float32), {}

