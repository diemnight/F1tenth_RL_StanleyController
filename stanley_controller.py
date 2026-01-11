import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseArray, Pose
from std_msgs.msg import Float32
import pandas as pd
import numpy as np
import math
import os
from transforms3d.euler import quat2euler, euler2quat

# --- CONFIGURATION ---
CSV_PATH = '/root/sim_ws/src/f1tenth_gym_ros/racelines/FTMHalle_ws25.csv'
K_SOFT = 1.0       # Softening constant - prevents division by zero - prevents the car from steering infinitely hard if the speed drops to zero.
WB = 0.33          # Wheelbase - distance from front axle to rear axle
DEFAULT_GAIN = 3   # Starting gain
DEFAULT_SPEED = 2.0
# ---------------------

class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')
        
        # 1. Load Waypoints
        if not os.path.exists(CSV_PATH):
            self.get_logger().error(f"CSV file not found: {CSV_PATH}")
            return
            
        df = pd.read_csv(CSV_PATH)
        self.waypoints = df[['x', 'y', 'yaw', 'speed']].values
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")

        # 2. Dynamic Control Variables (Tunable by RL)
        self.k_gain = DEFAULT_GAIN 
        self.speed = DEFAULT_SPEED

        # 3. Publishers & Subscribers
        # Driving
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10) # Publishes drive commands - steering and speed
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.drive_callback, 10)    #listens to the car's odometry - position and orientation, runs drive_callback when a new message arrives
        

        # RL Interface
        # Listen for new K_GAIN commands
        self.gain_sub = self.create_subscription(Float32, '/rl/k_gain', self.gain_callback, 10)    # listens for new K_GAIN commands from the RL agent
        # Report current error to the RL Agent
        self.error_pub = self.create_publisher(Float32, '/rl/error', 10)    # publishes the current cross-track error to the RL agent



    def gain_callback(self, msg):
        """Updates the control gain when the RL agent sends a command"""
        self.k_gain = msg.data
        #Print to verify connection
        self.get_logger().info(f"RL Set Gain: {self.k_gain:.2f}")

    def drive_callback(self, msg):
        # Extract Car Position - it is the position of the rear axle
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        
        # Extract Car Orientation (Yaw)
        q = msg.pose.pose.orientation
        _, _, car_yaw = quat2euler([q.w, q.x, q.y, q.z])

        # A. Find the Closest Waypoint (Target)
        # Front Axle Position
        front_x = pos_x + WB * math.cos(car_yaw)
        front_y = pos_y + WB * math.sin(car_yaw)

        # Calculate distance to all points
        dists = np.sum((self.waypoints[:, :2] - np.array([front_x, front_y]))**2, axis=1)
        target_idx = np.argmin(dists)
        target_pt = self.waypoints[target_idx]
        

        # B. Calculate Errors
        # 1. Heading Error
        track_yaw = target_pt[2]
        heading_error = track_yaw - car_yaw
        while heading_error > math.pi: heading_error -= 2*math.pi
        while heading_error < -math.pi: heading_error += 2*math.pi

        # 2. Cross Track Error (CTE)
        dx = target_pt[0] - front_x
        dy = target_pt[1] - front_y
        
        # Project vector onto car's lateral axis
        cte = -math.sin(car_yaw)*dx + math.cos(car_yaw)*dy
        
        # PUBLISH ERROR FOR RL
        error_msg = Float32()
        error_msg.data = abs(cte) # Publish magnitude
        self.error_pub.publish(error_msg)


        # C. Stanley Control Law
        # Uses self.k_gain (dynamic)
        crosstrack_term = math.atan2(self.k_gain * cte, self.speed + K_SOFT)
        steering_angle = heading_error + crosstrack_term

        # Clamp limits
        steering_angle = max(min(steering_angle, 0.4), -0.4)

        # D. Publish Drive Command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = float(self.speed)
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()