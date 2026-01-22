import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import pandas as pd
import os

CSV_PATH = '/root/sim_ws/src/f1tenth_gym_ros/racelines/Spielberg_map.csv'


class RacelineVisualizer(Node):
    def __init__(self):
        super().__init__('raceline_visualizer')

        if not os.path.exists(CSV_PATH):
            self.get_logger().error('Raceline CSV not found')
            return

        df = pd.read_csv(CSV_PATH)
        self.waypoints = df[['x', 'y']].values

        self.pub = self.create_publisher(Marker, '/raceline', 1)
        self.timer = self.create_timer(1.0, self.publish_marker)

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'raceline'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.05  # line width

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = []
        for x, y in self.waypoints:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)

        self.pub.publish(marker)


def main():
    rclpy.init()
    node = RacelineVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
