import rclpy
from rclpy import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import message_filters
from search import euler_from_quaternion

class LaserOdomSync(Node):
    def __init__(self):
        super().__init__('laser_odom_sync_node')
        
        # Synchronise laser and odometry data
        laser_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        odom_sub = message_filters.Subscriber(self, Odometry, '/odom')

        # ApproximateTimeSynchronizer ensures messages are synchronized even if their timestamps are slightly off
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [laser_sub, odom_sub], 
            queue_size=10, 
            slop=0.1  # Tolerance in seconds between the two messages
        )
    
    def sync_callback(self, laser_scan_msg, odom_msg):
        # Process the synchronised messages.
        self.get_logger().info('Laser and Odometry data synchronized!')

        self.laser_scan_msg = laser_scan_msg
        self.odom_msg = odom_msg
        
        # Store data for navigation
        self.scan = laser_scan_msg.ranges
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w,
        )
        
def main(args=None):
    rclpy.init(args=args)
    node = LaserOdomSync()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()