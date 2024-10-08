import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import numpy as np
import heapq
import random
import time
from scipy.interpolate import splprep, splev

lookahead_distance = 0.5  # Lookahead distance for the Pure Pursuit algorithm
speed = 0.25  # Speed of the robot
expansion_size = (
    3  # Expansion size for the costmap - expands the obstacles in the occupancy grid
)
target_error = 0.15  # Target error for reaching the goal
robot_r = 0.5  # Robot radius for local control


class Explorer(Node):
    def __init__(self):
        super().__init__("explorer")
        self.occupancy_grid_sub = self.create_subscription(
            OccupancyGrid, "map", self.map_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.scan_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.robot_position = (0, 0)
        self.robot_orientation = 0.0
        self.frontiers = []
        self.visited = set()  # Track visited cells
        self.occupancy_grid = None
        self.obstacles = []

    def map_callback(self, msg):
        self.occupancy_grid = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )
        self.frontiers = find_frontiers(msg)
        self.get_logger().info(
            f"Received occupancy grid. Found frontiers: {self.frontiers}"
        )
        self.explore()

    def odom_callback(self, msg):
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        _, _, self.robot_orientation = self.euler_from_quaternion(orientation_q)
        

    def scan_callback(self, msg):
        self.obstacles = []
        closest_obstacle_distance = float("inf")
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, range in enumerate(msg.ranges):
            if range < msg.range_max:
                angle = angle_min + i * angle_increment
                x = range * math.cos(angle)
                y = range * math.sin(angle)
                # Transform to global coordinates
                global_x = (
                    self.robot_position[0]
                    + x * math.cos(self.robot_orientation)
                    - y * math.sin(self.robot_orientation)
                )
                global_y = (
                    self.robot_position[1]
                    + x * math.sin(self.robot_orientation)
                    + y * math.cos(self.robot_orientation)
                )
                self.obstacles.append((global_x, global_y))
                if range < closest_obstacle_distance:
                    closest_obstacle_distance = range

        # Adjust velocity based on the closest obstacle
        if closest_obstacle_distance < robot_r:
            self.avoid_obstacle(closest_obstacle_distance)

    def avoid_obstacle(self, obstacle_distance):
        cmd = Twist()
        if obstacle_distance < robot_r:  # If obstacle is within robot radius
            cmd.linear.x = 0.0  # Stop forward motion
            # Determine the direction to turn based on laser scan readings
            if len(self.obstacles) > 1:
                # Check the angles of detected obstacles to decide which way to turn
                left_obstacle = [ob for ob in self.obstacles if ob[0] < self.robot_position[0]]
                right_obstacle = [ob for ob in self.obstacles if ob[0] > self.robot_position[0]]
                
                if len(left_obstacle) > 0:
                    cmd.angular.z = -0.5  # Turn right
                elif len(right_obstacle) > 0:
                    cmd.angular.z = 0.5  # Turn left
                else:
                    cmd.angular.z = 0.5  # Default turn (can adjust this logic)
            else:
                cmd.angular.z = 0.5  # Turn in a default direction if only one obstacle
        self.cmd_pub.publish(cmd)

    def detect_obstacle_in_path(self, point):
        """Check if there is an obstacle near the given point."""
        x, y = point
        for ox, oy in self.obstacles:
            distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if distance < robot_r:  # If an obstacle is within the robot's radius
                return True
        return False

    def euler_from_quaternion(self, quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        quat: geometry_msgs.msg.Quaternion
        """
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def explore(self):
        if not self.frontiers:
            self.get_logger().info("No frontiers found. Exploration stopped.")
            return

        # Remove visited frontiers
        self.frontiers = [f for f in self.frontiers if f not in self.visited]

        prioritized_frontiers = self.prioritize_frontiers(
            self.frontiers, self.robot_position
        )

        # Check if prioritized frontiers are available
        if not prioritized_frontiers:
            self.get_logger().info(
                "No prioritized frontiers available. Exploration complete or switching to random exploration."
            )
            self.random_explore()  # Switch to random exploration or end exploration
            return

        max_attempts = 5  # Limit the number of replanning attempts
        attempts = 0

        # Try multiple frontiers if one fails
        while attempts < max_attempts and prioritized_frontiers:
            target = prioritized_frontiers[0]
            self.get_logger().info(f"Exploring. Target frontier: {target}")

            # Plan a path to the target
            path = astar(
                self.occupancy_grid, self.robot_position, target, self.obstacles
            )

            if path:
                self.get_logger().info(f"Found path to target: {path}")
                for point in path:
                    if point not in self.visited:
                        self.visited.add(point)

                        # Obstacle detection along the path
                        if self.detect_obstacle_in_path(point):
                            self.get_logger().info("Obstacle detected, adjusting heading...")
                            self.adjust_heading()
                        else:
                            # Proceed with movement
                            cmd = Twist()
                            cmd.linear.x = speed
                            self.cmd_pub.publish(cmd)
                            self.get_logger().info(f"Moving to {point}")
                            return  # Exit after successful move

            else:
                self.get_logger().info(
                    "No valid path found. Replanning with a new frontier."
                )
                attempts += 1

            # If the current target fails, remove it and re-prioritize
            prioritized_frontiers.pop(0)

        # If all attempts fail, switch to random exploration
        if attempts >= max_attempts or not prioritized_frontiers:
            self.get_logger().info(
                "Too many failed attempts. Switching to random exploration."
            )
            self.random_explore()

    def adjust_heading(self):
        # Rotate the robot slightly to avoid the obstacle
        cmd = Twist()
        cmd.angular.z = 0.5  # Adjust the rotation speed as needed
        self.cmd_pub.publish(cmd)
        time.sleep(1)  # Rotate for a short duration
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)    
        
    def follow_path(self, path):
        for point in path:
            if point not in self.visited:
                self.visited.add(point)
                cmd = Twist()
                cmd.linear.x = speed
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"Moving to {point}")

                # Replan if obstacles detected
                if self.detect_obstacle_in_path(point):
                    self.get_logger().info("Obstacle detected, replanning...")
                    self.explore()
                    break

    def random_explore(self):
        self.get_logger().info("Attempting random exploration.")
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(neighbors)
        for i, j in neighbors:
            neighbor = (
                int(self.robot_position[0] + i),
                int(self.robot_position[1] + j),
            )
            if neighbor not in self.visited and self.is_free(neighbor):
                self.visited.add(neighbor)
                cmd = Twist()
                cmd.linear.x = speed
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"Moving to random neighbor: {neighbor}")
                break

    def is_free(self, cell):
        x, y = int(cell[0]), int(cell[1])
        return (
            0 <= x < self.occupancy_grid.shape[0]
            and 0 <= y < self.occupancy_grid.shape[1]
            and self.occupancy_grid[x, y] == 0
        )

    def prioritize_frontiers(self, frontiers, robot_position):
        """
        Prioritizes frontiers based on their distance from the robot's current position: the closer the frontier, the 
        higher the priority.
        Args:
            frontiers (list of tuple): A list of frontiers where each frontier is represented as a tuple (x, y).
            robot_position (tuple): The current position of the robot represented as a tuple (x, y).
        Returns:
            list of tuple: A list of frontiers sorted by their distance from the robot's current position in ascending order.
        """

        prioritized_frontiers = []
        for frontier in frontiers:
            distance = math.sqrt(
                (frontier[0] - robot_position[0]) ** 2
                + (frontier[1] - robot_position[1]) ** 2
            )
            heapq.heappush(prioritized_frontiers, (distance, frontier))
        return [
            heapq.heappop(prioritized_frontiers)[1]
            for _ in range(len(prioritized_frontiers))
        ]


def find_frontiers(occupancy_grid):
    frontiers = []
    width = occupancy_grid.info.width
    height = occupancy_grid.info.height
    data = np.array(occupancy_grid.data).reshape((height, width))

    frontiers = []
    for y in range(height):
        for x in range(width):
            if data[y, x] == -1:  # Unknown cell
                if (
                    (x > 0 and data[y, x - 1] >= 0)
                    or (x < width - 1 and data[y, x + 1] >= 0)
                    or (y > 0 and data[y - 1, x] >= 0)
                    or (y < height - 1 and data[y + 1, x] >= 0)
                ):
                    frontiers.append((x, y))
    return frontiers


def prioritize_frontiers(frontiers, robot_position):
    prioritized_frontiers = []
    for frontier in frontiers:
        distance = math.sqrt(
            (frontier[0] - robot_position[0]) ** 2
            + (frontier[1] - robot_position[1]) ** 2
        )
        heapq.heappush(prioritized_frontiers, (distance, frontier))
    return [
        heapq.heappop(prioritized_frontiers)[1]
        for _ in range(len(prioritized_frontiers))
    ]


def astar(array, start, goal, obstacles):
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        close_set.add(current)

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            # Ensure neighbor coordinates are integers
            if not (isinstance(neighbor[0], int) and isinstance(neighbor[1], int)):
                continue

            # Check if the neighbor is within the bounds of the grid
            if not (
                0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]
            ):
                continue

            # Check if the neighbor is an obstacle
            if array[neighbor[0], neighbor[1]] == 1:
                continue

            # Check if the neighbor is too close to a detected obstacle
            if any(
                math.sqrt((neighbor[0] - ox) ** 2 + (neighbor[1] - oy) ** 2) < robot_r
                for ox, oy in obstacles
            ):
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(
                neighbor, float("inf")
            ):
                continue

            if tentative_g_score < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    closest_node = min(close_set, key=lambda node: heuristic(node, goal), default=None)
    if closest_node:
        return reconstruct_path(came_from, closest_node)

    return False


def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]

        tck, _ = splprep([x, y], s=0)  # "u" is not accessed, so we use "_" to ignore it

        u_new = np.linspace(0, 1, sn)
        rx, ry = splev(u_new, tck)

        path = [(rx[i], ry[i]) for i in range(len(rx))]
    except Exception as e:
        print(f"An error occurred during bspline_planning: {e}")
        path = array.tolist()
    return path


def follow_smooth_path(self, path):
    smooth_path = bspline_planning(path, 100)
    for point in smooth_path:
        cmd = Twist()
        cmd.linear.x = speed
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"Moving to smooth path point: {point}")


def fGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    top_groups = [group for group in sorted_groups if len(group[1]) > 2]
    return top_groups[:5]


def main(args=None):
    rclpy.init(args=args)
    explorer = Explorer()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
