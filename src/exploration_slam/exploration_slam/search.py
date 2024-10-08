

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from scipy.interpolate import splprep, splev
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq
import math
import random
import threading
import time


lookahead_distance = 0.1 # Lookahead distance for the Pure Pursuit algorithm
speed = 0.3  # Speed of the robot
expansion_size = 2  # Expansion size for the costmap - expands the obstacles in the occupancy grid
target_error = 0.5  # Target error for reaching the goal 
robot_r = 0.4  # Robot radius for local control

pathGlobal = 0  # Global path for exploration; 0 means no path, -1 means exploration completed


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).
    Args:
        x (float): x-component of the quaternion.
        y (float): y-component of the quaternion.
        z (float): z-component of the quaternion.
        w (float): w-component of the quaternion.
    Returns:
        tuple: A tuple containing the roll, pitch, and yaw angles in radians.
    """
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def heuristic(a, b):
    """
    Calculates the Euclidean distance between two points a and b.
    Args:
        a, b (tuple): Coordinates of two points (x, y).
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.hypot(b[0] - a[0], b[1] - a[1])

def astar(array, start, goal):
    """
    Perform A* search algorithm to find the shortest path from start to goal in a 2D grid.

    Args:
        array (numpy.ndarray): 2D grid where 0 is free cell, 1 is obstacle, and 2 adds a penalty.
        start (tuple): Starting coordinates (x, y).
        goal (tuple): Goal coordinates (x, y).

    Returns:
        list: Coordinates representing the path from start to goal, including both.
              If no path is found, returns the closest path to the goal.
              If no path is possible, returns False.
    """
    
    def reconstruct_path(came_from, current):
        """
        Reconstructs the path from start to goal using the 'came_from' map.
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path[::-1]  # Reverse the path

    # 8 possible movements: cardinal directions + diagonals
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # Initialize structures for A* search
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))  # Priority queue (f_score, node)
    
    came_from = {}  # Track optimal path
    gscore = {start: 0}  # Cost from start to each node
    fscore = {start: heuristic(start, goal)}  # Estimated total cost from start to goal
    closed_set = set()  # Explored nodes

    while open_set:
        current = heapq.heappop(open_set)[1]  # Node with the lowest f_score

        if current == goal:
            return reconstruct_path(came_from, current)  # Goal reached, return path

        closed_set.add(current)  # Mark current node as explored

        # Explore neighbors of the current node
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if neighbor is within bounds and not an obstacle
            if not (0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]):
                continue
            if array[neighbor[0], neighbor[1]] == 1:  # Obstacle check
                continue

            # Calculate tentative g_score
            tentative_gscore = gscore[current] + heuristic(current, neighbor)

            # Apply additional penalty for cells marked as '10' (higher traversal cost)
            if array[neighbor[0], neighbor[1]] == 10:
                tentative_gscore += 5  # Penalty for proximity to obstacles
            
            # Apply penalty for revisiting nodes
            if neighbor in closed_set:
                tentative_gscore += 5

            # If the neighbor is already in the closed set and the new score isn't better, skip it
            if neighbor in closed_set and tentative_gscore >= gscore.get(neighbor, float('inf')):
                continue

            # If a better path to the neighbor is found, update the scores and path
            if tentative_gscore < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))

    # If goal is unreachable, return the path to the closest explored node
    closest_node = min(closed_set, key=lambda node: heuristic(node, goal), default=None)
    if closest_node:
        return reconstruct_path(came_from, closest_node)

    return False  # No valid path or closest node found


def bspline_planning(array, sn):
    """
    Perform B-spline interpolation to smooth a given path.
    Args:
        array (list of tuple): The input path as a list of (x, y) coordinates.
        sn (int): The number of points in the smoothed path.
    Returns:
        list of tuple: The smoothed path as a list of (x, y) coordinates.
    """
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]

        # Check if there are enough points to fit a cubic spline
        if len(x) < 4:
            print("Not enough points to fit a cubic spline")
            return array.tolist()

        # Fit B-spline to the path
        tck, u = splprep([x, y], s=0)

        # Generate new points along the B-spline
        u_new = np.linspace(0, 1, sn)
        rx, ry = splev(u_new, tck)

        path = [(rx[i], ry[i]) for i in range(len(rx))]
    except Exception as e:
        print(f"An error occurred during bspline_planning: {e}")
        path = array.tolist()
    return path


def pure_pursuit(current_x, current_y, current_heading, path, index):
    """
    Implements the Pure Pursuit algorithm to follow a given path.
    Args:
        current_x (float): The current x-coordinate of the robot.
        current_y (float): The current y-coordinate of the robot.
        current_heading (float): The current heading (orientation) of the robot in radians.
        path (list of tuples): The path to follow, represented as a list of (x, y) coordinates.
        index (int): The current index in the path from which to start searching for the lookahead point.
    Returns:
        tuple: A tuple containing:
            - v (float): The desired speed of the robot.
            - desired_steering_angle (float): The desired steering angle in radians.
            - index (int): The updated index in the path.
    """

    global lookahead_distance
    v = speed
    closest_point = None

    # Find the lookahead point
    for i in range(index, len(path)):
        x, y = path[i]
        distance = math.hypot(current_x - x, current_y - y)
        if distance > lookahead_distance:
            closest_point = (x, y)
            index = i
            break

    if closest_point is None:
        closest_point = path[-1]
        index = len(path) - 1

    target_heading = math.atan2(
        closest_point[1] - current_y, closest_point[0] - current_x
    )
    desired_steering_angle = target_heading - current_heading

    # Normalize the steering angle to the range [-pi, pi]
    desired_steering_angle = (desired_steering_angle + math.pi) % (
        2 * math.pi
    ) - math.pi

    # Limit the steering angle to avoid sharp turns
    if abs(desired_steering_angle) > math.pi / 6:
        desired_steering_angle = math.copysign(math.pi / 4, desired_steering_angle)
        v = 0.0

    return v, desired_steering_angle, index


def frontierB(matrix):
    """
    This function finds the frontiers in the map.
    Args:
        matrix (numpy.ndarray): The occupancy grid map where 0.0 represents free space and negative values represent unexplored space.
    Returns:
        numpy.ndarray: The modified occupancy grid with frontiers marked as 2.
    """
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0.0:
                if (
                    (i > 0 and matrix[i - 1][j] < 0)
                    or (i < rows - 1 and matrix[i + 1][j] < 0)
                    or (j > 0 and matrix[i][j - 1] < 0)
                    or (j < cols - 1 and matrix[i][j + 1] < 0)
                ):
                    matrix[i][j] = 2
    return matrix


def assign_groups(matrix):
    """
    Assigns groups to elements in a matrix based on a depth-first search (DFS) algorithm.
    Args:
        matrix (list of list of int): A 2D list representing the matrix where groups are to be assigned(from dfs()).
    Returns:
        tuple: A tuple containing the modified matrix and a dictionary of groups.
    """
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups


def dfs(matrix, i, j, group, groups):
    """
    Perform a depth-first search on a matrix to group connected cells. This function is used to assign groups to 
    elements in the matrix, and is used in assign_groups(). 

    Args:
        matrix (list of list of int): The 2D grid to search.
        i (int): The current row index.
        j (int): The current column index.
        group (int): The current group identifier.
        groups (dict): A dictionary to store groups of connected cells.

    Returns:
        int: The next group identifier.
    """
    stack = [(i, j)]
    while stack:
        ci, cj = stack.pop()
        if ci < 0 or ci >= len(matrix) or cj < 0 or cj >= len(matrix[0]):
            continue
        if matrix[ci][cj] != 2:
            continue
        if group in groups:
            groups[group].append((ci, cj))
        else:
            groups[group] = [(ci, cj)]
        matrix[ci][cj] = 0
        stack.extend(
            [
                (ci + 1, cj),
                (ci - 1, cj),
                (ci, cj + 1),
                (ci, cj - 1),
                (ci + 1, cj + 1),
                (ci - 1, cj - 1),
                (ci - 1, cj + 1),
                (ci + 1, cj - 1),
            ]
        )
    return group + 1


def fGroups(groups):
    """
    Sorts groups by the number of elements in descending order and returns the top five groups
    with more than two elements.
    Args:
        groups (dict): A dictionary where keys are group identifiers and values are lists of group members.
    Returns:
        list: A list of the top five groups (as tuples) with more than two members, sorted by size in descending order.
    """

    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:5] if len(g[1]) > 2]
    return top_five_groups


def calculate_centroid(x_coords, y_coords):
    """
    Calculates the centroid of a group of points.
    Args:
        x_coords (list of float): The x-coordinates of the points.
        y_coords (list of float): The y-coordinates of the points.
    Returns:
        tuple: The centroid of the group as a tuple (x, y).
    """
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid


def FindClosestGroup(matrix, groups, current, resolution, originX, originY, explored):
    """
    Finds the closest group of points from the current position using the A* algorithm,
    prioritizing exploring new areas.

    Args:
        matrix (list of list of int): The grid representing the map.
        groups (list of tuples): A list of groups where each group is a tuple containing
                                 a group identifier and a list of points (tuples) in that group.
        current (tuple): The current position as a tuple (x, y).
        resolution (float): The resolution of the grid.
        originX (float): The x-coordinate of the origin.
        originY (float): The y-coordinate of the origin.
        explored (set of tuples): A set of points that have been explored.

    Returns:
        list of tuple: The path to the closest group of points, where each point in the path
                       is a tuple (x, y) in world coordinates.
    """

    targetP = None
    distances = []
    paths = []
    scores = []
    max_score = -1  # max score index

    for i, (group_id, points) in enumerate(groups):
        middle = calculate_centroid([p[0] for p in points], [p[1] for p in points])
        path = astar(matrix, current, middle)

        # Calculate world coordinates
        path = [
            (p[1] * resolution + originX, p[0] * resolution + originY) for p in path
        ]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)

        # Score calculation prioritizing unexplored areas
        unexplored_count = sum(1 for p in points if p not in explored)
        if total_distance > 0:
            score = (unexplored_count / len(points)) / total_distance
        else:
            score = float("inf")  # prioritize if directly at the group

        scores.append(score)

        # Update max_score if this group's score is better
        if max_score == -1 or score > scores[max_score]:
            max_score = i

    # Select target path based on the highest score
    if max_score != -1:
        targetP = paths[max_score]
    else:
        # Fallback in case of no valid groups (shouldn't happen in normal cases)
        index = random.randint(0, len(groups) - 1)
        target = groups[index][1]
        random_point = target[random.randint(0, len(target) - 1)]
        path = astar(matrix, current, random_point)
        targetP = [
            (p[1] * resolution + originX, p[0] * resolution + originY) for p in path
        ]

    return targetP


def pathLength(path):
    """
    Calculate the total length of a path.
    Args:
        path (list of tuple): The path as a list of (x, y) coordinates.
    Returns:
        float: The total length of the path.
    """
    # Check if path is empty
    if not path:
        return float('inf')
    
    for i in range(len(path)):
        path[i] = (path[i][0], path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:, 0], differences[:, 1])
    total_distance = np.sum(distances)
    return total_distance


def costmap(data, width, height, resolution):
    """
    Generates a costmap by expanding walls in the given occupancy grid data.
    Args:
        data (list or array-like): The occupancy grid data where each cell value represents the occupancy state.
        width (int): The width of the occupancy grid.
        height (int): The height of the occupancy grid.
        resolution (float): The resolution of the occupancy grid.
    Returns:
        numpy.ndarray: The modified occupancy grid with expanded walls, scaled by the resolution.
    """

    data = np.array(data).reshape(height, width)
    wall = np.where(data == 100)
    for i in range(-expansion_size, expansion_size + 1):
        for j in range(-expansion_size, expansion_size + 1):
            if i == 0 and j == 0:
                continue
            x = wall[0] + i
            y = wall[1] + j
            x = np.clip(x, 0, height - 1)
            y = np.clip(y, 0, width - 1)
            data[x, y] = 100
    data = data * resolution
    return data

def exploration(data, width, height, resolution, column, row, originX, originY):
    """
    Perform exploration on a given map data to find the next path for the robot.
    Args:
        data (list of list of int): The occupancy grid map data.
        width (int): The width of the map.
        height (int): The height of the map.
        resolution (float): The resolution of the map.
        column (int): The column index of the robot's current position.
        row (int): The row index of the robot's current position.
        originX (float): The X coordinate of the map's origin.
        originY (float): The Y coordinate of the map's origin.
    Returns:
        None: The function updates the global variable `pathGlobal` with the next path or -1 if no path is found.
    """

    global pathGlobal
    data = costmap(data, width, height, resolution)
    data[row][column] = 0
    data[data > 5] = 1
    data = frontierB(data)
    data, groups = assign_groups(data)
    groups = fGroups(groups)
    if len(groups) == 0:
        path = -1
    else:
        data[data < 0] = 1
        path = FindClosestGroup(
            data, groups, (row, column), resolution, originX, originY, explored=set()
        )
        if path is not None:
            path = bspline_planning(path, len(path) * 5)
        else:
            path = -1
    pathGlobal = path
    return


def localControl(scan):
    """
    Determines the local control commands for a robot based on scan data.
    This function processes scan data to decide the linear and angular velocities
    (v and w) for a robot. It checks for obstacles within a certain range and
    adjusts the velocities accordingly to avoid collisions.
    Args:
        scan (list of float): A list of distance measurements from a 360-degree
                              LIDAR scan. Each element represents the distance
                              to an obstacle at a specific angle.
    Returns:
        tuple: A tuple containing the linear velocity (v) and angular velocity (w).
               If an obstacle is detected within the robot's radius in the front
               60 degrees, the robot will turn left. If an obstacle is detected
               within the robot's radius in the rear 60 degrees, the robot will
               turn right. If no obstacles are detected within these ranges,
               both velocities will be None.
    """
    v = speed  # Maintain forward speed unless there's an obstacle
    w = 0.0

     # Define distance thresholds for obstacle avoidance
    front_threshold = 0.4
    back_threshold = 0.4
    obstacle_proximity = 0.4

    # Check for obstacles in the front (angles 0 to 60)
    front_obstacle = any(scan[i] < front_threshold for i in range(30))
    while front_obstacle:
        v = 0.0  # Stop if an obstacle is detected ahead
        w = -0.4 # Proportional turning  # Turn left to avoid obstacle
        break
    # Check for obstacles in the back (angles 300 to 360)
    back_obstacle = any(scan[i] < back_threshold for i in range(315, 345))
    while back_obstacle:
        v = 0.0  # Stop if an obstacle is detected behind
        w = 0.4  # Turn right to avoid obstacle
        break
    # Encourage exploration by adjusting speed based on proximity to obstacles
    if any(scan[i] < obstacle_proximity for i in range(len(scan))):
        v *= 0.5  # Reduce speed to 50% if any obstacle is too close
    else:
        v = speed  # Reset to normal speed if no obstacles are too close

    return v, w

class NavigationControl(Node):
    def __init__(self):
        super().__init__("Exploration")
        
        # Subscribers
        self.subscription = self.create_subscription(
            OccupancyGrid, "map", self.map_callback, 10
        )
        self.subscription = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )
        self.subscription = self.create_subscription(
            LaserScan, "scan", self.scan_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 10
        )
        
        # Publisher
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.get_logger().info("EXPLORATION MODE ACTIVE")
        
        # Exploration variables
        self.exploration_flag = True  # Exploration flag
        threading.Thread(target=self.get_target).start()
        self.explored = set()
        self.current_velocity = 0.0
        self.last_position = None
        self.current_position = None
        self.last_time = None
        self.is_stuck = False

        # Stuck detection variables
        self.movement_threshold = 0.05 # Movement threshold for stuck detection
        self.stuck_time_threshold = 2 # Time threshold for stuck detection
        self.velocity_threshold = 0.01 # Velocity threshold for stuck detection

    def get_target(self):
        """
        Perform exploration and navigation based on sensor data. Continuously checks for map, odometry, and scan data
        availability. Once available, performs exploration and sets a new target if needed. Navigates towards the target
        using local control or pure pursuit algorithms. Sets a new exploration target upon reaching the current target.
        """
        twist = Twist()
        while True:
            v, w = 0.0, 0.0
            if not all(
                hasattr(self, attr) for attr in ["map_data", "odom_data", "scan_data"]
            ):
                time.sleep(0.1)
                continue

            if self.exploration_flag:
                if isinstance(pathGlobal, int) and pathGlobal == 0:
                    column = int((self.x - self.originX) / self.resolution)
                    row = int((self.y - self.originY) / self.resolution)
                    exploration(
                        self.data,
                        self.width,
                        self.height,
                        self.resolution,
                        column,
                        row,
                        self.originX,
                        self.originY,
                    )
                    self.path = pathGlobal
                else:
                    self.path = pathGlobal

                # Check for completion of exploration or empty path
                if not self.path or (isinstance(self.path, int) and self.path == -1):
                    self.get_logger().info("EXPLORATION COMPLETED or PATH EMPTY")
                    rclpy.shutdown()
                else:
                    # Ensure the target is not already explored
                    while True:
                        target_position = (self.path[-1][0], self.path[-1][1])
                        if target_position not in self.explored:
                            # Only set the target if it hasn't been explored
                            self.c = int((self.path[-1][0] - self.originX) / self.resolution)
                            self.r = int((self.path[-1][1] - self.originY) / self.resolution)
                            self.exploration_flag = False
                            self.i = 0
                            self.get_logger().info("NEW TARGET SET")
                            t = pathLength(self.path) / speed - 0.2
                            self.t = threading.Timer(t, self.target_callback)
                            self.t.start()
                            break
                        else:
                            self.get_logger().info("TARGET ALREADY EXPLORED, FINDING NEW TARGET")
                            # Call exploration function or logic to find a new target
                            column = int((self.x - self.originX) / self.resolution)
                            row = int((self.y - self.originY) / self.resolution)
                            exploration(
                                self.data,
                                self.width,
                                self.height,
                                self.resolution,
                                column,
                                row,
                                self.originX,
                                self.originY,
                            )
                            self.path = pathGlobal  # Update the path again after exploring

            else:
                v, w = localControl(self.scan)
                if v is None or w is None:
                    v, w, self.i = pure_pursuit(
                        self.x, self.y, self.yaw, self.path, self.i
                    )

                if (
                    abs(self.x - self.path[-1][0]) < target_error
                    and abs(self.y - self.path[-1][1]) < target_error
                ):
                    v, w = 0.0, 0.0
                    self.exploration_flag = True
                    self.get_logger().info("TARGET REACHED")
                    if self.t.is_alive():
                        self.t.join()

            twist.linear.x = v
            twist.angular.z = w
            self.publisher.publish(twist)
            time.sleep(0.1)
    
    def check_if_stuck(self):
        # Wait until we have initial position data
        if self.last_position is None:
            self.last_position = self.current_position
            self.last_time = time.time()
            return
        
        # Calculate distance moved
        distance_moved = math.sqrt(
            (self.current_position[0] - self.last_position[0]) ** 2 + 
            (self.current_position[1] - self.last_position[1]) ** 2
        )

        # If the robot has moved less than the threshold, check if stuck.
        if distance_moved < self.movement_threshold and self.current_velocity < self.velocity_threshold:
            if time.time() - self.last_time > self.stuck_time_threshold:
                self.is_stuck = True
                self.get_logger().info("Robot is stuck!")
            else:
                self.is_stuck = False
        else:
            # Reset position and time since the robot has moved
            self.last_position = self.current_position
            self.last_time = time.time()
            self.is_stuck = False

    def target_callback(self):
        exploration(
            self.data,
            self.width,
            self.height,
            self.resolution,
            self.c,
            self.r,
            self.originX,
            self.originY,
        )

    def scan_callback(self, msg):
        self.scan_data = msg
        self.scan = msg.ranges

    def map_callback(self, msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.check_if_stuck()
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def cmd_vel_callback(self, msg):
        self.current_velocity = math.sqrt(
            msg.linear.x ** 2 + msg.linear.y ** 2
        )


def main(args=None):
    rclpy.init(args=args)
    navigation_control = NavigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
