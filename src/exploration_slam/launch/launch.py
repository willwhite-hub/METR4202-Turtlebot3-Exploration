from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


# ros2 launch exploration_slam launch.py --ros-args -p world:=/path/to/your/world_file.world
def generate_launch_description():
    print("Launching exploration_slam package")
    launch_file_dir = os.path.join(
        get_package_share_directory("exploration_slam"), "launch"
    )

    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")

    # Declare world argument
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=os.path.join(
            get_package_share_directory("exploration_slam"),
            "worlds",
            "metr4202_final_demo_no_aruco.world",
        ),
        description="Name of the world to be used.",
    )

    # Get the world file name
    world_file = LaunchConfiguration("world")

    # SLAM Toolbox Node
    slam_toolbox_cmd = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"max_laser_range": 3.5},
            {"options.num_threads": 12}
        ],
    )

    # Get Rviz config file path
    rviz_config_dir = os.path.join(
        get_package_share_directory("exploration_slam"),
        "rviz",
        "exploration_config.rviz",
    )

    # Rviz Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_dir],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Node for the exploration process
    search_cmd = Node(
        package="exploration_slam",
        executable="search",
        name="search",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        ),
    )

    spawn_turtlebot3_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )


    return LaunchDescription(
        [
            slam_toolbox_cmd,
            search_cmd,
            robot_state_publisher_cmd,
            spawn_turtlebot3_cmd,
            world_arg,
            rviz_node,
            gzserver_cmd,
            gzclient_cmd,
        ]
    )
