from setuptools import find_packages, setup

package_name = "exploration_slam"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/launch.py",
                "launch/robot_state_publisher.launch.py",
                "launch/spawn_turtlebot3.launch.py",
            ],
        ),
        ("share/" + package_name + "/rviz", ["rviz/exploration_config.rviz"]),
        (
            "share/" + package_name + "/worlds",
            ["worlds/metr4202_final_demo_no_aruco.world"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ethan",
    maintainer_email="ethan.pinto@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "search = exploration_slam.search:main",
        ],
    },
)
