"""Xbox controller -> Yanshee: joy_node + teleop_twist_joy + yanshee_driver.

  ros2 launch yanshee_teleop teleop.launch.py [robot_ip:=192.168.1.62] [dry_run:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("yanshee_teleop"), "config", "xbox.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.62"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        DeclareLaunchArgument("joy_dev", default_value="0", description="joystick index (/dev/input/jsN)"),
        Node(package="joy", executable="joy_node", name="joy_node",
             parameters=[params, {"device_id": LaunchConfiguration("joy_dev")}]),
        Node(package="teleop_twist_joy", executable="teleop_node", name="teleop_twist_joy_node",
             parameters=[params]),
        Node(package="yanshee_teleop", executable="driver", name="yanshee_driver", output="screen",
             parameters=[params, {"robot_ip": LaunchConfiguration("robot_ip"),
                                  "dry_run": LaunchConfiguration("dry_run")}]),
    ])
