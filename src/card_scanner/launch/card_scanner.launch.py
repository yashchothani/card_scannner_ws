from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # Camera runs on the ROBOT — not started here
        Node(
            package="card_scanner",
            executable="card_scanner_node.py",
            name="card_scanner_node",
            output="screen",
            parameters=[{
                "camera_topic":  "/camera/image_raw",
                "cmd_vel_topic": "/cmd_vel",
                "rotate_speed":  1.2,
                "scan_frames":   10,      # 10 frames at 4fps = ~2.5 seconds
                "show_window":   False,
            }],
        ),
    ])
