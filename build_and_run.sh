#!/bin/bash
set -e
cd ~/card_scanner_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select card_scanner
source install/setup.bash
echo "==> Launching..."
ros2 launch card_scanner card_scanner.launch.py