#!/bin/bash
set -e
echo "==> Installing ROS2 packages..."
sudo apt-get update
sudo apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-v4l2-camera \
    tesseract-ocr \
    python3-pip

echo "==> Installing Python packages..."
pip3 install opencv-python pytesseract numpy

echo "✅  Done!"