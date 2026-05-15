# Card Scanner (ROS 2)

## Overview
This project runs a ROS 2 node that detects playing cards in a camera feed and highlights the card that changed after a flip. It is designed for a TurtleBot3 Burger with a camera and uses OpenCV for detection and comparison.

## What You Did In This Project
- Built a ROS 2 package (`card_scanner`) with a Python node and launch file.
- Implemented a card detection pipeline that segments bright cards from a dark floor using Otsu thresholding and morphological filtering.
- Added a bottom-of-frame scan ROI to focus on the tabletop area.
- Added live visualization with a main display and a dedicated "Mask Debug" window.
- Implemented a scan/compare workflow to detect the changed card by comparing a saved template against the live frame.
- Added a flip test that compares normal, 180-degree rotation, and horizontal/vertical flips to detect a flipped card.
- Provided helper scripts to install dependencies and build/run the node.

## How It Works (High Level)
1. The node subscribes to the camera image topic and keeps the latest frame.
2. Cards are detected using:
   - Grayscale + Gaussian blur
   - Otsu threshold
   - Morphological close/open
   - Contour filters (area, aspect ratio, solidity)
3. Press `S` to save the "Scan 1" card templates.
4. After a card is flipped, press `C` to compare current cards with saved templates.
5. The card with the highest difference gap is marked as changed.

## Requirements
- ROS 2 Humble
- System packages:
  - `ros-humble-cv-bridge`
  - `ros-humble-v4l2-camera`
  - `tesseract-ocr` (optional)
  - `python3-pip`
- Python packages:
  - `opencv-python`
  - `pytesseract`
  - `numpy`

## Setup
```bash
./install_dependencies.sh
```

## Build and Run
```bash
./build_and_run.sh
```

Or manually:
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select card_scanner
source install/setup.bash
ros2 launch card_scanner card_scanner.launch.py
```

## Controls
- `S`: Save Scan 1 templates
- `C`: Compare current view to Scan 1
- `R`: Reset
- `Q`: Quit

## Notes
- Detection works best with bright cards on a darker floor.
- The scan region focuses on the lower portion of the frame to reduce noise.
- Debug artifacts for crops are written to `/tmp`.

## Project Files
- `src/card_scanner/card_scanner/card_scanner_node.py`: Main ROS 2 node
- `src/card_scanner/launch/card_scanner.launch.py`: Launch file
- `build_and_run.sh`: Build + launch helper
- `install_dependencies.sh`: Dependency installer
