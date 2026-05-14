#!/usr/bin/env python3
"""
Live card scanner using Snehil's detection logic.
Shows live feed with green boxes around detected cards.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time


class CardDetector:
    """
    White cards on dark carpet — uses Otsu's automatic threshold.
    Otsu finds the optimal split between the dark floor and bright card
    pixel distributions without any hardcoded brightness value.

    Pipeline:
      1. Grayscale + Gaussian blur  — remove carpet texture noise (low-pass)
      2. Otsu threshold             — auto-separate bright card / dark floor
      3. Morphological close        — fill small holes inside the card blob
      4. Morphological open         — remove tiny noise specks
      5. Contour filter             — area, aspect ratio, solidity
    """

    MIN_CARD_AREA      = 300
    MAX_CARD_AREA_FRAC = 0.50    # up to half the frame (very close card)
    MIN_SOLIDITY       = 0.35
    MIN_ASPECT         = 0.08    # very flat card seen nearly edge-on
    MAX_ASPECT         = 0.98

    def _build_mask(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur — low-pass filter suppresses carpet grain noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Otsu automatically finds the threshold between dark floor & white card
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Closing — dilation then erosion — fills gaps inside the card blob
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        # Opening — erosion then dilation — removes small noise specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        return mask

    def find_cards(self, frame):
        """Returns list of (x, y, w, h) sorted left to right."""
        H, W  = frame.shape[:2]
        mask  = self._build_mask(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes    = []
        max_area = self.MAX_CARD_AREA_FRAC * H * W

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CARD_AREA or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 5:
                continue

            aspect = min(w, h) / max(w, h)
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            solidity = area / float(w * h)
            if solidity < self.MIN_SOLIDITY:
                continue

            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: b[0])
        return boxes

    def get_debug_mask(self, frame):
        return self._build_mask(frame)


class CardScannerNode(Node):
    WINDOW = "Card Scanner"

    def __init__(self):
        super().__init__("card_scanner_node")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        cam = self.get_parameter("camera_topic").value

        self.bridge   = CvBridge()
        self.detector = CardDetector()
        self.lock     = threading.Lock()
        self.frame    = None
        self._running = True

        self.img_sub = self.create_subscription(
            Image, cam, self._img_cb, 10)
        self.get_logger().info("Waiting for camera on {} ...".format(cam))

    def _img_cb(self, msg):
        try:
            f = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.frame = f
        except Exception as e:
            self.get_logger().error(str(e))

    def display_loop(self):
        """Runs on the main thread — live 30 fps feed with detection overlay."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 800, 600)
        cv2.namedWindow("Mask Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask Debug", 400, 300)

        prev_count = 0

        while self._running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.033)
                continue

            # debug mask — shows what the detector sees as card candidates
            debug_mask = self.detector.get_debug_mask(frame)
            cv2.imshow("Mask Debug", debug_mask)

            boxes = self.detector.find_cards(frame)
            disp  = frame.copy()

            for i, (x, y, w, h) in enumerate(boxes, start=1):
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = "CARD {}".format(i)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                ty = max(y - 4, th + 8)
                cv2.rectangle(disp, (x, ty-th-6), (x+tw+8, ty+2), (0, 200, 0), -1)
                cv2.putText(disp, label, (x+4, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            if boxes:
                status = "DETECTED: {} card(s)".format(len(boxes))
                color  = (0, 255, 0)
            else:
                status = "No card detected"
                color  = (0, 100, 255)

            cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(disp, status, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(self.WINDOW, disp)

            if len(boxes) != prev_count:
                print("  Detected {} card(s)".format(len(boxes)) if boxes else "  No card")
                prev_count = len(boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._running = False
                break

            time.sleep(0.033)


def main(args=None):
    rclpy.init(args=args)
    node = CardScannerNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.display_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
