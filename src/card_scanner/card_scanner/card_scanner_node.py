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

    # ── Comparison helpers ────────────────────────────────────────────────────

    def _crop(self, frame, box):
        x, y, w, h = box
        return frame[y:y+h, x:x+w]

    def _similarity(self, roi1, roi2):
        """Normalised cross-correlation between two same-size grayscale patches."""
        s = (64, 96)
        g1 = cv2.resize(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), s).astype(np.float32)
        g2 = cv2.resize(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), s).astype(np.float32)
        f1 = g1 - g1.mean();  f2 = g2 - g2.mean()
        denom = np.sqrt((f1**2).sum() * (f2**2).sum()) + 1e-8
        return float(np.sum(f1 * f2) / denom)   # -1 … +1

    def _compare(self, frame1, boxes1, frame2, boxes2):
        """
        For each card in scan1, find the best-matching card in scan2 by
        position, then measure similarity. Lowest similarity = changed card.
        Returns list of dicts sorted by similarity (worst first).
        """
        results = []
        for i, b1 in enumerate(boxes1):
            x1, y1, w1, h1 = b1
            cx1 = x1 + w1 // 2

            # match to nearest card in scan2 by horizontal centre
            if boxes2:
                b2 = min(boxes2, key=lambda b: abs((b[0]+b[2]//2) - cx1))
            else:
                results.append({"card": i+1, "sim": 0.0, "changed": True})
                continue

            roi1 = self._crop(frame1, b1)
            roi2 = self._crop(frame2, b2)
            sim  = self._similarity(roi1, roi2)
            results.append({"card": i+1, "box": b1, "sim": sim,
                             "changed": sim < 0.82})

        results.sort(key=lambda r: r["sim"])
        return results

    # ── Display loop ──────────────────────────────────────────────────────────

    def display_loop(self):
        """
        Live feed with card detection.
        Keys:
          S — save current frame as Scan 1 (before)
          C — compare current frame with Scan 1, highlight changed card
          R — reset
          Q — quit
        """
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 800, 600)
        cv2.namedWindow("Mask Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask Debug", 400, 300)

        scan1_frame  = None
        scan1_boxes  = None
        result_text  = "S=Save scan1   C=Compare   R=Reset   Q=Quit"
        result_color = (0, 220, 255)
        changed_box  = None   # box to highlight in red after compare
        prev_count   = 0

        while self._running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.033)
                continue

            cv2.imshow("Mask Debug", self.detector.get_debug_mask(frame))

            boxes = self.detector.find_cards(frame)
            disp  = frame.copy()

            # Draw all detected cards in green
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = "CARD {}".format(i)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty = max(y - 4, th + 8)
                cv2.rectangle(disp, (x, ty-th-6), (x+tw+6, ty+2), (0, 180, 0), -1)
                cv2.putText(disp, label, (x+3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Highlight changed card in red
            if changed_box is not None:
                x, y, w, h = changed_box
                cv2.rectangle(disp, (x-4, y-4), (x+w+4, y+h+4), (0, 0, 255), 5)
                cv2.putText(disp, "CHANGED!", (x, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Scan1 indicator
            if scan1_frame is not None:
                cv2.putText(disp, "SCAN1 SAVED ({} cards)".format(len(scan1_boxes)),
                            (disp.shape[1]-320, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Status bar
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(disp, result_text, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)

            cv2.imshow(self.WINDOW, disp)

            if len(boxes) != prev_count:
                print("  Detected {} card(s)".format(len(boxes)) if boxes else "  No card")
                prev_count = len(boxes)

            key = cv2.waitKey(1) & 0xFF

            # ── S: save scan 1 ────────────────────────────────────────────────
            if key == ord('s'):
                if not boxes:
                    result_text  = "No cards visible — cannot save!"
                    result_color = (0, 0, 255)
                else:
                    scan1_frame  = frame.copy()
                    scan1_boxes  = boxes[:]
                    changed_box  = None
                    result_text  = "Scan 1 saved ({} cards). Change a card, then press C.".format(len(boxes))
                    result_color = (0, 255, 0)
                    print("\n  [SCAN 1] Saved {} card(s). Change a card then press C.\n".format(len(boxes)))

            # ── C: compare with scan 1 ────────────────────────────────────────
            elif key == ord('c'):
                if scan1_frame is None:
                    result_text  = "Press S first to save Scan 1!"
                    result_color = (0, 0, 255)
                elif not boxes:
                    result_text  = "No cards visible in current frame!"
                    result_color = (0, 0, 255)
                else:
                    results = self._compare(scan1_frame, scan1_boxes, frame, boxes)
                    worst   = results[0]
                    if worst["changed"]:
                        changed_box  = worst.get("box")
                        result_text  = "CARD {} CHANGED! (sim={:.2f})".format(
                            worst["card"], worst["sim"])
                        result_color = (0, 0, 255)
                        print("\n  >>> CARD {} WAS CHANGED! similarity={:.3f}\n".format(
                            worst["card"], worst["sim"]))
                    else:
                        changed_box  = None
                        result_text  = "No change detected (best sim={:.2f})".format(
                            worst["sim"])
                        result_color = (0, 220, 100)
                        print("\n  No change detected.\n")
                    for r in results:
                        print("  Card {}: similarity={:.3f} {}".format(
                            r["card"], r["sim"],
                            "<-- CHANGED" if r.get("changed") else ""))

            # ── R: reset ──────────────────────────────────────────────────────
            elif key == ord('r'):
                scan1_frame  = None
                scan1_boxes  = None
                changed_box  = None
                result_text  = "Reset. Press S to save Scan 1."
                result_color = (0, 220, 255)
                print("  Reset.")

            # ── Q: quit ───────────────────────────────────────────────────────
            elif key == ord('q'):
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
